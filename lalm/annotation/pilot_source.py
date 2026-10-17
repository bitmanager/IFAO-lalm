"""Bounded SOVA candidate snapshot from production S3; never a training release.

Run with an ingest-only env file and CA. Outputs local immutable audio + provenance;
source labels remain separate from model inputs. Existing encoded-hash guards are
checked; PCM/group isolation remains a publication gate.
"""
import argparse
import hashlib
import io
import json
import math
import shlex
from pathlib import Path

import boto3
from botocore.config import Config
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import webrtcvad


class S3File(io.RawIOBase):
    def __init__(self, client, bucket, key):
        self.client, self.bucket, self.key = client, bucket, key
        head = client.head_object(Bucket=bucket, Key=key)
        self.size, self.etag, self.pos = head['ContentLength'], head['ETag'], 0
        self.bytes_read = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        return self.pos
    def read(self, n=-1):
        if self.pos >= self.size: return b''
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if not n: return b''
        body = self.client.get_object(Bucket=self.bucket, Key=self.key, IfMatch=self.etag,
             Range=f'bytes={self.pos}-{self.pos+n-1}')['Body']
        try:
            value = body.read()
        finally:
            body.close()
        if len(value) != n:
            raise IOError('incomplete_source_range')
        self.pos += len(value); self.bytes_read += len(value)
        return value


def prepare(env_file, ca, endpoint, out, count):
    env = {}
    for line in env_file.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1); env[k] = shlex.split(v)[0]
    client = boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=env['AWS_ACCESS_KEY_ID'],
                         aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'], verify=str(ca),
                         config=Config(connect_timeout=10, read_timeout=60, retries={'max_attempts': 2}))
    prefix = 'staging/sova-rudevices/hf-d9197eacfb0afff29d90a2d4e7d0d98a5dfb54bc/'
    objects = client.list_objects_v2(Bucket='speech-corpora', Prefix=prefix)['Contents']
    key = sorted(o['Key'] for o in objects if o['Key'].endswith('.parquet'))[0]
    guards, guard_sources = set(), []
    for page in client.get_paginator('list_objects_v2').paginate(Bucket='speech-datasets', Prefix='protection/'):
        for o in page.get('Contents', []):
            if not o['Key'].endswith('source-audio-sha256.txt'): continue
            data = client.get_object(Bucket='speech-datasets', Key=o['Key'])['Body'].read()
            guard_sources.append({'key': o['Key'], 'sha256': hashlib.sha256(data).hexdigest()})
            for line in data.decode().splitlines():
                h = line.split()[0]
                if len(h) == 64: guards.add(h)
    if not guards: raise ValueError('Missing benchmark protection hashes')
    # Verify actual IAM rather than trusting policy text.
    try:
        client.list_objects_v2(Bucket='speech-benchmarks', MaxKeys=1)
    except client.exceptions.ClientError as e:
        if e.response['Error']['Code'] != 'AccessDenied': raise
    else: raise ValueError('Ingest can list benchmark bucket')
    remote = S3File(client, 'speech-corpora', key)
    parquet = pq.ParquetFile(remote)
    print(json.dumps({'schema': str(parquet.schema_arrow), 'row_groups': parquet.num_row_groups,
                      'protection_hashes': len(guards)}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    selected, held, seen = [], {}, set()
    for batch in parquet.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            audio = row.get('audio', {})
            raw = audio.get('bytes') if isinstance(audio, dict) else None
            reference = row.get('transcription', row.get('text', row.get('sentence', '')))
            if not raw or not isinstance(reference, str) or not reference.strip():
                held['missing_audio_or_label'] = held.get('missing_audio_or_label', 0)+1; continue
            h = hashlib.sha256(raw).hexdigest()
            if h in guards or h in seen:
                held['protected_or_duplicate'] = held.get('protected_or_duplicate', 0)+1; continue
            values, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
            if values.shape[1] != 1 or not 2 <= len(values)/rate <= 15:
                held['duration_or_multichannel'] = held.get('duration_or_multichannel', 0)+1; continue
            values = values[:, 0]
            if not np.isfinite(values).all() or np.max(np.abs(values)) < 0.003:
                held['invalid_or_silent'] = held.get('invalid_or_silent', 0)+1; continue
            pcm_hash = hashlib.sha256((np.clip(values, -1, 1)*32767).astype('<i2').tobytes()).hexdigest()
            if rate != 16000:
                held['unsupported_rate'] = held.get('unsupported_rate', 0)+1; continue
            buf = io.BytesIO(); sf.write(buf, values, 16000, format='WAV', subtype='PCM_16')
            data = buf.getvalue(); normalized_sha = hashlib.sha256(data).hexdigest()
            if normalized_sha in guards: continue
            pcm = (np.clip(values, -1, 1)*32767).astype('<i2').tobytes()
            vad = webrtcvad.Vad(2)
            voiced = sum(vad.is_speech(pcm[i:i+640], 16000) for i in range(0, len(pcm)-639, 640))
            if voiced < 3:
                held['vad_no_speech'] = held.get('vad_no_speech', 0)+1; continue
            ident = f'sova-{h[:24]}' 
            (out/f'{ident}.wav').write_bytes(data)
            selected.append({'sample_id': ident, 'source_key': key, 'source_etag': remote.etag,
                'source_audio_sha256': h, 'decoded_source_pcm_sha256': pcm_hash,
                'audio_sha256': normalized_sha, 'audio_seconds': len(values)/16000,
                'source_label': reference, 'source_audio_path': audio.get('path'),
                'source_rate': rate, 'publication_eligible': False,
                'qc': {'finite': True, 'mono': True, 'non_silent': True,
                       'source_label_present': True, 'speech_detected': True, 'voiced_frames': voiced,
                       'backend': 'webrtcvad', 'version': '2.0.14-mode2-20ms'}})
            seen.add(h)
            if len(selected) >= count: break
        if len(selected) >= count: break
    report = {'source_bucket': 'speech-corpora', 'source_key': key, 'source_etag': remote.etag,
              'range_bytes_read': remote.bytes_read, 'selected': selected, 'held': held,
              'guards': guard_sources, 'benchmark_iam_denied': True,
              'publication_blockers': ['decoded-PCM benchmark overlap', 'speaker/group isolation', 'human audit']}
    (out/'source-manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'selected': len(selected), 'held': held, 'range_bytes_read': remote.bytes_read}), flush=True)
    if len(selected) != count: raise ValueError('Insufficient candidates')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--env-file', type=Path, required=True)
    p.add_argument('--ca', type=Path, required=True); p.add_argument('--endpoint', required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--count', type=int, default=40)
    a = p.parse_args()
    if not 1 <= a.count <= 100: p.error('pilot count must be 1..100')
    prepare(a.env_file, a.ca, a.endpoint, a.out, a.count)
