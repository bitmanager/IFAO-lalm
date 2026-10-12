"""External CPU source worker, controlled by Airflow's durable source registry.

All output is candidate/review-only. No source text is passed to AGY. Exact source
and normalized hashes are checked; group/PCM benchmark audits gate training release.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import shutil
import imageio_ffmpeg
import subprocess
import tarfile
import time
import wave

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import webrtcvad
from psycopg.types.json import Jsonb

from .contract import PREPROCESSING
from .pilot_source import S3File
from .store import s3, connect, dataset_bucket

STOP = False


def audio_rows(client, asset):
    """Yield native identity and original audio bytes without rewriting sources."""
    if asset['codec'] == 'parquet':
        remote = S3File(client, asset['bucket'], asset['object_key'])
        if remote.etag != asset['etag']: raise ValueError('source_changed')
        index = 0
        for batch in pq.ParquetFile(remote).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                a = row.get('audio')
                raw = a.get('bytes') if isinstance(a, dict) else None
                text = row.get('transcription', row.get('text', row.get('sentence')))
                yield str(index), raw, text
                index += 1
    elif asset['codec'] == 'tar':
        body = client.get_object(Bucket=asset['bucket'], Key=asset['object_key'], IfMatch=asset['etag'])['Body']
        with tarfile.open(fileobj=body, mode='r|*') as archive:
            for member in archive:
                if member.isfile() and member.name.lower().endswith(('.wav', '.flac', '.mp3', '.ogg', '.opus', '.m4a')):
                    if member.size > 100*1024*1024:
                        yield member.name, None, None
                    else:
                        yield member.name, archive.extractfile(member).read(), None
    else:
        if asset['bytes'] > 100*1024*1024:
            yield asset['object_key'], None, None
        else:
            yield asset['object_key'], client.get_object(Bucket=asset['bucket'], Key=asset['object_key'], IfMatch=asset['etag'])['Body'].read(), None


def segments(raw):
    audio, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
    if not np.isfinite(audio).all() or len(audio) == 0: raise ValueError('invalid_audio')
    for channel in range(audio.shape[1]):
        samples = audio[:, channel]
        if rate != 16000:
            r = subprocess.run([shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error', '-f', 'f32le', '-ar', str(rate),
                '-ac', '1', '-i', 'pipe:0', '-f', 's16le', '-ar', '16000', '-ac', '1', 'pipe:1'],
                input=samples.astype('<f4').tobytes(), capture_output=True, timeout=120, check=True)
            pcm = np.frombuffer(r.stdout, dtype='<i2')
        else:
            pcm = (np.clip(samples, -1, 1)*32767).astype('<i2')
        vad = webrtcvad.Vad(2)
        voiced = np.array([vad.is_speech(pcm[i:i+320].tobytes(), 16000)
                           for i in range(0, len(pcm)-319, 320)], dtype=bool)
        # Up to 20 seconds per window, cut near a pause when possible; 0.5s overlap.
        start = 0
        while start < len(pcm):
            end = min(start+320000, len(pcm))
            if end < len(pcm):
                quiet = np.flatnonzero(~voiced[max(start//320+50, end//320-150):end//320])
                if len(quiet): end = (max(start//320+50, end//320-150)+int(quiet[-1])+1)*320
            if np.count_nonzero(voiced[start//320:end//320]) >= 3:
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as w:
                    w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(pcm[start:end].tobytes())
                yield channel, start/16000, end/16000, buf.getvalue()
            if end >= len(pcm): break
            start = max(start+320, end-8000)


def prepare_asset(asset):
    client = s3()
    registry = json.loads(client.get_object(Bucket=dataset_bucket(), Key='benchmarks/holdouts.json')['Body'].read())
    protected = set(registry['audio_sha256'])
    cursor = asset['cursor_json'].get('rows', 0)
    made = asset['cursor_json'].get('samples', 0)
    errors = asset['cursor_json'].get('errors', 0)
    buffered = []
    last_index = cursor - 1

    def flush():
        if buffered:
            payload = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in buffered).encode()
            key = 'incoming/'+hashlib.sha256(payload).hexdigest()+'.jsonl'
            client.put_object(Bucket=dataset_bucket(), Key=key, Body=payload)
            # Wait for Airflow to accept this batch before preparing another.
            # This bounds unimported work as well as the durable job queue.
            while not STOP:
                with connect() as conn:
                    accepted = conn.execute('SELECT 1 FROM source_manifests WHERE object_key=%s', (key,)).fetchone()
                if accepted: break
                time.sleep(5)
            if STOP: return False
        with connect() as conn:
            conn.execute('UPDATE source_assets SET cursor_json=%s,updated_at=now() WHERE id=%s',
                (Jsonb({'rows': last_index+1, 'samples': made, 'errors': errors}), asset['id']))
        buffered.clear()
        return True

    for index, (member, raw, label) in enumerate(audio_rows(client, asset)):
        if index < cursor: continue
        if STOP: return
        while True:
            with connect() as conn:
                pending = conn.execute("SELECT count(*) AS n FROM jobs WHERE state IN ('PENDING','RUNNING','RETRYABLE_FAILED')").fetchone()['n']
            if pending < int(os.environ.get('PREPARE_MAX_PENDING', '1500')): break
            if STOP: return
            time.sleep(10)
        rows = []
        try:
            if not raw: raise ValueError('missing_or_oversized_audio')
            raw_sha = hashlib.sha256(raw).hexdigest()
            if raw_sha not in protected:
                for channel, start, end, data in segments(raw):
                    sha = hashlib.sha256(data).hexdigest()
                    if sha in protected: continue
                    key = f'prepared/annotation-v1/{sha}.wav'
                    client.put_object(Bucket='speech-corpora', Key=key, Body=data)
                    sample = {'sample_id': sha, 'conversation_id': 'unresolved-'+sha,
                        'audio_bucket': 'speech-corpora', 'audio_key': key, 'audio_sha256': sha,
                        'source_audio_sha256': raw_sha, 'split': 'train',
                        'preprocessing_version': PREPROCESSING, 'review_only': True,
                        'adaptive_third_pass': True, 'pilot_id': 'corpus-'+asset['id'],
                        'speech_qc': {'audio_sha256': sha, 'speech_detected': True,
                            'backend': 'webrtcvad', 'version': '2.0.14-mode2-20ms'},
                        'source_provenance': {'bucket': asset['bucket'], 'key': asset['object_key'],
                            'etag': asset['etag'], 'member': member, 'channel': channel,
                            'start_seconds': start, 'end_seconds': end, 'max_overlap_seconds': 0.5,
                            'original_label': label, 'group_identity_verified': False}}
                    for variant in ('verbatim', 'coverage'):
                        rows.append(dict(sample, prompt_variant=variant))
                    made += 1
        except (ValueError, RuntimeError, sf.LibsndfileError, subprocess.SubprocessError):
            errors += 1
        buffered.extend(rows)
        last_index = index
        if len(buffered) >= 256 or (index+1) % 128 == 0:
            if not flush(): return
    if not flush(): return
    with connect() as conn:
        conn.execute('UPDATE source_assets SET state=%s,updated_at=now() WHERE id=%s',
                     ('PREPARED' if made else 'NEEDS_REVIEW', asset['id']))


def main():
    global STOP
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop);signal.signal(signal.SIGINT, stop)
    # Only one instance is permitted by the advisory session lock.
    with connect() as lock:
        lock.autocommit = True
        if not lock.execute('SELECT pg_try_advisory_lock(173922) AS ok').fetchone()['ok']:
            raise RuntimeError('source_worker_already_running')
        while not STOP:
            with connect() as conn:
                asset = conn.execute("SELECT * FROM source_assets WHERE state IN ('QUEUED','PREPARING') "
                     "ORDER BY CASE WHEN object_key LIKE '%sova-rudevices%' THEN 0 ELSE 1 END,created_at "
                     "FOR UPDATE SKIP LOCKED LIMIT 1").fetchone()
                if asset: conn.execute("UPDATE source_assets SET state='PREPARING',updated_at=now() WHERE id=%s", (asset['id'],))
            if not asset:
                time.sleep(10);continue
            try:
                prepare_asset(asset)
                print(json.dumps({'source_id': asset['id'], 'event': 'source_pass_finished'}), flush=True)
            except Exception as error:
                with connect() as conn:
                    conn.execute("UPDATE source_assets SET state='NEEDS_REVIEW',cursor_json=cursor_json || %s,updated_at=now() WHERE id=%s",
                        (Jsonb({'error_type': type(error).__name__}), asset['id']))
                print(json.dumps({'source_id': asset['id'], 'error_type': type(error).__name__}), flush=True)


if __name__ == '__main__':main()
