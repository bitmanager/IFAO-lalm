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
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import webrtcvad
from psycopg.types.json import Jsonb

from .contract import PREPROCESSING
from .pilot_source import S3File
from .store import s3, connect, dataset_bucket
from .source_labels import original_label
from .source_identity import production_conversation
from .source_scheduling import select_source
from .gigaam_reference import reference as gigaam_reference, POLICY as GIGAAM_POLICY
from .gigaam_reference import requires_reference, load_reference_index

STOP = False


def parquet_source_kind(schema):
    """Distinguish embedded audio from source-label tables without guessing paths."""
    if 'audio' in schema.names:
        audio = schema.field('audio').type
        if pa.types.is_struct(audio) and 'bytes' in [f.name for f in audio]:
            if pa.types.is_binary(audio.field('bytes').type) or pa.types.is_large_binary(audio.field('bytes').type):
                return 'embedded_audio'
        raise ValueError('unsupported_parquet_audio_schema')
    if 'filepath' in schema.names and pa.types.is_string(schema.field('filepath').type):
        return 'reference_metadata'
    raise ValueError('unsupported_parquet_source_schema')


def tar_audio_rows(body):
    """Read adjacent WebDataset/OpenSTT sidecars, preserving their source names.

    Only one audio is buffered. Sidecars before their audio use a bounded cache;
    original archive bytes remain authoritative when a layout is non-adjacent.
    """
    pending = None
    labels = {}
    with tarfile.open(fileobj=body, mode='r|*', bufsize=1024*1024) as archive:
        for member in archive:
            if not member.isfile(): continue
            stem, _, suffix = member.name.rpartition('.')
            suffix = suffix.lower()
            if suffix in ('wav', 'flac', 'mp3', 'ogg', 'opus', 'm4a'):
                if pending:
                    name, raw, previous_stem = pending
                    sidecars = labels.pop(previous_stem, [])
                    yield name, raw, {'archive_sidecars': sidecars} if sidecars else None
                raw = archive.extractfile(member).read() if member.size <= 100*1024*1024 else None
                pending = (member.name, raw, stem)
            elif suffix in ('txt', 'json') and member.size <= 256*1024:
                try:
                    text = archive.extractfile(member).read().decode('utf-8')
                    value = json.loads(text) if suffix == 'json' else text
                except (UnicodeError, ValueError):
                    continue
                labels.setdefault(stem, []).append({'member': member.name, 'value': value})
                if len(labels) > 64:
                    candidates = [k for k in labels if not pending or k != pending[2]]
                    if candidates: labels.pop(candidates[0])
        if pending:
            name, raw, stem = pending
            sidecars = labels.pop(stem, [])
            yield name, raw, {'archive_sidecars': sidecars} if sidecars else None


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
        with S3File(client, asset['bucket'], asset['object_key']) as remote:
            if remote.etag != asset['etag']: raise ValueError('source_changed')
            yield from tar_audio_rows(remote)
    else:
        if asset['bytes'] > 100*1024*1024:
            yield asset['object_key'], None, None
        else:
            raw = client.get_object(Bucket=asset['bucket'], Key=asset['object_key'], IfMatch=asset['etag'])['Body'].read()
            yield asset['object_key'], raw, original_label(asset, raw)


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


def pending_work(conn):
    # Count both imported jobs and staged batches; an import must not count twice.
    return conn.execute("SELECT (SELECT count(*) FROM jobs WHERE state IN "
        "('PENDING','RUNNING','RETRYABLE_FAILED')) + "
        "COALESCE((SELECT sum(p.row_count) FROM prepared_manifests p "
        "WHERE NOT EXISTS (SELECT 1 FROM source_manifests s WHERE s.object_key=p.object_key)),0) AS n").fetchone()['n']


def prepare_asset(asset):
    client = s3()
    if asset['codec'] == 'parquet':
        remote = S3File(client, asset['bucket'], asset['object_key'])
        if remote.etag != asset['etag']: raise ValueError('source_changed')
        parquet = pq.ParquetFile(remote)
        if parquet_source_kind(parquet.schema_arrow) == 'reference_metadata':
            # Preserve the original Parquet in place. Local filepath values are
            # references, never instructions to read arbitrary worker files.
            metadata = {'source_kind': 'reference_metadata',
                        'metadata_rows': parquet.metadata.num_rows,
                        'columns': parquet.schema_arrow.names,
                        'audio_link_status': 'not_resolved_by_this_table'}
            with connect() as conn:
                conn.execute("UPDATE source_assets SET state='REFERENCE_METADATA',"
                             "cursor_json=cursor_json || %s,updated_at=now() WHERE id=%s",
                             (Jsonb(metadata), asset['id']))
            return
    registry = json.loads(client.get_object(Bucket=dataset_bucket(), Key='benchmarks/holdouts.json')['Body'].read())
    protected = set(registry['audio_sha256'])
    conversation = production_conversation(asset['object_key'])
    if conversation is not None and conversation in set(registry['conversation_ids']):
        with connect() as conn:
            conn.execute("UPDATE source_assets SET state='EXCLUDED_HOLDOUT',"
                         "cursor_json=cursor_json || %s,updated_at=now() WHERE id=%s",
                         (Jsonb({'exclusion_reason': 'protected_conversation',
                                 'conversation_id': conversation}), asset['id']))
        return
    load_reference_index(asset)
    cursor = asset['cursor_json'].get('rows', 0)
    made = asset['cursor_json'].get('samples', 0)
    errors = asset['cursor_json'].get('errors', 0)
    reference_only = requires_reference(asset)
    selection = {}
    if reference_only:
        selection = {'selection_policy': 'gigaam-reference-only-v1',
            'selection_policy_start_row': asset['cursor_json'].get('selection_policy_start_row', cursor),
            'unreferenced_rows': asset['cursor_json'].get('unreferenced_rows', 0),
            'unaligned_windows': asset['cursor_json'].get('unaligned_windows', 0)}
    buffered = []
    last_index = cursor - 1

    def flush():
        key = None
        if buffered:
            payload = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in buffered).encode()
            key = 'incoming/'+hashlib.sha256(payload).hexdigest()+'.jsonl'
            client.put_object(Bucket=dataset_bucket(), Key=key, Body=payload)
        # S3 output precedes one DB transaction recording its backlog and cursor.
        # A crash can replay the same immutable manifest; job IDs deduplicate it.
        with connect() as conn:
            if key:
                conn.execute('INSERT INTO prepared_manifests(object_key,row_count) VALUES (%s,%s) '
                             'ON CONFLICT DO NOTHING', (key, len(buffered)))
            conn.execute('UPDATE source_assets SET cursor_json=%s,updated_at=now() WHERE id=%s',
                (Jsonb({'rows': last_index+1, 'samples': made, 'errors': errors, **selection}), asset['id']))
        buffered.clear()
        return True

    for index, (member, raw, label) in enumerate(audio_rows(client, asset)):
        if index < cursor: continue
        if STOP: return
        while True:
            with connect() as conn:
                pending = pending_work(conn)
            if pending < int(os.environ.get('PREPARE_MAX_PENDING', '1500')): break
            if STOP: return
            time.sleep(10)
        rows = []
        try:
            if not raw: raise ValueError('missing_or_oversized_audio')
            raw_sha = hashlib.sha256(raw).hexdigest()
            reference = gigaam_reference(asset, member, raw_sha)
            if reference_only and reference is None:
                selection['unreferenced_rows'] += 1
            if raw_sha not in protected and (not reference_only or reference):
                info = sf.info(io.BytesIO(raw)) if reference else None
                for channel, start, end, data in segments(raw):
                    reference_duration = reference.get('source_duration') if reference else None
                    duration_matches = (not reference_only or
                        isinstance(reference_duration, (int, float))
                        and abs(info.duration - reference_duration) <= 1 / 16000)
                    aligned = (reference and info.channels == 1 and start == 0
                               and abs(end - info.duration) <= 1 / 16000 and duration_matches)
                    if reference_only and not aligned:
                        selection['unaligned_windows'] += 1
                        continue
                    sha = hashlib.sha256(data).hexdigest()
                    if sha in protected: continue
                    key = f'prepared/annotation-v1/{sha}.wav'
                    client.put_object(Bucket='speech-corpora', Key=key, Body=data)
                    sample = {'sample_id': sha, 'conversation_id': conversation or 'unresolved-'+sha,
                        'audio_bucket': 'speech-corpora', 'audio_key': key, 'audio_sha256': sha,
                        'source_audio_sha256': raw_sha, 'split': 'train',
                        'preprocessing_version': PREPROCESSING, 'review_only': True,
                        'adaptive_third_pass': True, 'pilot_id': 'corpus-'+asset['id'],
                        'speech_qc': {'audio_sha256': sha, 'speech_detected': True,
                            'backend': 'webrtcvad', 'version': '2.0.14-mode2-20ms'},
                        'source_provenance': {'bucket': asset['bucket'], 'key': asset['object_key'],
                            'etag': asset['etag'], 'member': member, 'channel': channel,
                            'start_seconds': start, 'end_seconds': end, 'max_overlap_seconds': 0.5,
                            'original_label': (label['text'] if isinstance(label, dict)
                                and label.get('annotation_source') == 'original_source_index'
                                and isinstance(label.get('text'), str) and not label.get('text_conflict') else label),
                            'original_label_metadata': (label if isinstance(label, dict)
                                and label.get('annotation_source') == 'original_source_index' else None),
                            'group_identity_verified': conversation is not None,
                            'group_identity_source': 'production_session_path' if conversation else 'unresolved'}}
                    # A whole-file ASR label must never supervise a short window
                    # or one channel of a multichannel recording.
                    if aligned:
                        sample['annotation_policy'] = GIGAAM_POLICY
                        sample['source_provenance']['gigaam_reference'] = dict(
                            reference, prepared_audio_sha256=sha,
                            full_source_coverage=True)
                    for variant in ('verbatim',):
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
                asset = select_source(conn)
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
