"""Small, review-only native API batches scheduled by Airflow and run by AGY-03."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

from .contract import source_text
from .queue import protected
from .store import bucket, connect, s3

ACCOUNT = 'agy-03'
ROOT = 'evaluation/native-api-canary-v1/recurring'
LEGACY = ('20261014-expanded-01', '20261014-shadow-01')


def schedule():
    """One 30-audio batch per six-hour slot; no overlap or automatic replay."""
    now = datetime.now(timezone.utc)
    batch_id = now.strftime('%Y%m%d-') + f'{now.hour // 6 * 6:02d}'
    prefix = f'{ROOT}/{batch_id}'
    client = s3()
    with connect() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtext('native_shadow_scheduler'))")
        existing = conn.execute('SELECT id,state FROM native_shadow_batches WHERE id=%s', (batch_id,)).fetchone()
        if existing:
            return {'batch': batch_id, 'state': existing['state']}
        active = conn.execute("SELECT id FROM native_shadow_batches WHERE state IN ('PENDING','RUNNING') LIMIT 1").fetchone()
        if active:
            return {'skipped': 'previous_batch_active', 'batch': active['id']}
        latest = conn.execute('SELECT id,state FROM native_shadow_batches ORDER BY created_at DESC LIMIT 1').fetchone()
        if latest and latest['state'] == 'NEEDS_REVIEW':
            return {'skipped': 'previous_batch_needs_review', 'batch': latest['id']}
        account = conn.execute('SELECT enabled,blocked_reason,next_request_at FROM accounts WHERE id=%s',
                               (ACCOUNT,)).fetchone()
        if not account or not account['enabled'] or account['blocked_reason']:
            return {'skipped': 'account_unavailable'}
        if (account['next_request_at'] - now).total_seconds() > 120:
            return {'skipped': 'account_cooldown'}
        used = {r['audio_sha256'] for r in conn.execute('SELECT audio_sha256 FROM native_shadow_samples')}
        # Previous canaries predate this registry. Exclude their audios too.
        for name in LEGACY:
            key = f'evaluation/native-api-canary-v1/{name}/manifest.json'
            try:
                manifest = json.loads(client.get_object(Bucket=bucket(), Key=key)['Body'].read())
            except client.exceptions.NoSuchKey:
                continue
            used.update(r['sample']['audio_sha256'] for r in manifest)
        rows = conn.execute("SELECT sample,result FROM jobs WHERE result IS NOT NULL "
            "AND sample->>'prompt_variant'='verbatim' AND sample->>'split'='train' "
            "AND updated_at >= now()-interval '48 hours' ORDER BY updated_at DESC LIMIT 20000").fetchall()
        chosen, conversations = [], set()
        for row in rows:
            sample, result = row['sample'], row['result']
            sha, conversation = sample.get('audio_sha256'), sample.get('conversation_id')
            if (result.get('language') != 'ru' or sha in used or conversation in conversations
                    or source_text(sample) is None or protected(conn, sample)
                    or not isinstance(result.get('audio_seconds'), (int, float))
                    or not 1 <= result['audio_seconds'] <= 20):
                continue
            chosen.append({'sample': sample,
                'control': {'audit': 'fresh_unverified', 'language_class': 'unknown'},
                'cli_previous_passes': {'verbatim': {
                    'verbatim_text': result.get('verbatim_text'), 'language': result.get('language')}}})
            used.add(sha); conversations.add(conversation)
            if len(chosen) == 30:
                break
        if len(chosen) < 30:
            return {'skipped': 'insufficient_fresh_audio', 'available': len(chosen)}
        data = json.dumps(chosen, ensure_ascii=False).encode()
        key = f'{prefix}/manifest.json'
        client.put_object(Bucket=bucket(), Key=key, Body=data, ContentType='application/json')
        conn.execute('INSERT INTO native_shadow_batches(id,account_id,manifest_key,state) VALUES (%s,%s,%s,%s)',
                     (batch_id, ACCOUNT, key, 'PENDING'))
        with conn.cursor() as cursor:
            cursor.executemany('INSERT INTO native_shadow_samples(audio_sha256,batch_id) VALUES (%s,%s)',
                               [(r['sample']['audio_sha256'], batch_id) for r in chosen])
        return {'batch': batch_id, 'state': 'PENDING', 'samples': len(chosen), 'manifest_key': key}


def run_one():
    """Claim one batch. Any interrupted or ambiguous run needs a human decision."""
    with connect() as conn:
        row = conn.execute("SELECT id,manifest_key FROM native_shadow_batches "
            "WHERE account_id=%s AND state='PENDING' ORDER BY created_at "
            'LIMIT 1 FOR UPDATE SKIP LOCKED', (ACCOUNT,)).fetchone()
        if not row:
            return False
        conn.execute("UPDATE native_shadow_batches SET state='RUNNING',started_at=now() WHERE id=%s", (row['id'],))
    prefix = f'{ROOT}/{row["id"]}'
    try:
        with tempfile.TemporaryDirectory(prefix='native-shadow-', dir='/work') as temp:
            manifest = Path(temp) / 'manifest.json'
            s3().download_file(bucket(), row['manifest_key'], str(manifest))
            process = subprocess.run([sys.executable, '-m', 'lalm.annotation.native_canary',
                '--manifest', str(manifest), '--prefix', prefix, '--max-samples', '30', '--adaptive'],
                timeout=850, check=False)
        key = f'{prefix}/{ACCOUNT}/results.json'
        if process.returncode:
            raise RuntimeError('canary_process_failed')
        report = json.loads(s3().get_object(Bucket=bucket(), Key=key)['Body'].read())
        if len(report.get('comparisons', [])) != 30 or any(
                r.get('status') not in ('complete',) for r in report.get('records', [])):
            raise ValueError('incomplete_native_report')
        with connect() as conn:
            conn.execute("UPDATE native_shadow_batches SET state='COMPLETE',result_key=%s,finished_at=now() "
                         "WHERE id=%s AND state='RUNNING'", (key, row['id']))
        print(json.dumps({'event': 'native_shadow_complete', 'batch': row['id']}), flush=True)
    except Exception as error:
        with connect() as conn:
            conn.execute("UPDATE native_shadow_batches SET state='NEEDS_REVIEW',error_code=%s,finished_at=now() "
                         "WHERE id=%s AND state='RUNNING'", (type(error).__name__, row['id']))
        print(json.dumps({'event': 'native_shadow_needs_review', 'batch': row['id'],
                          'error_type': type(error).__name__}), flush=True)
    return True


def loop(stop: threading.Event):
    if os.environ.get('ACCOUNT_ID') != ACCOUNT or os.environ.get('NATIVE_SHADOW_ENABLED', 'true') != 'true':
        return
    # A crashed container must not replay a batch whose requests may have reached Google.
    with connect() as conn:
        conn.execute("UPDATE native_shadow_batches SET state='NEEDS_REVIEW',error_code='interrupted_worker', "
                     "finished_at=now() WHERE account_id=%s AND state='RUNNING'", (ACCOUNT,))
    while not stop.is_set():
        try:
            if not run_one():
                stop.wait(20)
        except Exception as error:
            print(json.dumps({'event': 'native_shadow_poll_error', 'type': type(error).__name__}), flush=True)
            stop.wait(30)
