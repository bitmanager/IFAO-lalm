"""Temporarily reserve one existing slot for a fixed, isolated native API batch.

Reservation expires after 15 minutes even if this process dies. Production queue
jobs are untouched. Results are private experimental artifacts, never a release.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

from .contract import source_text, triad_decision
from .native_api import discover, generate
from .queue import protected
from .store import bucket, connect, s3
from .worker import agy_env


def reserve(account):
    with connect() as c:
        row = c.execute('SELECT * FROM accounts WHERE id=%s FOR UPDATE', (account,)).fetchone()
        if (not row or not row['enabled'] or row['blocked_reason']
                or row['next_request_at'] > datetime.now(timezone.utc) + timedelta(seconds=3)):
            raise ValueError('account_not_ready_for_canary')
        deadline = c.execute("UPDATE accounts SET next_request_at=now()+interval '15 minutes' "
                             'WHERE id=%s RETURNING next_request_at', (account,)).fetchone()['next_request_at']
        return row['next_request_at'], deadline


def release(account, original, deadline):
    with connect() as c:
        # Preserve any longer quota block or operational change made during the test.
        row = c.execute('UPDATE accounts SET next_request_at=greatest(%s,now()) '
                        'WHERE id=%s AND next_request_at=%s RETURNING id',
                        (original, account, deadline)).fetchone()
        return bool(row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--prefix', required=True)
    args = p.parse_args()
    account = os.environ['ACCOUNT_ID']
    samples = json.loads(args.manifest.read_text())
    if not 1 <= len(samples) <= 6 or len({s['sample']['audio_sha256'] for s in samples}) != len(samples):
        p.error('Canary requires one to six distinct samples')
    for row in samples:
        sample = row['sample']
        with connect() as c:
            if sample['split'] != 'train' or protected(c, sample):
                p.error('Protected or non-training sample')
        if source_text(sample) is None:
            p.error('Canary needs an explicitly attached source annotation')
    original, deadline = reserve(account)
    client = s3()
    prefix = args.prefix.rstrip('/') + '/' + account
    started = time.monotonic()
    stop = threading.Event()
    records = []
    def terminate(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        # Existing CLI requests complete before starting the experiment.
        while True:
            with connect() as c:
                n = c.execute("SELECT count(*) AS n FROM jobs WHERE account_id=%s AND state='RUNNING'", (account,)).fetchone()['n']
                ready = c.execute('SELECT enabled,blocked_reason FROM accounts WHERE id=%s', (account,)).fetchone()
            if not ready['enabled'] or ready['blocked_reason']:
                raise ValueError('account_became_blocked')
            if not n:
                break
            if time.monotonic() - started > 360:
                raise TimeoutError('existing_cli_requests_did_not_drain')
            time.sleep(2)
        # Native CLI owns credential refresh. This catalog call has no generation.
        auth = subprocess.run(['agy', 'models'], capture_output=True, text=True,
                              env=agy_env(), timeout=40)
        if auth.returncode != 0 or 'gemini-3.8-flash-low' not in auth.stdout:
            raise ValueError('native_auth_catalog_failed')
        project = discover()
        with tempfile.TemporaryDirectory(prefix='native-canary-', dir='/work') as temp:
            paths = {}
            for row in samples:
                sample = row['sample']; sha = sample['audio_sha256']
                head = client.head_object(Bucket=sample['audio_bucket'], Key=sample['audio_key'])
                if not 44 < head['ContentLength'] < 2 * 1024**2:
                    raise ValueError('canary_audio_size')
                path = Path(temp) / (sha + '.wav')
                client.download_file(sample['audio_bucket'], sample['audio_key'], str(path))
                if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                    raise ValueError('canary_audio_hash_mismatch')
                paths[sha] = path
            inference_started = time.monotonic()
            def one(task):
                row, variant = task
                sample = row['sample']; sha = sample['audio_sha256']
                if stop.is_set():
                    return {'audio_sha256': sha, 'prompt_variant': variant, 'status': 'skipped_after_failure'}
                try:
                    result, raw = generate(paths[sha], project, variant,
                        source_text(sample) if variant == 'source_correction' else None)
                except Exception as error:
                    stop.set()
                    return {'audio_sha256': sha, 'prompt_variant': variant,
                            'status': 'not_dispatched', 'error_type': type(error).__name__,
                            'inference_http_attempts': 0}
                if result.get('http_status') in (401, 403, 429) or result.get('http_status', 0) == 0:
                    stop.set()
                key = f'{prefix}/raw/{sha}/{variant}.json'
                if raw:
                    client.put_object(Bucket=bucket(), Key=key, Body=raw, ContentType='application/json')
                result.update(account=account, raw_key=key if raw else None)
                print(json.dumps({'sample': sha[:12], 'variant': variant, 'status': result['status'],
                                  'seconds': result.get('inference_seconds')}), flush=True)
                return result
            tasks = [(row, variant) for row in samples for variant in ('verbatim', 'coverage', 'source_correction')]
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                records = list(pool.map(one, tasks))
            comparisons = []
            for row in samples:
                sha = row['sample']['audio_sha256']
                passes = {r['prompt_variant']: r for r in records
                          if r['audio_sha256'] == sha and r.get('status') == 'complete'}
                comparisons.append({'audio_sha256': sha, 'control': row.get('control'),
                    'native_decision': triad_decision(passes), 'native_passes': passes,
                    'cli_previous_passes': row.get('cli_previous_passes', {})})
            report = {'purpose': 'native_api_canary_review_only_not_training', 'account': account,
                'captured_at': datetime.now(timezone.utc).isoformat(), 'concurrency': 5,
                'inference_wall_seconds': round(time.monotonic()-inference_started, 3),
                'records': records, 'comparisons': comparisons,
                'inference_http_attempts': sum(r.get('inference_http_attempts', 0) for r in records)}
            client.put_object(Bucket=bucket(), Key=prefix+'/results.json',
                              Body=json.dumps(report, ensure_ascii=False).encode(), ContentType='application/json')
            print(json.dumps({'event': 'canary_report_saved', 'key': prefix+'/results.json',
                              'attempts': report['inference_http_attempts']}), flush=True)
    except Exception as error:
        client.put_object(Bucket=bucket(), Key=prefix+'/failure.json',
                          Body=json.dumps({'error_type': type(error).__name__, 'records': records}).encode())
        print(json.dumps({'event': 'canary_failed', 'error_type': type(error).__name__}), flush=True)
        raise SystemExit(1)
    finally:
        print(json.dumps({'event': 'cli_resumed', 'restored': release(account, original, deadline)}), flush=True)


if __name__ == '__main__':
    main()
