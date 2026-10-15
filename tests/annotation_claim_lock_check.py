"""Real PostgreSQL claim checks in a disposable schema; no S3 or inference.

Run in the annotation control/scheduler image. --queue-source tests a staged
module before deployment; --expect-wait reproduces the older blocking claim.
"""
import argparse
import concurrent.futures
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from unittest.mock import patch
from uuid import uuid4

from psycopg import sql
from psycopg.types.json import Jsonb
from lalm.annotation import queue
from lalm.annotation.store import connect as real_connect


def run(queue_source=None, expect_wait=False):
    target = queue
    if queue_source:
        spec = importlib.util.spec_from_file_location(
            'lalm.annotation.queue_candidate', queue_source)
        target = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(target)
    schema = 'annotation_claim_test_' + uuid4().hex
    model = 'gemini-3.8-flash-low'
    with real_connect() as conn:
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))

    @contextmanager
    def database():
        with real_connect() as conn:
            conn.execute(sql.SQL('SET LOCAL search_path TO {}').format(sql.Identifier(schema)))
            yield conn

    def reset():
        with database() as conn:
            conn.execute('TRUNCATE attempts,jobs,accounts CASCADE')
            conn.execute("INSERT INTO accounts(id,enabled,concurrency,min_interval_seconds) "
                         "VALUES ('a',true,3,0),('b',true,3,0)")
            for i in range(40):
                sample = {'conversation_id': 'fixture-'+str(i),
                    'audio_sha256': hashlib.sha256(str(i).encode()).hexdigest(),
                    'annotation_model': model, 'prompt_variant': 'verbatim'}
                conn.execute('INSERT INTO jobs(id,sample)VALUES(%s,%s)',
                             ('fixture-'+str(i), Jsonb(sample)))

    try:
        with database() as conn:
            conn.execute(Path(queue.__file__).with_name('schema.sql').read_text())
        with patch.object(target, 'connect', database), \
                patch.object(target, 'allowed_models', return_value=(model,)):
            reset()
            # A lock held by one account must not form a thread convoy.
            with database() as locker, concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                locker.execute("SELECT id FROM accounts WHERE id='a' FOR UPDATE")
                future = pool.submit(target.claim, 'a')
                waited = False
                result = None
                try:
                    try:
                        result = future.result(timeout=1)
                    except concurrent.futures.TimeoutError:
                        waited = True
                    other = target.claim('b')
                    assert other is not None, 'Independent account must still claim'
                finally:
                    locker.commit()
                    future.result(timeout=10)
            if expect_wait:
                assert waited, 'Expected old blocking behavior was not reproduced'
                return {'old_account_lock_wait_reproduced': True,
                        'independent_account_claimed': True, 'inference_requests': 0}
            assert not waited and result is None, 'Busy account must immediately skip its lock'

            reset()
            claims = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                for _ in range(8):
                    claims.extend(x for x in pool.map(lambda _: target.claim('a'), range(16)) if x)
                    if len(claims) >= 3:
                        break
                    time.sleep(.02)
            assert len(claims) == 3, 'Concurrency ceiling must be reached but never exceeded'
            assert len({j['id'] for j in claims}) == 3, 'A job must never be issued twice'
            assert all(j['attempt'] == 1 and j['lease_token'] for j in claims)
            with database() as conn:
                assert conn.execute("SELECT count(*) n FROM jobs WHERE state='RUNNING'").fetchone()['n'] == 3
                assert conn.execute('SELECT count(*) n FROM attempts').fetchone()['n'] == 3

            reset()
            with database() as conn:
                conn.execute("UPDATE accounts SET next_request_at=now()+interval '1 hour' WHERE id='a'")
            assert target.claim('a') is None, 'Future quota/cooldown must stay closed'
            with database() as conn:
                conn.execute("UPDATE accounts SET next_request_at=now(),enabled=false WHERE id='a'")
            assert target.claim('a') is None, 'Disabled accounts must stay closed'
            with database() as conn:
                conn.execute("UPDATE accounts SET enabled=true,blocked_reason='quota' WHERE id='a'")
            assert target.claim('a') is None, 'Blocked accounts must stay closed'
            with database() as conn:
                assert conn.execute('SELECT count(*) n FROM attempts').fetchone()['n'] == 0
                conn.execute("UPDATE accounts SET blocked_reason='' WHERE id='a'")
            assert target.claim('a') is not None, 'An empty reason retains its previous semantics'

            reset()
            with database() as conn:
                conn.execute("UPDATE jobs SET sample=jsonb_set(sample,'{annotation_model}',"
                             "to_jsonb('gemini-4-argon-low'::text)),created_at=now()-interval '1 day' "
                             "WHERE id='fixture-0'")
            claimed = target.claim('a')
            assert claimed and claimed['sample']['annotation_model'] == model
            return {'busy_account_skipped': True, 'independent_account_claimed': True,
                    'concurrency_cap': 3, 'duplicate_claims': 0,
                    'quota_disabled_and_blocked_guards': True,
                    'model_whitelist': True, 'inference_requests': 0}
    finally:
        with real_connect() as conn:
            conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-source', type=Path)
    parser.add_argument('--expect-wait', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.queue_source, args.expect_wait)))
