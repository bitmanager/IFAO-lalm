"""Check account cooldown without replaying a job in a rolled-back schema."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from psycopg import sql

from lalm.annotation.queue import claim, finish
from lalm.annotation.store import connect

with connect() as conn:
    name = 'annotation_test_' + uuid4().hex
    conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(name)))
    conn.execute(sql.SQL('SET LOCAL search_path TO {}').format(sql.Identifier(name)))
    conn.execute(Path('lalm/annotation/schema.sql').read_text())
    conn.execute("INSERT INTO accounts(id,enabled,concurrency,min_interval_seconds) "
                 "VALUES ('test',true,1,0)")
    conn.execute("INSERT INTO jobs(id,sample) VALUES "
                 "('one','{\"conversation_id\":\"test\",\"audio_sha256\":\"a\"}'), "
                 "('two','{\"conversation_id\":\"test\",\"audio_sha256\":\"b\"}')")

    @contextmanager
    def same_connection():
        yield conn

    with patch('lalm.annotation.queue.connect', same_connection):
        job = claim('test')
        finish(job, 'NEEDS_REVIEW', error_code='eligibility_service_unavailable',
               block='transport_cooldown', retry_seconds=60)
        account = conn.execute("SELECT *, next_request_at>now()+interval '59 seconds' "
                               "AS cooling FROM accounts WHERE id='test'").fetchone()
        assert account['enabled'] and account['blocked_reason'] is None and account['cooling']
        assert claim('test') is None, 'Cooldown must stop new dispatches'
        conn.execute("UPDATE accounts SET next_request_at=now() WHERE id='test'")
        following = claim('test')
        assert following['id'] != job['id'], 'Ambiguous job must never be replayed'
        assert conn.execute('SELECT state FROM jobs WHERE id=%s', (job['id'],)).fetchone()['state'] == 'NEEDS_REVIEW'
    conn.rollback()
print('Transport cooldown and no-replay checks passed; transaction rolled back')
