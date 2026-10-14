"""Check real reservation rollback without touching live accounts or calling AGY."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from psycopg import sql
from lalm.annotation.native_canary import reserve, release
from lalm.annotation.store import connect


with connect() as conn:
    name = 'native_test_' + uuid4().hex
    conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(name)))
    conn.execute(sql.SQL('SET LOCAL search_path TO {}').format(sql.Identifier(name)))
    conn.execute(Path('lalm/annotation/schema.sql').read_text())
    conn.execute("INSERT INTO accounts(id,enabled,concurrency,min_interval_seconds) VALUES ('test',true,5,0)")

    @contextmanager
    def same_connection():
        yield conn

    with patch('lalm.annotation.native_canary.connect', same_connection):
        original, deadline = reserve('test')
        assert conn.execute("SELECT next_request_at>now() ready FROM accounts WHERE id='test'").fetchone()['ready']
        assert release('test', original, deadline)
        assert conn.execute("SELECT next_request_at<=now() ready FROM accounts WHERE id='test'").fetchone()['ready']
        original, deadline = reserve('test')
        conn.execute("UPDATE accounts SET next_request_at=%s+interval '1 hour' WHERE id='test'", (deadline,))
        assert not release('test', original, deadline), 'Do not erase a later quota block'
        conn.execute("UPDATE accounts SET enabled=false,next_request_at=now() WHERE id='test'")
        try:
            reserve('test')
        except ValueError:
            pass
        else:
            raise AssertionError('Disabled accounts must not be reserved')
    conn.rollback()
print('Canary reservation and guarded rollback passed; private transaction rolled back')
