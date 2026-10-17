"""Transactional check against an existing PostgreSQL; leaves no schema or rows."""
from pathlib import Path
from uuid import uuid4
from psycopg import sql
from lalm.annotation.store import connect
from lalm.annotation.prepare import pending_work

with connect() as conn:
    name = 'annotation_test_' + uuid4().hex
    conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(name)))
    conn.execute(sql.SQL('SET LOCAL search_path TO {}').format(sql.Identifier(name)))
    conn.execute(Path('/opt/speech/lalm/annotation/schema.sql').read_text())
    conn.execute("INSERT INTO jobs(id,sample) VALUES ('test','{}')")
    conn.execute("INSERT INTO prepared_manifests(object_key,row_count) VALUES ('incoming/test.jsonl',2)")
    assert pending_work(conn) == 3
    conn.execute("INSERT INTO source_manifests(object_key,sha256,row_count) VALUES ('incoming/test.jsonl','test',2)")
    assert pending_work(conn) == 1, 'Imported manifest must not be counted twice'
    conn.execute("UPDATE jobs SET state='NEEDS_REVIEW'")
    assert pending_work(conn) == 0, 'Review results must not block new processing'
    conn.rollback()
print('PostgreSQL backlog checks passed; transaction rolled back')
