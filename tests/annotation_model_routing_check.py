"""Exercise real queue routing and deduplication in a rolled-back private schema."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from psycopg import sql

from lalm.annotation.contract import MODEL, PREPROCESSING
from lalm.annotation.queue import claim, enqueue
from lalm.annotation.store import connect

with connect() as conn:
    name = 'annotation_test_' + uuid4().hex
    conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(name)))
    conn.execute(sql.SQL('SET LOCAL search_path TO {}').format(sql.Identifier(name)))
    conn.execute(Path('lalm/annotation/schema.sql').read_text())
    for account in ('pro', 'ultra'):
        conn.execute('INSERT INTO accounts(id,enabled,concurrency,min_interval_seconds) '
                     'VALUES (%s,true,2,0)', (account,))
    sample = {'sample_id': 'one', 'conversation_id': 'call', 'audio_key': 'audio/one.wav',
              'audio_sha256': 'a' * 64, 'split': 'train', 'preprocessing_version': PREPROCESSING,
              'pilot_id': 'precision-ru-triad-v1-test', 'prompt_variant': 'verbatim',
              'speech_qc': {'audio_sha256': 'a' * 64, 'speech_detected': True,
                            'backend': 'webrtcvad', 'version': 'test'}}
    legacy = enqueue(conn, sample)
    argon = enqueue(conn, dict(sample, annotation_model='gemini-4-argon-low'))
    assert legacy != argon
    assert enqueue(conn, dict(sample, annotation_model='gemini-4-argon-low')) == argon
    assert conn.execute('SELECT count(*) n FROM jobs').fetchone()['n'] == 2

    @contextmanager
    def same_connection():
        yield conn

    with patch('lalm.annotation.queue.connect', same_connection):
        with patch.dict('os.environ', {'ANNOTATION_ALLOWED_MODELS': MODEL}):
            assert claim('pro')['id'] == legacy
            assert claim('pro') is None, 'Pro must never claim Argon'
        with patch.dict('os.environ', {'ANNOTATION_ALLOWED_MODELS': 'gemini-4-argon-low'}):
            assert claim('ultra')['id'] == argon
    conn.rollback()
print('Model routing and per-model deduplication passed; transaction rolled back')
