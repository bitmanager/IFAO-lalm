"""Run in the control image against disposable schema/bucket; no AGY requests."""
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import unittest
from unittest.mock import patch
import uuid
import wave

from psycopg import sql
from lalm.annotation import queue
from lalm.annotation.contract import MODEL, PREPROCESSING
from lalm.annotation.pipeline import ingest, publish
from lalm.annotation.store import bucket, connect, initialize, s3
from lalm.annotation.worker import classify, process


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = 'test_' + uuid.uuid4().hex
        with connect() as conn:
            conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(cls.schema)))
        os.environ['PGOPTIONS'] = '-c search_path=' + cls.schema
        os.environ['S3_BUCKET'] = cls.schema.replace('_', '-')
        initialize()
        Path('/work').mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        client = s3()
        for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket()):
            for obj in page.get('Contents', []):
                client.delete_object(Bucket=bucket(), Key=obj['Key'])
        client.delete_bucket(Bucket=bucket())
        with connect() as conn:
            conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(cls.schema)))

    def setUp(self):
        with connect() as conn:
            conn.execute('TRUNCATE jobs,attempts,accounts,holdouts,publications,source_manifests CASCADE')
            conn.execute("INSERT INTO accounts(id,enabled,concurrency,min_interval_seconds) VALUES ('test',true,2,0)")

    def sample(self, n=0):
        data = io.BytesIO()
        with wave.open(data, 'wb') as wav:
            wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            wav.writeframes(struct.pack('<h', 100 + n) * 1600)
        h = hashlib.sha256(data.getvalue()).hexdigest()
        key = 'audio/' + h + '.wav'
        s3().put_object(Bucket=bucket(), Key=key, Body=data.getvalue())
        return {'sample_id': str(n), 'conversation_id': 'call-' + str(n), 'audio_key': key,
                'audio_sha256': h, 'split': 'train', 'preprocessing_version': PREPROCESSING,
                'speech_qc': {'audio_sha256': h, 'speech_detected': True, 'backend': 'test', 'version': 'test'}}

    def test_atomic_limits_idempotency_and_quota(self):
        samples = [self.sample(n) for n in range(5)]
        with connect() as conn:
            for sample in samples * 2:
                queue.enqueue(conn, sample)
            self.assertEqual(conn.execute('SELECT count(*) AS n FROM jobs').fetchone()['n'], 5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            jobs = [j for j in pool.map(lambda _: queue.claim('test'), range(10)) if j]
        self.assertEqual(len(jobs), 2)
        self.assertNotEqual(jobs[0]['id'], jobs[1]['id'])
        queue.finish(jobs[0], 'NEEDS_REVIEW', block='quota_exhausted')
        self.assertIsNone(queue.claim('test'))

    def test_holdout_and_split_guard(self):
        sample = self.sample()
        with connect() as conn:
            queue.enqueue(conn, sample)
        with self.assertRaisesRegex(ValueError, 'split_contamination'), connect() as conn:
            queue.enqueue(conn, dict(sample, sample_id='other', split='validation'))
        with connect() as conn:
            conn.execute("INSERT INTO holdouts VALUES ('audio_sha256',%s)", (sample['audio_sha256'],))
        self.assertIsNone(queue.claim('test'))
        with self.assertRaisesRegex(ValueError, 'holdout_contamination'), connect() as conn:
            queue.enqueue(conn, dict(sample, sample_id='new'))

    def test_expired_attempt_is_not_retried(self):
        with connect() as conn:
            queue.enqueue(conn, self.sample())
        job = queue.claim('test')
        with connect() as conn:
            conn.execute("UPDATE jobs SET lease_until=now()-interval '1 second'")
        self.assertEqual(queue.recover_expired(), 1)
        self.assertIsNone(queue.claim('test'))
        with self.assertRaisesRegex(RuntimeError, 'stale_lease'):
            queue.finish(job, 'VALIDATED')

    def test_complete_pipeline_and_publication_retry(self):
        sample = self.sample()
        client = s3()
        client.put_object(Bucket=bucket(), Key='benchmarks/holdouts.json',
                          Body=json.dumps({'conversation_ids': [], 'audio_sha256': []}).encode())
        client.put_object(Bucket=bucket(), Key='incoming/test.jsonl', Body=json.dumps(sample).encode())
        ingest()
        ingest()
        job = queue.claim('test')

        def fake_agy(path, workspace, variant="verbatim"):
            result = {'status': 'SUCCESS', 'num_turns': 1, 'structured_output': {
                'audio_heard': True, 'language': 'ru', 'transcript': 'Тестовая расшифровка.',
                'uncertain_fragments': [], 'context_consistency_notes': []}}
            events = [{'event': 'init', 'init': {'model': MODEL}},
                      {'event': 'step_update', 'step_update': {'step_type': 'tool', 'tool_name': 'view_file',
                       'state': 'DONE', 'tool_info': {'parameters': {'AbsolutePath': str(path)}}}},
                      {'event': 'result', 'result': result}]
            return 0, '\n'.join(json.dumps(x) for x in events), '', False

        with patch('lalm.annotation.worker.run_agy', fake_agy):
            process(job)
        result = publish()
        self.assertEqual(result['published'], 1)
        self.assertEqual(publish()['published'], 0)
        # Simulate S3 success followed by database transaction rollback.
        with connect() as conn:
            conn.execute("UPDATE jobs SET state='VALIDATED'")
        self.assertEqual(publish()['manifest_key'], result['manifest_key'])
        self.assertGreater(client.head_object(Bucket=bucket(), Key=result['manifest_key'])['ContentLength'], 0)

    def test_independent_passes_third_pass_and_no_training_publication(self):
        from lalm.annotation.review import reconcile, compare
        from psycopg.types.json import Jsonb
        sample = dict(self.sample(), pilot_id='test-pilot', review_only=True, adaptive_third_pass=True)
        with connect() as conn:
            for variant, text in [('verbatim', 'я не согласен'), ('coverage', 'я согласен')]:
                ident = queue.enqueue(conn, dict(sample, prompt_variant=variant))
                conn.execute("UPDATE jobs SET state='VALIDATED',result=%s WHERE id=%s",
                             (Jsonb({'verbatim_text': text}), ident))
        self.assertEqual(publish()['published'], 0)
        first = reconcile()
        self.assertEqual(first['third_passes_requested'], 1)
        self.assertEqual(reconcile()['third_passes_requested'], 0)
        with connect() as conn:
            self.assertEqual(conn.execute('SELECT count(*) AS n FROM jobs').fetchone()['n'], 3)
        third = queue.claim('test')
        self.assertEqual(third['sample']['prompt_variant'], 'literal_audit')
        self.assertFalse(compare('я не согласен', 'я согласен')['normalized_agreement'])
        self.assertTrue(compare('Ёлка!', 'елка')['normalized_agreement'])

    def test_timeouts_and_authorization_are_not_blind_retries(self):
        self.assertEqual(classify('', True)[0], 'NEEDS_REVIEW')
        self.assertEqual(classify('Individual quota reached')[2], 'quota_exhausted')
        self.assertEqual(classify('403 access denied')[2], 'authentication_or_access_denied')
        self.assertEqual(classify('429 Too many requests')[0], 'RETRYABLE_FAILED')


unittest.main()
