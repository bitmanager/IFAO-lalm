"""Exercise batched follow-ups and failed-report recovery; no inference or S3 writes."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from psycopg import sql
from psycopg.types.json import Jsonb

from lalm.annotation.contract import PREPROCESSING, TRIAD_VERSION, job_id
from lalm.annotation.queue import enqueue
from lalm.annotation.review import _enqueue_followups, reconcile
from lalm.annotation.store import connect


def sample(index):
    digest = hashlib.sha256(str(index).encode()).hexdigest()
    return {'sample_id': str(index), 'conversation_id': 'test-' + str(index),
            'audio_key': 'audio/' + digest + '.wav', 'audio_sha256': digest,
            'split': 'train', 'preprocessing_version': PREPROCESSING,
            'pilot_id': TRIAD_VERSION + '-batch-test', 'review_only': True,
            'source_provenance': {'original_label': 'тест'},
            'speech_qc': {'audio_sha256': digest, 'speech_detected': True,
                          'backend': 'test', 'version': 'test'}}


class Reports:
    fail = True
    writes = 0

    def __init__(self):
        self.payloads = []

    def put_object(self, **kwargs):
        if self.fail:
            raise RuntimeError('simulated_report_failure')
        self.writes += 1
        self.payloads.append(json.loads(kwargs['Body']))


with connect() as conn:
    name = 'annotation_test_' + uuid4().hex
    conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(name)))
    conn.execute(sql.SQL('SET LOCAL search_path TO {}').format(sql.Identifier(name)))
    conn.execute(Path('lalm/annotation/schema.sql').read_text())
    for index in range(131):
        row = sample(index)
        if index == 130:
            row['source_provenance'] = {}
        ident = enqueue(conn, row)
        result = {'verbatim_text': 'тест', 'language': 'en' if index == 129 else 'ru',
                  'uncertain_spans': []}
        conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',result=%s WHERE id=%s",
                     (Jsonb(result), ident))

    @contextmanager
    def transaction():
        # Savepoints model batch rollback while the entire test stays disposable.
        with conn.transaction():
            yield conn

    reports = Reports()
    with patch('lalm.annotation.review.connect', transaction), \
            patch('lalm.annotation.review.s3', return_value=reports):
        try:
            reconcile()
        except RuntimeError as error:
            assert str(error) == 'simulated_report_failure'
        else:
            raise AssertionError('Expected report failure')
        assert conn.execute('SELECT count(*) n FROM annotation_review_cursors').fetchone()['n'] == 0
        assert conn.execute('SELECT count(*) n FROM jobs').fetchone()['n'] == 389
        for index in (129, 130):
            assert conn.execute("SELECT count(*) n FROM jobs WHERE sample->>'sample_id'=%s",
                                (str(index),)).fetchone()['n'] == 1
        reports.fail = False
        assert reconcile()['third_passes_requested'] == 0, 'Committed children must not be duplicated'
        assert conn.execute('SELECT count(*) n FROM jobs').fetchone()['n'] == 389
        assert conn.execute('SELECT count(*) n FROM annotation_review_cursors').fetchone()['n'] == 1

        allowed = dict(sample('guard-a'), prompt_variant='coverage')
        held = dict(sample('guard-b'), prompt_variant='coverage', source_audio_sha256='f' * 64)
        conn.execute("INSERT INTO holdouts VALUES ('audio_sha256',%s)", ('f' * 64,))
        try:
            _enqueue_followups([allowed, held])
        except ValueError as error:
            assert str(error) == 'holdout_contamination'
        else:
            raise AssertionError('Raw-audio holdout guard must remain active')
        assert conn.execute('SELECT 1 FROM jobs WHERE id=%s', (job_id(allowed),)).fetchone() is None
        assert conn.execute('SELECT count(*) n FROM publications').fetchone()['n'] == 0

        # Agreement is only a report flag: negations, numbers, empty speech and
        # conflicting historical labels must retain their rejection decisions.
        cases = {
            'equal': (('Ёж, да!', 'еж да', 'ЕЖ: ДА.'), True, 'candidate_3of3'),
            'negation': (('Я не согласен', 'Я согласен', 'Я не согласен'),
                         False, 'rejected_disagreement'),
            'number': (('Оплачу 15 рублей', 'Оплачу 50 рублей', 'Оплачу 15 рублей'),
                       False, 'rejected_disagreement'),
            'empty': (('', '', ''), True, 'rejected_first'),
            'history': (('да', 'да', 'да'), True, 'rejected_prior_conflict'),
            'history-null': (('да', 'да', 'да'), True, 'rejected_prior_conflict'),
        }
        for label, (texts, _, _) in cases.items():
            for variant, text in zip(('verbatim', 'coverage', 'source_correction'), texts):
                row = dict(sample('agreement-' + label), prompt_variant=variant)
                ident = enqueue(conn, row)
                result = {'verbatim_text': text, 'language': 'ru', 'uncertain_spans': []}
                # The disposable outer transaction freezes now(); the review
                # cursor uses clock_timestamp(), so new fixtures need that clock.
                conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',result=%s,updated_at=clock_timestamp() WHERE id=%s",
                             (Jsonb(result), ident))
        prior = dict(sample('agreement-history'), pilot_id='legacy-history',
                     conversation_id='legacy-history')
        ident = enqueue(conn, prior)
        conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',result=%s,updated_at=clock_timestamp() WHERE id=%s",
                     (Jsonb({'verbatim_text': 'нет', 'language': 'ru', 'uncertain_spans': []}), ident))
        prior = dict(sample('agreement-history-null'), pilot_id=None,
                     conversation_id='legacy-null-history')
        ident = enqueue(conn, prior)
        conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',result=%s,updated_at=clock_timestamp() WHERE id=%s",
                     (Jsonb({'verbatim_text': 'нет', 'language': 'ru', 'uncertain_spans': []}), ident))
        assert reconcile()['third_passes_requested'] == 0
        actual = {row['sample_id']: row for payload in reports.payloads[-1:]
                  for row in payload['rows']}
        for label, (_, agreement, decision) in cases.items():
            row = actual['agreement-' + label]
            assert row['normalized_agreement'] is agreement, label
            assert row['decision'] == decision, label
            assert row['human_review_required'] is True
            assert row['automatic_publication'] is False
        assert conn.execute('SELECT count(*) n FROM publications').fetchone()['n'] == 0
        writes = reports.writes
        assert reconcile() == {'reports': [], 'third_passes_requested': 0}
        assert reports.writes == writes

        # Changed pilots remain historical evidence for each other. Filtering
        # all changed pilots at once would incorrectly approve these conflicts.
        current = sample('agreement-multi')
        other = dict(sample('other-current-audio'), sample_id='agreement-multi',
                     conversation_id='other-current', pilot_id=TRIAD_VERSION + '-other')
        for source, text in ((current, 'да'), (other, 'нет')):
            for variant in ('verbatim', 'coverage', 'source_correction'):
                ident = enqueue(conn, dict(source, prompt_variant=variant))
                conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',result=%s,updated_at=clock_timestamp() WHERE id=%s",
                             (Jsonb({'verbatim_text': text, 'language': 'ru', 'uncertain_spans': []}), ident))
        start = len(reports.payloads)
        assert reconcile()['third_passes_requested'] == 0
        actual = {(payload['summary']['pilot_id'], row['sample_id']): row
                  for payload in reports.payloads[start:] for row in payload['rows']}
        for source in (current, other):
            row = actual[(source['pilot_id'], source['sample_id'])]
            assert row['normalized_agreement'] is True
            assert row['decision'] == 'rejected_prior_conflict'
            assert row['human_review_required'] is True
            assert row['automatic_publication'] is False
    conn.rollback()
print('Batch guards and failed-report recovery passed; transaction rolled back')
