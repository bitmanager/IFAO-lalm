"""Exercise the real policy trigger on temporary PostgreSQL objects only."""
import json
from pathlib import Path

from lalm.annotation.store import connect


sql = Path('lalm/annotation/source_policy.sql').read_text().replace(
    'FUNCTION exclude_openstt_annotation_jobs',
    'FUNCTION pg_temp.exclude_openstt_annotation_jobs')
with connect() as conn:
    conn.execute('CREATE TEMP TABLE jobs '
                 '(id text, sample jsonb, state text, error_code text) ON COMMIT DROP')
    conn.execute(sql)
    for corpus in ['openstt-phone1', 'openstt-phone2', 'openstt-balalaika',
                   'openstt-future', 'balalaika', 'mtedx-ru-hf']:
        sample = json.dumps({'source_provenance': {'key': f'staging/{corpus}/source'}})
        for state in ['PENDING', 'RETRYABLE_FAILED']:
            row = conn.execute('INSERT INTO jobs(id,sample,state) VALUES (%s,%s::jsonb,%s) '
                               'RETURNING state,error_code', (corpus, sample, state)).fetchone()
            if corpus.startswith('openstt-'):
                assert row['state'] == 'NEEDS_REVIEW'
                assert row['error_code'] == 'corpus_excluded_by_user'
            else:
                assert row['state'] == state and row['error_code'] is None
    row = conn.execute("UPDATE jobs SET state='PENDING' WHERE id='openstt-phone1' "
                       'RETURNING state').fetchone()
    assert row['state'] == 'NEEDS_REVIEW'
    conn.rollback()
print('OpenSTT insert/retry policy checked; temporary objects rolled back')
