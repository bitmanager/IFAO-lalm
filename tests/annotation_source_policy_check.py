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
    for key, member, blocked in [
        ('staging/golos-balalaika/rev/test/shard.tar', 'audio.wav', True),
        ('staging/balalaika-longform/rev/data/validation/shard.tar', 'audio.wav', True),
        ('staging/source/mixed.tar', 'DEV/0001.wav', True),
        ('staging/balalaika-longform/rev/data/excluded/excluded-00000.tar', 'audio.flac', True),
        ('staging/source/mixed.tar', 'excluded/0001.wav', True),
        ('staging/source/train/shard.tar', 'train/excludedness.wav', False),
        ('staging/source/mixed.tar', 'test-0001.wav', True),
        ('staging/source/train/shard.tar', 'train/0001.wav', False),
        ('staging/source/train/shard.tar', 'train/testament.wav', False),
    ]:
        sample = json.dumps({'source_provenance': {'key': key, 'member': member}})
        row = conn.execute("INSERT INTO jobs(id,sample,state) VALUES ('split-check',%s::jsonb,'PENDING') "
                           'RETURNING state,error_code', (sample,)).fetchone()
        assert row['state'] == ('NEEDS_REVIEW' if blocked else 'PENDING')
        assert row['error_code'] == ('held_out_source_requires_split_adapter' if blocked else None)
    conn.rollback()
print('OpenSTT insert/retry policy checked; temporary objects rolled back')
