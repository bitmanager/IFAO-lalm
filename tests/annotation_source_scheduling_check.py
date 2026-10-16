"""Run against PostgreSQL; all fixtures are temporary and rolled back."""
from lalm.annotation.store import connect
from lalm.annotation.source_scheduling import select_source


with connect() as conn:
    conn.execute('CREATE TEMP TABLE source_assets (LIKE public.source_assets INCLUDING ALL) ON COMMIT DROP')
    for ident, corpus, state, stamp in [
        ('o1', 'openstt-phone2', 'PREPARING', '2025-01-01'),
        ('o2', 'openstt-balalaika', 'QUEUED', '2025-01-01'),
        ('o3', 'openstt-phone1', 'QUEUED', '2025-01-01'),
        ('o4', 'openstt-future', 'QUEUED', '2025-01-01'),
        ('s1', 'sova-rudevices', 'PREPARING', '2026-01-01'),
        ('s2', 'sova-rudevices', 'QUEUED', '2026-01-01'),
        ('g1', 'golos', 'QUEUED', '2026-01-02'),
        ('g2', 'golos', 'QUEUED', '2026-01-02'),
        ('b1', 'balalaika', 'QUEUED', '2026-01-03'),
    ]:
        conn.execute('INSERT INTO source_assets(id,bucket,object_key,etag,bytes,codec,state,created_at) '
                     'VALUES (%s,%s,%s,%s,1,%s,%s,%s)',
                     (ident, 'test', f'staging/{corpus}/{ident}', 'test', 'audio', state, stamp))
    assert select_source(conn)['id'] == 's1', 'Must resume the in-flight source'
    conn.execute("UPDATE source_assets SET state='PREPARED',updated_at=now() WHERE id='s1'")
    assert select_source(conn)['id'] == 'g1', 'Must not prioritize all SOVA objects'
    conn.execute("UPDATE source_assets SET state='PREPARED',updated_at=now() WHERE id='g1'")
    assert select_source(conn)['id'] == 'b1', 'Must rotate rather than exhaust Golos'
    conn.execute("UPDATE source_assets SET state='PREPARED'")
    for ident, key in [
        ('mt', 'staging/mtedx-ru-hf/revision/ru/test-00000-of-00001.parquet'),
        ('mv', 'staging/mtedx-ru-hf/revision/ru/valid-00000-of-00001.parquet'),
        ('ma', 'staging/mtedx-ru/revision/mtedx_ru.tgz'),
        ('mu', 'staging/mtedx-ru-hf/revision/ru/unknown.parquet'),
        ('mn', 'staging/mtedx-ru-hf/revision/ru/train-00000-of-00018.parquet'),
    ]:
        conn.execute('INSERT INTO source_assets(id,bucket,object_key,etag,bytes,codec,state) '
                     "VALUES (%s,'test',%s,'test',1,'parquet','QUEUED')", (ident, key))
    assert select_source(conn)['id'] == 'mn', 'Only explicit MTEDx train shards may use generic preparation'
    conn.execute("UPDATE source_assets SET state='PREPARED' WHERE id='mn'")
    assert select_source(conn) is None, 'Must not fall through to MTEDx held-out or mixed inputs'
    for ident, key in [
        ('gt', 'staging/golos-balalaika/revision/test/shard_000.tar'),
        ('lv', 'staging/balalaika-longform/revision/data/validation/validation-00000.tar'),
        ('gd', 'staging/golos/revision/DEV/audio.tar'),
        ('le', 'staging/balalaika-longform/revision/data/excluded/excluded-00000.tar'),
        ('gn', 'staging/golos/revision/train/train-00000.tar'),
    ]:
        conn.execute('INSERT INTO source_assets(id,bucket,object_key,etag,bytes,codec,state) '
                     "VALUES (%s,'test',%s,'test',1,'tar','QUEUED')", (ident, key))
    assert select_source(conn)['id'] == 'gn', 'Must preserve train inputs while excluding named held-out paths'
    conn.execute("UPDATE source_assets SET state='PREPARED' WHERE id='gn'")
    assert select_source(conn) is None
    conn.rollback()
print('Source scheduling verified on PostgreSQL; fixtures rolled back')
