"""Run against PostgreSQL; all fixtures are temporary and rolled back."""
from lalm.annotation.store import connect
from lalm.annotation.source_scheduling import select_source


with connect() as conn:
    conn.execute('CREATE TEMP TABLE source_assets (LIKE public.source_assets INCLUDING ALL) ON COMMIT DROP')
    for ident, corpus, state, stamp in [
        ('o1', 'openstt-phone2', 'PREPARING', '2025-01-01'),
        ('o2', 'openstt-balalaika', 'QUEUED', '2025-01-01'),
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
    conn.rollback()
print('Source scheduling verified on PostgreSQL; fixtures rolled back')
