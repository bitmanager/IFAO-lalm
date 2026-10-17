"""Prefer SOVA and Golos with CTC references, then resume other sources."""


def select_source(conn):
    return conn.execute("""
        WITH activity AS (
            SELECT split_part(object_key,'/',2) AS corpus,
                   max(updated_at) FILTER (
                       WHERE cursor_json ? 'rows'
                          OR state IN ('PREPARED','NEEDS_REVIEW','EXCLUDED_HOLDOUT')
                   ) AS last_served
            FROM source_assets GROUP BY 1
        )
        SELECT a.* FROM source_assets a
        JOIN activity c ON c.corpus=split_part(a.object_key,'/',2)
        WHERE a.state IN ('QUEUED','PREPARING')
          AND split_part(a.object_key,'/',2) NOT LIKE 'openstt-%'
          AND a.object_key !~* '(^|/)(test|valid|validation|dev|eval|evaluation|excluded)(/|[-_.]|$)'
          -- The generic preparer assigns train to every row. MTEDx's archive
          -- mixes official splits; only explicitly named HF train shards fit.
          AND split_part(a.object_key,'/',2) <> 'mtedx-ru'
          AND (split_part(a.object_key,'/',2) <> 'mtedx-ru-hf'
               OR a.object_key ~ '/train-[0-9]+-of-[0-9]+[.]parquet$')
        ORDER BY (c.corpus='sova-rudevices') DESC,
                 (c.corpus='golos' AND a.object_key LIKE '%/crowd/train/%') DESC,
                 (a.state='PREPARING') DESC,
                 COALESCE(c.last_served,'epoch'::timestamptz), a.created_at, a.id
        FOR UPDATE OF a SKIP LOCKED LIMIT 1
    """).fetchone()
