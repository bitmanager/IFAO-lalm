"""Inventory accessible source objects without reading protected evaluation data."""
import hashlib
from .store import connect, s3


def discover():
    client = s3()
    counts = {}
    seen = 0
    for page in client.get_paginator('list_objects_v2').paginate(Bucket='speech-corpora', Prefix='staging/'):
        with connect() as conn:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.parquet'): codec = 'parquet'
                elif key.endswith(('.tar', '.tar.gz', '.tgz')): codec = 'tar'
                elif key.endswith(('.wav', '.flac', '.mp3', '.ogg', '.opus', '.m4a')): codec = 'audio'
                else: continue
                ident = hashlib.sha256(('speech-corpora\0'+key+'\0'+obj['ETag']).encode()).hexdigest()
                conn.execute('INSERT INTO source_assets(id,bucket,object_key,etag,bytes,codec) '
                             'VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                             (ident, 'speech-corpora', key, obj['ETag'], obj['Size'], codec))
                counts[codec] = counts.get(codec, 0)+1; seen += 1
    return {'source_objects': seen, 'by_codec': counts}


def queue_sources():
    with connect() as conn:
        rows = conn.execute("UPDATE source_assets SET state='QUEUED',updated_at=now() "
                            "WHERE state='DISCOVERED' RETURNING id").fetchall()
        counts = conn.execute('SELECT state,codec,count(*) AS n FROM source_assets GROUP BY state,codec').fetchall()
    return {'newly_queued': len(rows), 'sources': counts}
