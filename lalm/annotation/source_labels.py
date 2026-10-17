"""Read original Golos labels by encoded-audio SHA256; S3 remains authoritative."""
import hashlib
import json
import re
import sqlite3
import tempfile
from functools import lru_cache


def build_index(conn, lines):
    conn.execute('CREATE TABLE labels (sha TEXT PRIMARY KEY, records TEXT NOT NULL)')
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        sha = row.get('source_audio_sha256')
        if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
            raise ValueError('invalid_source_index_hash')
        old = conn.execute('SELECT records FROM labels WHERE sha=?', (sha,)).fetchone()
        records = json.loads(old[0]) if old else []
        # Keep provenance and eligibility flags, including duplicate-source records.
        records.append(row)
        conn.execute('INSERT OR REPLACE INTO labels VALUES (?,?)',
                     (sha, json.dumps(records, ensure_ascii=False)))
    conn.commit()


def lookup(conn, sha, reference):
    row = conn.execute('SELECT records FROM labels WHERE sha=?', (sha,)).fetchone()
    if row is None:
        return None
    records = json.loads(row[0])
    texts = set()
    for record in records:
        for field in ('original_metadata', 'original_manifest_record'):
            value = record.get(field) or {}
            text = value.get('text') if isinstance(value, dict) else None
            if isinstance(text, str) and text.strip():
                texts.add(text.strip())
    return {'text': next(iter(texts)) if len(texts) == 1 else None,
            'annotation_source': 'original_source_index',
            'text_conflict': len(texts) > 1, 'source_index': reference,
            'lookup_audio_sha256': sha, 'records': records}


@lru_cache(maxsize=1)
def _index(bucket, key):
    from .store import s3
    client = s3()
    head = client.head_object(Bucket=bucket, Key=key)
    temporary = tempfile.TemporaryDirectory(prefix='source-label-index-')
    conn = sqlite3.connect(temporary.name + '/labels.sqlite')
    body = client.get_object(Bucket=bucket, Key=key, IfMatch=head['ETag'])['Body']
    try:
        build_index(conn, body.iter_lines())
    except Exception:
        conn.close()
        temporary.cleanup()
        raise
    finally:
        body.close()
    # The local SQLite file is a disposable lookup cache, never a dataset store.
    return conn, temporary, {'bucket': bucket, 'key': key, 'etag': head['ETag']}


def original_label(asset, raw):
    key = asset['object_key']
    if not key.startswith('staging/golos-farfield-train-first124003/') or '/audio/' not in key:
        return None
    from .store import dataset_bucket
    conn, _, reference = _index(dataset_bucket(), key.rsplit('/audio/', 1)[0] + '/source-index.jsonl')
    return lookup(conn, hashlib.sha256(raw).hexdigest(), reference)
