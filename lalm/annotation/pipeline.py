"""Small Airflow tasks: S3 manifests in, durable queue, immutable Parquet out."""
import hashlib
import io
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from .contract import CONTRACT_SHA
from .queue import enqueue, protected, recover_expired
from .store import bucket, connect, s3


def ingest():
    client = s3()
    # Required even when empty: operator explicitly establishes the benchmark registry.
    holdout = client.get_object(Bucket=bucket(), Key='benchmarks/holdouts.json')
    registry = json.loads(holdout['Body'].read())
    if set(registry) != {'conversation_ids', 'audio_sha256'}:
        raise ValueError('invalid_holdout_registry')
    with connect() as conn:
        conn.execute('SELECT pg_advisory_xact_lock(173921)')
        for field, kind in [('conversation_ids', 'conversation_id'), ('audio_sha256', 'audio_sha256')]:
            if not isinstance(registry[field], list) or not all(isinstance(x, str) and x for x in registry[field]):
                raise ValueError('invalid_holdout_registry')
            for value in registry[field]:
                conn.execute('INSERT INTO holdouts(kind,value) VALUES (%s,%s) ON CONFLICT DO NOTHING', (kind, value))
    count = 0
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket(), Prefix='incoming/'):
        for item in page.get('Contents', []):
            if not item['Key'].endswith('.jsonl'):
                continue
            body = client.get_object(Bucket=bucket(), Key=item['Key'])['Body'].read()
            manifest_sha = hashlib.sha256(body).hexdigest()
            with connect() as conn:
                conn.execute('SELECT pg_advisory_xact_lock(173921)')
                old = conn.execute('SELECT sha256 FROM source_manifests WHERE object_key=%s', (item['Key'],)).fetchone()
                if old:
                    if old['sha256'] != manifest_sha:
                        raise ValueError('source_manifest_mutated')
                    continue
                imported = 0
                for line in body.splitlines():
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    sample['source_manifest_sha256'] = manifest_sha
                    sample['source_manifest_key'] = item['Key']
                    enqueue(conn, sample)
                    count += 1
                    imported += 1
                conn.execute('INSERT INTO source_manifests(object_key,sha256,row_count) VALUES (%s,%s,%s)',
                             (item['Key'], manifest_sha, imported))
    return {'seen': count, 'expired_quarantined': recover_expired()}


def publish():
    client = s3()
    with connect() as conn:
        conn.execute('SELECT pg_advisory_xact_lock(173921)')
        jobs = conn.execute("SELECT * FROM jobs WHERE state='VALIDATED' ORDER BY id FOR UPDATE LIMIT 1000").fetchall()
        rows, ids = [], []
        for job in jobs:
            sample = job['sample']
            if protected(conn, sample):
                conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',error_code='holdout_contamination' WHERE id=%s", (job['id'],))
                continue
            result = job['result']
            ids.append(job['id'])
            rows.append({'job_id': job['id'], 'sample_id': sample['sample_id'],
                         'conversation_id': sample['conversation_id'], 'split': sample['split'],
                         'audio_key': sample['audio_key'], 'audio_sha256': sample['audio_sha256'],
                         'verbatim_text': result['verbatim_text'], 'language': result['language'],
                         'annotation_json': json.dumps(result, ensure_ascii=False),
                         'provenance_json': json.dumps(sample, ensure_ascii=False)})
        if not rows:
            return {'published': 0}
        ident = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        key = f'datasets/annotations/{ident}/manifest.parquet'
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows), buf)
        payload = buf.getvalue()
        try:
            client.put_object(Bucket=bucket(), Key=key, Body=payload, IfNoneMatch='*')
        except client.exceptions.ClientError as error:
            if error.response['ResponseMetadata']['HTTPStatusCode'] != 412:
                raise
            existing = client.get_object(Bucket=bucket(), Key=key)['Body'].read()
            if existing != payload:
                raise ValueError('immutable_manifest_conflict') from error
        conn.execute('INSERT INTO publications(id,manifest_key,count) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING',
                     (ident, key, len(rows)))
        conn.execute("UPDATE jobs SET state='PUBLISHED',updated_at=now() WHERE id=ANY(%s)", (ids,))
        return {'published': len(rows), 'manifest_key': key}


def status():
    with connect() as conn:
        counts = conn.execute('SELECT state,count(*) AS n FROM jobs GROUP BY state').fetchall()
        accounts = conn.execute('SELECT id,enabled,concurrency,blocked_reason,next_request_at FROM accounts ORDER BY id').fetchall()
    return {'jobs': counts, 'accounts': accounts, 'contract_sha256': CONTRACT_SHA}
