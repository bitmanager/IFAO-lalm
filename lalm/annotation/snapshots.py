"""Publish consistent, immutable Parquet snapshots for S3-native analysis."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from .store import connect, s3, dataset_bucket

from .inspection import TABLES


def arrow_type(oid):
    if oid in (20, 21, 23): return pa.int64()
    if oid in (700, 701): return pa.float64()
    if oid == 16: return pa.bool_()
    if oid == 1184: return pa.timestamp('us', tz='UTC')
    if oid == 1114: return pa.timestamp('us')
    return pa.string()


def publish_snapshots():
    client = s3()
    tables = {}
    with connect() as conn, tempfile.TemporaryDirectory(prefix='speech-snapshot-') as temp:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        stamp = conn.execute('SELECT transaction_timestamp() AS stamp').fetchone()['stamp'].isoformat()
        derived = {}
        for name, view in TABLES.items():
            path = Path(temp)/(name+'.parquet')
            if name in ('examples', 'progress'):
                if not derived:
                    from .inspection_aggregate import derive_tables
                    derived = derive_tables(Path(temp)/'passes.parquet', temp)
                path = derived[name]
                count = pq.read_metadata(path).num_rows
            else:
                count = _write_view(conn, view, name, path)
            with path.open('rb') as body:
                digest = hashlib.file_digest(body, 'sha256').hexdigest()
            key = f'analytics/tables/{name}/{digest}.parquet'
            with path.open('rb') as body:
                client.put_object(Bucket=dataset_bucket(), Key=key, Body=body)
            tables[name] = {'key': key, 'rows': count, 'sha256': digest}
    manifest = {'version': 2, 'snapshot_at': stamp, 'bucket': dataset_bucket(),
                'purpose': 'inspection_only_not_training_release', 'tables': tables,
                'aggregation': {'engine': 'duckdb', 'version': version('duckdb'),
                                'original_annotation_json': 'typed_provenance_json'}}
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    key = 'analytics/snapshots/'+hashlib.sha256(payload).hexdigest()+'.json'
    client.put_object(Bucket=dataset_bucket(), Key=key, Body=payload)
    # Publish discovery pointer only after all immutable artifacts are present.
    client.put_object(Bucket=dataset_bucket(), Key='analytics/latest.json', Body=payload)
    return {'snapshot_key': key, 'rows': {k: v['rows'] for k,v in tables.items()}}


def _write_view(conn, view, name, path):
    with conn.cursor(name='snapshot_'+name) as cursor:
        cursor.execute('SELECT * FROM '+view)
        rows = cursor.fetchmany(10000)
        schema = pa.schema([(d.name, arrow_type(d.type_code)) for d in cursor.description])
        count = 0
        with pq.ParquetWriter(path, schema, compression='zstd') as writer:
            while rows:
                for row in rows:
                    for field in schema:
                        value = row[field.name]
                        if value is not None and pa.types.is_string(field.type):
                            row[field.name] = (json.dumps(value, ensure_ascii=False, sort_keys=True)
                                              if isinstance(value, (dict, list)) else str(value))
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                count += len(rows)
                rows = cursor.fetchmany(10000)
    return count
