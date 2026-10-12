"""Read immutable inspection snapshots, shared by notebooks and pipeline diagnostics."""
import os, json, hashlib, tempfile
from pathlib import Path
import boto3

TABLES = {'sources': 'review_sources', 'passes': 'review_passes',
          'examples': 'review_examples', 'progress': 'review_progress',
          'publications': 'review_publications'}

def open_snapshot():
    import duckdb
    s3 = boto3.client('s3', endpoint_url=os.environ['S3_ENDPOINT'])
    bucket = os.environ.get('S3_DATASET_BUCKET', 'speech-datasets')
    snapshot = json.loads(s3.get_object(Bucket=bucket, Key='analytics/latest.json')['Body'].read())
    assert snapshot['purpose'] == 'inspection_only_not_training_release'


    cache = tempfile.TemporaryDirectory(prefix='speech-inspect-')
    db = duckdb.connect()

    def load_table(name):
        assert name in snapshot['tables']
        meta = snapshot['tables'][name]
        path = Path(cache.name) / (name + '.parquet')
        if not path.exists():
            s3.download_file(snapshot['bucket'], meta['key'], str(path))
            with path.open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == meta['sha256']
        db.read_parquet(str(path)).create_view(name, replace=True)

    for name in TABLES:
        load_table(name)
    return s3, snapshot, cache, db


def read_text_object(client, bucket, key, max_bytes=2 * 1024**2):
    """Bound downloads when inspecting request/response artifacts."""
    response = client.get_object(Bucket=bucket, Key=key)
    try:
        if response['ContentLength'] > max_bytes:
            raise ValueError('Object exceeds inspection size limit')
        return response['Body'].read(max_bytes + 1).decode('utf-8')
    finally:
        response['Body'].close()
