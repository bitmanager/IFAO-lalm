"""Read immutable inspection snapshots, shared by notebooks and pipeline diagnostics."""
import os, json, hashlib, tempfile, re
from pathlib import Path

TABLES = {'sources': 'review_sources', 'passes': 'review_passes',
          'examples': 'review_examples', 'progress': 'review_progress',
          'publications': 'review_publications'}

def open_snapshot():
    import boto3
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


def normalized_words(text):
    return re.findall(r"[\w]+", text.lower().replace('ё', 'е'))


def agreement_summary(row):
    """Agreement of all available nonempty passes; this is not a quality label."""
    columns = ('gemini_verbatim', 'gemini_coverage', 'gemini_literal_audit',
               'gemini_source_correction')
    texts = [normalized_words(row.get(c)) for c in columns
             if isinstance(row.get(c), str) and normalized_words(row.get(c))]
    count = len(texts)
    equal = count >= 2 and all(text == texts[0] for text in texts[1:])
    status = (f'{count} совпали' if equal else
              'расхождение' if count >= 2 else 'неполный ответ')
    reference = row.get('gemini_source_correction')
    return {'agreement_status': status, 'agreement_passes': count,
            'three_pass_agreement': equal and count == 3,
            'source_conditioned_agreement': isinstance(reference, str) and bool(reference.strip())}
