"""Pinned SOVA CTC references; source text never enters the blind Gemini prompt."""
from functools import lru_cache
import hashlib
import io

import pyarrow.parquet as pq

from .store import s3, dataset_bucket


REFERENCE_SHA = 'cc693e4862f452a2d7a56f586b28acec3e38e578a5ad89302a958d84d2f5aedd'
REFERENCE_KEY = 'diagnostics/gigaam-ctc-sova-20261015/' + REFERENCE_SHA + '.parquet'
POLICY = 'gigaam-single-pass-v1'


@lru_cache(maxsize=1)
def references():
    response = s3().get_object(Bucket=dataset_bucket(), Key=REFERENCE_KEY)
    try:
        raw = response['Body'].read()
    finally:
        response['Body'].close()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA:
        raise ValueError('gigaam_reference_checksum_mismatch')
    labels = {}
    for row in pq.read_table(io.BytesIO(raw)).to_pylist():
        locator = (row['source_s3_key'], str(row['source_row']))
        if locator in labels:
            raise ValueError('duplicate_gigaam_reference')
        labels[locator] = row
    return labels


def reference(asset, member, source_sha):
    if not asset['object_key'].startswith('staging/sova-rudevices/'):
        return None
    row = references().get((asset['object_key'], str(member)))
    if not row or row['audio_sha256'] != source_sha:
        return None
    if not isinstance(row['gigaam_transcription'], str) or not row['gigaam_transcription'].strip():
        return None
    return {'text': row['gigaam_transcription'], 'model': row['gigaam_model'],
            'weights_sha256': row['gigaam_weights_sha256'],
            'source_audio_sha256': source_sha, 'reference_key': REFERENCE_KEY,
            'reference_sha256': REFERENCE_SHA}
