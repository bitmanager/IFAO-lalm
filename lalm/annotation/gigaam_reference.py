"""Pinned CTC references; source text never enters the blind Gemini prompt."""
from functools import lru_cache
import hashlib
import io

import pyarrow.parquet as pq

from .store import s3, dataset_bucket
from .contract import GIGAAM_REFERENCES


REFERENCE_KEY, REFERENCE_SHA = GIGAAM_REFERENCES['sova']
GOLOS_REFERENCE_KEY, GOLOS_REFERENCE_SHA = GIGAAM_REFERENCES['golos']
APPROVED_REFERENCES = set(GIGAAM_REFERENCES.values())
POLICY = 'gigaam-single-pass-v1'


@lru_cache(maxsize=2)
def references(key=REFERENCE_KEY, sha=REFERENCE_SHA):
    if (key, sha) not in APPROVED_REFERENCES:
        raise ValueError('unapproved_gigaam_reference')
    response = s3().get_object(Bucket=dataset_bucket(), Key=key)
    try:
        raw = response['Body'].read()
    finally:
        response['Body'].close()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('gigaam_reference_checksum_mismatch')
    labels = {}
    for row in pq.read_table(io.BytesIO(raw)).to_pylist():
        locator = (row['source_s3_key'], str(row['source_row']))
        if locator in labels:
            raise ValueError('duplicate_gigaam_reference')
        labels[locator] = row
    return labels


def requires_reference(asset):
    key = asset['object_key']
    return key.startswith('staging/golos/') and '/crowd/train/' in key


def load_reference_index(asset):
    # Fail once at the asset boundary; a corrupt index must not be treated as
    # thousands of bad audio rows by the per-record exception handler.
    if asset['object_key'].startswith('staging/sova-rudevices/'):
        references()
    elif requires_reference(asset):
        references(GOLOS_REFERENCE_KEY, GOLOS_REFERENCE_SHA)


def reference(asset, member, source_sha):
    if asset['object_key'].startswith('staging/sova-rudevices/'):
        key, sha = REFERENCE_KEY, REFERENCE_SHA
    elif requires_reference(asset):
        key, sha = GOLOS_REFERENCE_KEY, GOLOS_REFERENCE_SHA
    else:
        return None
    row = references(key, sha).get((asset['object_key'], str(member)))
    if not row or row['audio_sha256'] != source_sha:
        return None
    if not isinstance(row['gigaam_transcription'], str) or not row['gigaam_transcription'].strip():
        return None
    result = {'text': row['gigaam_transcription'], 'model': row['gigaam_model'],
            'weights_sha256': row['gigaam_weights_sha256'],
            'source_audio_sha256': source_sha, 'reference_key': key,
            'reference_sha256': sha}
    if 'source_duration' in row:
        result['source_duration'] = row['source_duration']
    return result
