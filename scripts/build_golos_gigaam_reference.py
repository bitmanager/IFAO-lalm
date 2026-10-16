"""Recover existing CTC labels from QA metadata; never enqueue QA/audio copies."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from lalm.annotation.pilot_source import S3File
from lalm.annotation.store import connect, s3, dataset_bucket


WEIGHTS = 'a20a9b5b8ed929c4a60141753cd3dd1284138879a46fb4a1cef6430ec019a02f'


def extract(row, source_keys):
    """Use only explicit full-source CTC metadata, never the generated answer."""
    meta = json.loads(row['metadata_json'])
    if not isinstance(meta, dict):
        raise ValueError('invalid_ctc_metadata')
    uri = urlsplit(meta['audio_uri'])
    key = uri.path.lstrip('/')
    locator = meta['audio_locator']
    duration, rate, frames = meta['duration'], row['sample_rate'], row['num_samples']
    if (row.get('corpus') != 'golos' or row.get('split') != 'train'
            or meta.get('corpus') != 'golos' or meta.get('split') != 'train'
            or meta.get('transcript_kind') != 'GigaAM-v3-CTC pseudo-label'
            or meta.get('transcript_weights_sha256') != WEIGHTS
            or uri.scheme != 's3' or uri.netloc != 'speech-corpora'
            or key not in source_keys or uri.query or uri.fragment
            or not isinstance(locator, str) or not locator.startswith('member=')
            or meta.get('start') != 0
            or type(rate) is not int or rate <= 0
            or type(frames) is not int or frames <= 0
            or not isinstance(duration, (float, int)) or not math.isfinite(duration)
            or duration <= 0 or abs(duration - frames / rate) > 1 / rate):
        raise ValueError('unsupported_ctc_scope')
    member = locator.removeprefix('member=')
    sha, text = meta['audio_sha256'], meta['gigaam_transcript']
    if (not member or member.startswith('/') or '..' in member.split('/')
            or not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha)
            or not isinstance(text, str) or not 0 < len(text.strip()) <= 12000):
        raise ValueError('invalid_ctc_reference')
    return {'source_s3_key': key, 'source_row': member, 'audio_sha256': sha,
            'gigaam_transcription': text, 'gigaam_model': 'v3_ctc',
            'gigaam_weights_sha256': WEIGHTS, 'source_duration': float(duration)}


def build(max_shards=None):
    client = s3()
    with connect() as conn:
        assets = conn.execute("SELECT bucket,object_key,etag FROM source_assets WHERE "
            "object_key LIKE 'staging/qwen-answer-nohistory-v1-20261015/shards/train/golos-train-%' "
            "ORDER BY object_key").fetchall()
        sources = conn.execute("SELECT object_key,etag FROM source_assets WHERE "
            "split_part(object_key,'/',2)='golos' AND state IN ('QUEUED','PREPARING','PREPARED') "
            "AND object_key ~ '/train/train-[0-9]+[.]tar$'").fetchall()
    selected = assets if max_shards is None else assets[:max_shards]
    if not selected or not sources:
        raise ValueError('missing_source_assets')
    source_keys = {r['object_key'] for r in sources}
    labels, conflicts, counts, provenance = {}, set(), Counter(), []
    for asset in selected:
        with S3File(client, asset['bucket'], asset['object_key']) as remote:
            if remote.etag != asset['etag']:
                raise ValueError('qa_source_changed')
            parquet = pq.ParquetFile(remote)
            for batch in parquet.iter_batches(batch_size=1000,
                    columns=['metadata_json', 'sample_rate', 'num_samples', 'corpus', 'split']):
                for row in batch.to_pylist():
                    counts['input_rows'] += 1
                    try:
                        label = extract(row, source_keys)
                    except (ValueError, KeyError, TypeError):
                        counts['invalid_or_unsupported_rows'] += 1
                        continue
                    locator = label['source_s3_key'], label['source_row']
                    if locator in conflicts:
                        counts['conflicting_rows'] += 1
                    elif locator in labels and labels[locator] != label:
                        conflicts.add(locator)
                        labels.pop(locator)
                        counts['conflicting_rows'] += 1
                    elif locator in labels:
                        counts['identical_duplicate_rows'] += 1
                    else:
                        labels[locator] = label
            provenance.append(dict(asset, metadata_rows=parquet.metadata.num_rows,
                                   range_bytes_read=remote.bytes_read))
        print(json.dumps({'shards_read': len(provenance), 'shards_selected': len(selected),
                          'unique_references': len(labels), 'counts': dict(counts)}), flush=True)
    if not labels:
        raise ValueError('no_usable_ctc_references')
    with tempfile.TemporaryDirectory(prefix='golos-ctc-reference-') as tmp:
        path = Path(tmp) / 'references.parquet'
        rows = [labels[k] for k in sorted(labels)]
        pq.write_table(pa.Table.from_pylist(rows), path, compression='zstd')
        with path.open('rb') as body:
            sha = hashlib.file_digest(body, 'sha256').hexdigest()
        key = 'diagnostics/gigaam-single-pass-20261016/golos-reference/' + sha + '.parquet'
        with path.open('rb') as body:
            client.put_object(Bucket=dataset_bucket(), Key=key, Body=body)
        response = client.get_object(Bucket=dataset_bucket(), Key=key)
        try:
            assert hashlib.file_digest(response['Body'], 'sha256').hexdigest() == sha
        finally:
            response['Body'].close()
    report = {'purpose': 'existing_ctc_reference_not_admitted_not_training_release',
              'captured_at': datetime.now(timezone.utc).isoformat(),
              'build_version': 'golos-qa-ctc-reference-v1', 'pyarrow_version': pa.__version__,
              'reference_key': key, 'reference_sha256': sha, 'reference_rows': len(labels),
              'all_selected_qa_shards_read': len(provenance) == len(assets),
              'source_qa_assets': provenance, 'source_audio_assets': sources,
              'conflicting_locators_excluded': len(conflicts), 'counts': dict(counts),
              'inference_requests': 0, 'audio_downloaded': False, 'work_enqueued': False}
    raw = json.dumps(report, sort_keys=True).encode()
    proof = 'diagnostics/gigaam-single-pass-20261016/golos-reference-build-' + hashlib.sha256(raw).hexdigest() + '.json'
    client.put_object(Bucket=dataset_bucket(), Key=proof, Body=raw, ContentType='application/json')
    print(json.dumps({'evidence': proof, 'reference_key': key, 'reference_sha256': sha,
                      'reference_rows': len(labels), 'counts': dict(counts),
                      'conflicting_locators_excluded': len(conflicts)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-shards', type=int)
    args = parser.parse_args()
    if args.max_shards is not None and args.max_shards < 1:
        parser.error('--max-shards must be positive')
    build(args.max_shards)
