"""Build review-only conversation context from existing, timed ASR windows.

No inference requests or training publication. Channel IDs are not role labels.
"""
import hashlib
import json
import math


VERSION = 'conversation-window-history-v1'


def index_key(manifest_key, organization='Meteor'):
    from pathlib import PurePosixPath
    import re
    stem = PurePosixPath(manifest_key).stem
    if not re.fullmatch('[0-9a-f]{64}', stem):
        raise ValueError('invalid_source_manifest_key')
    return f'datasets/contextual-review/{organization.lower()}/{VERSION}/manifests/{stem}.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def build_records(rows, max_history=5):
    if not 1 <= max_history <= 20:
        raise ValueError('invalid_history_length')
    grouped = {}
    seen = set()
    for row in rows:
        sample, result = row['sample'], row['result']
        provenance = sample['source_provenance']
        identity = row['job_id']
        if identity in seen:
            raise ValueError('duplicate_context_job')
        seen.add(identity)
        start, end = (float(provenance[k]) for k in ('start_seconds', 'end_seconds'))
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end:
            raise ValueError('invalid_window_times')
        if not provenance.get('group_identity_verified') or not sample.get('conversation_id'):
            raise ValueError('unverified_conversation_identity')
        # Empty/foreign outputs provide neither targets nor context.
        if result.get('language') != 'ru' or not result.get('verbatim_text', '').strip():
            continue
        item = dict(row, start=start, end=end)
        grouped.setdefault(sample['conversation_id'], []).append(item)
    records = []
    for conversation, windows in sorted(grouped.items()):
        windows.sort(key=lambda r: (r['start'], r['end'], r['job_id']))
        for current in windows:
            sample, result = current['sample'], current['result']
            provenance = sample['source_provenance']
            # Both channels share one clock. Exclude current, overlapping and
            # future audio; uncertain predictions are not propagated as history.
            previous = [r for r in windows if r['job_id'] != current['job_id']
                        and r['end'] <= current['start']
                        and not r['result'].get('uncertain_spans')]
            previous.sort(key=lambda r: (r['end'], r['start'], r['job_id']))
            history = []
            for prior in previous[-max_history:]:
                p = prior['sample']['source_provenance']
                history.append({
                    'role': 'unknown', 'speaker_id': f"channel_{p['channel']}",
                    'start_ms': round(prior['start'] * 1000),
                    'end_ms': round(prior['end'] * 1000),
                    'text': prior['result']['verbatim_text'],
                    'source_job_id': prior['job_id'],
                    'source_result_sha256': digest(prior['result'])})
            records.append({
                'sample_id': sample['sample_id'], 'job_id': current['job_id'],
                'conversation_id': conversation,
                'audio_uri': f"s3://{sample['audio_bucket']}/{sample['audio_key']}",
                'audio_sha256': sample['audio_sha256'],
                'speaker_id': f"channel_{provenance['channel']}",
                'start_ms': round(current['start'] * 1000),
                'end_ms': round(current['end'] * 1000),
                'history': history,
                'target': {'verbatim_text': result['verbatim_text']},
                'metadata': {
                    'dataset_version': VERSION, 'history_source': 'gemini_window_predictions',
                    'history_unit': 'audio_window_not_aligned_utterance',
                    'source_result_sha256': digest(result),
                    'source_index_bucket': provenance['bucket'],
                    'source_index_key': provenance['key'],
                    'annotation_model': result['model'],
                    'source_contract_sha256': result['contract_sha256'],
                    'review_required': True, 'human_verified': False,
                    'training_release': False, 'channel_roles_verified': False,
                    'target_uncertain_spans': result.get('uncertain_spans', [])}})
    return records


def export_pilot(manifest_key, organization='Meteor', max_history=5):
    import io
    from datetime import datetime, timezone
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .store import connect, s3, dataset_bucket

    with connect() as conn:
        conn.execute("SET LOCAL statement_timeout='20s'")
        source = conn.execute("SELECT count(*) total, count(*) FILTER (WHERE state IN "
                              "('PENDING','RUNNING','RETRYABLE_FAILED')) unfinished "
                              "FROM jobs WHERE sample->>'source_manifest_key'=%s",
                              (manifest_key,)).fetchone()
        if not source['total'] or source['unfinished']:
            raise ValueError('source_manifest_not_terminal')
        rows = conn.execute("SELECT id AS job_id,sample,result FROM jobs "
                            "WHERE sample->>'source_manifest_key'=%s AND result IS NOT NULL "
                            "ORDER BY id", (manifest_key,)).fetchall()
    client, bucket = s3(), dataset_bucket()
    indices, selected = {}, []
    for row in rows:
        p = row['sample']['source_provenance']
        key = p['bucket'], p['key']
        if key not in indices:
            response = client.get_object(Bucket=key[0], Key=key[1])
            try:
                raw = response['Body'].read()
            finally:
                response['Body'].close()
            assert hashlib.sha256(raw).hexdigest() == p['source_index_sha256']
            indices[key] = json.loads(raw)
        index = indices[key]
        if index['source']['organization'] != organization:
            continue
        assert index['native_record']['dialog']['conversation_id'] == row['sample']['conversation_id']
        selected.append(row)
    records = build_records(selected, max_history)
    if not records:
        raise ValueError('no_contextual_records')
    output = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(records), output, compression='zstd')
    data = output.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    key = f'datasets/contextual-review/{organization.lower()}/{VERSION}/{sha}.parquet'
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType='application/vnd.apache.parquet')
    response = client.get_object(Bucket=bucket, Key=key)
    try:
        readback = response['Body'].read()
    finally:
        response['Body'].close()
    assert hashlib.sha256(readback).hexdigest() == sha
    assert pq.read_table(io.BytesIO(readback)).num_rows == len(records)
    receipt = {'at': datetime.now(timezone.utc).isoformat(), 'organization': organization,
               'manifest_key': manifest_key, 'dataset_key': key, 'dataset_sha256': sha,
               'rows': len(records), 'conversations': len({r['conversation_id'] for r in records}),
               'source_jobs_total': source['total'], 'source_unfinished_jobs': 0,
               'source_saved_jobs': len(rows),
               'rows_with_history': sum(bool(r['history']) for r in records),
               'full_parquet_sha256_verified': True, 'inference_requests': 0,
               'training_release': False, 'human_verified': False,
               'history_unit': 'prior_nonoverlapping_audio_windows', 'max_history': max_history}
    body = json.dumps(receipt, sort_keys=True).encode()
    receipt_key = f'diagnostics/contextual-review/{digest(receipt)}.json'
    client.put_object(Bucket=bucket, Key=receipt_key, Body=body, ContentType='application/json')
    # One canonical receipt per immutable input cohort; the notebook discovers
    # these small objects instead of listing historical Parquet versions.
    client.put_object(Bucket=bucket, Key=index_key(manifest_key, organization),
                      Body=body, ContentType='application/json')
    return dict(receipt, evidence=receipt_key)


def export_completed(organization='Meteor'):
    from botocore.exceptions import ClientError
    from .store import connect, s3, dataset_bucket
    with connect() as conn:
        conn.execute("SET LOCAL statement_timeout='30s'")
        rows = conn.execute("SELECT sample->>'source_manifest_key' manifest FROM jobs "
                            "WHERE sample->'source_provenance'->>'key' LIKE %s "
                            "AND sample->'source_provenance'->>'group_identity_source' "
                            "='native_current_relation_join' "
                            "AND sample->>'source_manifest_key' IS NOT NULL GROUP BY 1 "
                            "HAVING count(*) FILTER (WHERE state IN "
                            "('PENDING','RUNNING','RETRYABLE_FAILED'))=0 ORDER BY 1",
                            (f'staging/organization-calls-{organization.lower()}/%',)).fetchall()
    client, bucket = s3(), dataset_bucket()
    published, indexed, waiting = [], 0, 0
    for row in rows:
        manifest = row['manifest']
        try:
            client.head_object(Bucket=bucket, Key=index_key(manifest, organization))
        except ClientError as error:
            if error.response['ResponseMetadata']['HTTPStatusCode'] != 404:
                raise
        else:
            indexed += 1
            continue
        try:
            result = export_pilot(manifest, organization)
        except ValueError as error:
            if str(error) != 'source_manifest_not_terminal':
                raise
            waiting += 1
            continue
        published.append({k: result[k] for k in
                          ('manifest_key', 'dataset_key', 'rows', 'conversations', 'evidence')})
    return {'published': published, 'already_indexed': indexed,
            'waiting_full_manifest': waiting, 'inference_requests': 0}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--organization', default='Meteor')
    args = parser.parse_args()
    print(json.dumps(export_pilot(args.manifest, args.organization)))
