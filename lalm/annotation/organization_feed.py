"""Feed explicitly selected native calls through existing manifests and Airflow.

One bounded pilot at a time; checkpoints and the queue retain restart safety.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import time


VERSION = 'native-call-feed-v1'


def checkpoint_key(selection, metadata_hashes):
    value = {'version': VERSION, 'selection': selection, 'metadata': metadata_hashes}
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    return 'staging/organization-call-feed/' + hashlib.sha256(raw).hexdigest() + '.json'


def feed(metadata, selection, audio_directory, max_pending=6000):
    from botocore.exceptions import ClientError
    from .organization_calls import stage_pilot
    from .prepare import pending_work
    from .store import connect, s3, dataset_bucket

    if not 1 <= max_pending <= 100000:
        raise ValueError('invalid_pending_limit')
    selected = json.loads(Path(selection).read_text())
    if not isinstance(selected, list) or not selected:
        raise ValueError('empty_native_selection')
    conversations = [r['conversation_id'] for r in selected]
    hashes = [r['sha256'] for r in selected]
    if len(set(conversations)) != len(selected) or len(set(hashes)) != len(selected):
        raise ValueError('duplicate_native_selection')
    metadata_hashes = {name: hashlib.sha256((Path(metadata) / name).read_bytes()).hexdigest()
                       for name in ('dialogs-native.csv', 'relations-native.csv')}
    client, bucket = s3(), dataset_bucket()
    # One feeder can submit native batches. PostgreSQL releases this session
    # lock after process/container termination; no persistent maintenance hold.
    with connect() as lock:
        lock.autocommit = True
        if not lock.execute('SELECT pg_try_advisory_lock(173923) ok').fetchone()['ok']:
            raise RuntimeError('native_call_feeder_already_running')
        for offset in range(0, len(selected), 12):
            batch = selected[offset:offset + 12]
            key = checkpoint_key(batch, metadata_hashes)
            try:
                response = client.get_object(Bucket=bucket, Key=key)
            except ClientError as error:
                if error.response['ResponseMetadata']['HTTPStatusCode'] != 404:
                    raise
            else:
                try:
                    receipt = json.loads(response['Body'].read())
                finally:
                    response['Body'].close()
                manifest = receipt['manifest_key']
                response = client.get_object(Bucket=bucket, Key=manifest)
                try:
                    raw = response['Body'].read()
                finally:
                    response['Body'].close()
                assert hashlib.sha256(raw).hexdigest() == Path(manifest).stem
                with connect() as conn:
                    registered = conn.execute('SELECT row_count FROM prepared_manifests '
                                              'WHERE object_key=%s', (manifest,)).fetchone()
                assert registered and registered['row_count'] == receipt['segments']
                print(json.dumps({'event': 'native_batch_already_staged', 'offset': offset,
                                  'calls': len(batch), 'manifest_key': manifest}), flush=True)
                continue
            while True:
                with connect() as conn:
                    pending = pending_work(conn)
                if pending < max_pending:
                    break
                print(json.dumps({'event': 'native_feed_backpressure', 'pending': pending,
                                  'limit': max_pending, 'offset': offset}), flush=True)
                time.sleep(30)
            # stage_pilot publishes only after its native joins, encoded hashes,
            # segmentation, conversation/audio holdout and duplicate checks pass.
            with tempfile.TemporaryDirectory(prefix='native-feed-') as tmp:
                path = Path(tmp) / 'selection.json'
                path.write_text(json.dumps(batch, ensure_ascii=False))
                receipt = stage_pilot(metadata, path, audio_directory)
            client.put_object(Bucket=bucket, Key=key,
                              Body=json.dumps(receipt, sort_keys=True).encode(),
                              ContentType='application/json')
            print(json.dumps({'event': 'native_batch_staged', 'offset': offset,
                              'checkpoint_key': key, **receipt}), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--audio-directory', required=True)
    parser.add_argument('--max-pending', type=int, default=6000)
    args = parser.parse_args()
    feed(args.metadata, args.selection, args.audio_directory, args.max_pending)
