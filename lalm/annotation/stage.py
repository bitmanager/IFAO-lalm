"""Bridge existing assets.jsonl + calls.jsonl + speech QC to the S3 inbox.

Run next to the existing collector (or against its read-only mounted snapshot).
The benchmark registry must be installed before this command; no automatic split.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .contract import PREPROCESSING, validate_sample
from .store import bucket, s3


def stage(root, qc_path, split):
    client = s3()
    holdouts = json.loads(client.get_object(Bucket=bucket(), Key='benchmarks/holdouts.json')['Body'].read())
    excluded_calls, excluded_hashes = set(holdouts['conversation_ids']), set(holdouts['audio_sha256'])
    checks = json.loads(qc_path.read_text())
    if checks.get('complete') is not True:
        raise ValueError('incomplete_qc')
    qc = {r['audio_sha256']: r for r in checks['results']}
    assets = [json.loads(x) for x in (root / 'assets.jsonl').read_text().splitlines() if x]
    rows, held = [], 0
    for asset in assets:
        if asset.get('status') != 'complete':
            held += 1
            continue
        h = asset['user_sha256']
        check = qc.get(h, {})
        calls = asset['call_ids']
        # Hold every alias if any alias or the underlying audio is benchmark material.
        if h in excluded_hashes or set(calls) & excluded_calls or check.get('speech_detected') is not True:
            held += 1
            continue
        if len(calls) != 1:
            held += 1  # Ambiguous duplicates require an explicit canonical conversation.
            continue
        path = root / 'user-audio' / (asset['media_id'] + '.wav')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != h:
            raise ValueError('audio_hash_mismatch')
        key = f'audio/normalized/{h}.wav'
        sample = {'sample_id': asset['media_id'], 'conversation_id': calls[0],
                  'audio_key': key, 'audio_sha256': h, 'split': split,
                  'preprocessing_version': PREPROCESSING, 'speech_qc': check}
        validate_sample(sample)
        client.put_object(Bucket=bucket(), Key=key, Body=data)
        rows.append(sample)
    payload = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows).encode()
    ident = hashlib.sha256(payload).hexdigest()
    if rows:
        client.put_object(Bucket=bucket(), Key=f'incoming/{ident}.jsonl', Body=payload)
    return {'staged': len(rows), 'held': held, 'manifest_sha256': ident}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--speech-qc', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=['train', 'validation'])
    args = parser.parse_args()
    print(json.dumps(stage(args.root, args.speech_qc, args.split)))
