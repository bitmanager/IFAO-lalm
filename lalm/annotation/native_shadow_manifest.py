"""Select a bounded, review-only native audio shadow batch from completed CLI work."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from .contract import source_text
from .queue import protected
from .store import bucket, connect, s3


def build(seed_rows, recent_rows, conn):
    # The seed has the six listening controls; exclude every audio in its
    # previous 30-sample manifest so the new batch measures fresh material.
    controls = seed_rows[:6]
    if len(controls) != 6 or any(not r.get('control') for r in controls):
        raise ValueError('six_controls_required')
    used_hashes = {r['sample']['audio_sha256'] for r in seed_rows}
    used_conversations = {r['sample']['conversation_id'] for r in controls}
    selected = {'ru': [], 'other': []}
    targets = {'ru': 80, 'other': 14}
    for row in recent_rows:
        sample, result = row['sample'], row['result']
        sha, conversation = sample['audio_sha256'], sample['conversation_id']
        group = 'ru' if result.get('language') == 'ru' else 'other'
        if (len(selected[group]) >= targets[group] or sha in used_hashes
                or conversation in used_conversations or sample.get('split') != 'train'
                or source_text(sample) is None
                or not isinstance(result.get('audio_seconds'), (int, float))
                or not 1 <= result['audio_seconds'] <= 20
                or protected(conn, sample)):
            continue
        selected[group].append({'sample': sample,
            'control': {'audit': 'fresh_unverified', 'language_class': 'unknown'},
            'cli_previous_passes': {'verbatim': {
                'verbatim_text': result.get('verbatim_text'),
                'language': result.get('language')}}})
        used_hashes.add(sha)
        used_conversations.add(conversation)
        if all(len(selected[g]) == targets[g] for g in targets):
            break
    if any(len(selected[g]) != targets[g] for g in targets):
        raise ValueError('insufficient_distinct_unprotected_recent_audio')
    return controls + selected['ru'] + selected['other']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--s3-prefix', required=True)
    args = p.parse_args()
    seed = json.loads(args.seed_manifest.read_text())
    with connect() as conn:
        rows = conn.execute("SELECT sample,result FROM jobs WHERE result IS NOT NULL "
            "AND sample->>'prompt_variant'='verbatim' AND sample->>'split'='train' "
            "AND updated_at>=now()-interval '24 hours' "
            "ORDER BY updated_at DESC LIMIT 10000").fetchall()
        manifest = build(seed, rows, conn)
    data = json.dumps(manifest, ensure_ascii=False).encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(args.output, flags, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    s3().put_object(Bucket=bucket(), Key=args.s3_prefix.rstrip('/')+'/manifest.json',
                    Body=data, ContentType='application/json')
    print(json.dumps({'samples': len(manifest), 'groups': dict(Counter(
        'control' if i < 6 else 'ru' if i < 86 else 'other'
        for i in range(len(manifest)))), 'manifest_bytes': len(data)}))


if __name__ == '__main__':
    main()
