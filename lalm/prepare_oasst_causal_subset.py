"""Exact-ID subset of the approved OASST causal-history audit; no new audio or targets."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = '3858c70a763f0003f7b36c32ae4f18dc06c1b4df09b3eaad15bf36958359078f'
AUDIT_SHA256 = '0d5ae2b48e9beaecf1e8b4dc7d55b0f31897bf1a45b7f73c06ec805dec62fae8'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    from lhotse import CutSet
    assert sha(args.source) == SOURCE_SHA256, 'Original source changed'
    assert sha(args.audit) == AUDIT_SHA256, 'Approved causal audit changed'
    source = list(CutSet.from_file(args.source))
    rows = [json.loads(line) for line in args.audit.read_text().splitlines()]
    audit = {r['cut_id']:r for r in rows}
    assert len(source) == len(rows) == len(audit) == len({c.id for c in source}) == 1020
    assert set(audit) == {c.id for c in source}
    selected, excluded = [], []
    for cut in source:
        row = audit[cut.id]
        assert row['causal_quality_pass'] == (not row['reasons'])
        assert cut.custom['split'] == 'train' and 'task' not in cut.custom
        assert cut.custom['source_group_id'] == 'oasst2:' + row['root_id']
        assert len(cut.custom['history']) == 2 * row['history_assistants']
        assert abs(cut.duration - row['duration']) < 1e-9
        reasons = row['reasons'] + (['existing_suffix_or_English_quarantine'] if row['existing_quarantine'] else [])
        if reasons:
            excluded.append({**row, 'exclusion_reasons': reasons})
            continue
        original = cut.to_dict()
        cut.custom = {**cut.custom, 'training_eligible': False, 'main_training_connected': False,
            'oasst_causal_selection': {**row, 'audit_sha256': AUDIT_SHA256,
                'rule': 'preceding assistant rank0 and quality>=0.5; current user quality is a stratum only'}}
        restored = cut.to_dict()
        restored['custom'] = original['custom']
        assert restored == original, 'Audio, transcript or supervision changed'
        assert cut.custom['history'] == original['custom']['history']
        selected.append(cut)
    assert len(selected) == 638 and len(excluded) == 382
    assert sum(bool(c.custom['history']) for c in selected) == 154
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / 'current-user.jsonl.gz'
    CutSet.from_cuts(selected).to_file(manifest)
    (args.output / 'excluded.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in excluded))
    loaded = list(CutSet.from_file(manifest))
    assert [c.to_dict() for c in loaded] == [c.to_dict() for c in selected]
    report = dict(status='CPU_FORMAT_READY_FOR_REVIEW_NOT_ADMITTED', source=str(args.source),
        source_sha256=SOURCE_SHA256, audit=str(args.audit), audit_sha256=AUDIT_SHA256,
        adapter_sha256=sha(__file__), manifest=str(manifest), manifest_sha256=sha(manifest),
        cuts=len(selected), excluded=len(excluded), first_turns=484, with_history=154,
        hours=sum(c.duration for c in selected)/3600,
        current_user_quality_strata=dict(collections.Counter(c.custom['oasst_causal_selection']['current_user_quality_stratum'] for c in selected)),
        exact_source_ids_and_messages_preserved=True, source_manifest_unchanged=sha(args.source)==SOURCE_SHA256,
        official_split='train', training_admitted=False, new_audio=False, new_targets=False)
    (args.output / 'readiness.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
