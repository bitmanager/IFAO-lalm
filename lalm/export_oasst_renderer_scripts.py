"""Exact approved OASST chains -> existing MultiTalk renderer schema; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID

REVISION = '179dd21fc55192153d94adb0e0ce8f69e222bf75'
CASTING = [('female', 'female'), ('female', 'male'), ('male', 'female'), ('male', 'male')]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def convert(selected, official, blocked_roots, tokenizer, provenance):
    records, roots = [], set()
    for row in selected:
        root = row['root_id']
        if str(UUID(root)) != root or root in roots or root in blocked_roots:
            raise ValueError(f'Duplicate, unsafe or excluded root: {root}')
        roots.add(root)
        messages = row['messages']
        if row['source_split'] != 'train' or len(messages) < 4 or len(messages) % 2:
            raise ValueError(f'{root}: expected root-to-completed-assistant train prefix')
        if row['conversation_ids'] != [m['message_id'] for m in messages] or messages[0]['message_id'] != root:
            raise ValueError(f'{root}: original message IDs differ')
        for i, message in enumerate(messages):
            source = official[message['message_id']]
            expected_role = 'prompter' if i % 2 == 0 else 'assistant'
            expected_parent = messages[i-1]['message_id'] if i else None
            if any(source[k] != message[k] for k in ('message_id', 'parent_id', 'role', 'text', 'rank')):
                raise ValueError(f'{root}: source fields changed')
            labels = dict(zip(source['labels']['name'], source['labels']['value']))
            if labels.get('quality') != message['quality']:
                raise ValueError(f'{root}: source quality changed')
            if (source['source_split'] != 'train' or source['message_tree_id'] != root
                    or source['role'] != expected_role or source['parent_id'] != expected_parent
                    or source['lang'] != 'ru' or source['deleted'] is not False
                    or source['synthetic'] is not False or source['review_result'] is not True):
                raise ValueError(f'{root}: official split, ancestry or review eligibility failed')
            text = message['text']
            if not text.strip() or '<|' in text:
                raise ValueError(f'{root}: empty or control-token source text')
            if expected_role == 'prompter' and (len(text) > 240 or len(text.split()) > 40):
                raise ValueError(f'{root}: user exceeds approved pilot size; no truncation')
            if expected_role == 'assistant' and len(tokenizer.encode(text + '<|im_end|>', add_special_tokens=False)) > 256:
                raise ValueError(f'{root}: answer exceeds token budget; no truncation')
        user_gender, assistant_gender = CASTING[len(records) % len(CASTING)]
        records.append(dict(request_id='oasst2-' + root, source_group_id='oasst2:' + root,
            split='train', config={'language': 'ru'},
            participants=[dict(name='user', role='user', gender=user_gender),
                          dict(name='assistant', role='assistant', gender=assistant_gender)],
            dialogue=[dict(speaker='user' if m['role'] == 'prompter' else 'assistant', text=m['text']) for m in messages],
            source_provenance={**provenance, 'root_id': root, 'messages': messages,
                'source_split': row['source_split'], 'selection_key': row['selection_key'],
                'text_quality': 'original_reviewed_human_source_not_factual_gold',
                'casting': 'Deterministic synthetic voice-pool casting; not source demographics or speaker identity',
                'new_text_generation': False, 'training_ready': False}))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'selection', 'inventory', 'exclusions', 'tokenizer', 'renderer-adapter', 'upstream-tts', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-roots', type=int, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    metadata_path = args.snapshot / 'hub-metadata.json'
    metadata = json.loads(metadata_path.read_text())
    assert metadata['id'] == 'OpenAssistant/oasst2' and metadata['sha'] == REVISION
    official, source_hashes = {}, {}
    for file in metadata['siblings']:
        if not file['rfilename'].startswith('data/') or not file['rfilename'].endswith('.parquet'):
            continue
        path = args.snapshot / file['rfilename']
        assert path.stat().st_size == file['lfs']['size'] and sha256(path) == file['lfs']['sha256'], path
        source_hashes[str(path)] = file['lfs']['sha256']
        split = path.name.split('-')[0]
        assert split in ('train', 'validation')
        for row in pq.read_table(path).to_pylist():
            assert row['message_id'] not in official
            official[row['message_id']] = {**row, 'source_split': split}
    assert {r['source_split'] for r in official.values()} == {'train', 'validation'}
    blocked = {r['message_tree_id'] for r in official.values() if r['source_split'] == 'validation'}
    blocked.update(r['root'] for r in json.loads(args.inventory.read_text())['heldout_collisions'])
    blocked.update(json.loads(args.exclusions.read_text())['excluded_roots'])
    selected = [json.loads(line) for line in args.selection.read_text().splitlines() if line.strip()]
    assert len(selected) == args.expected_roots
    hashes = {str(p): sha256(p) for p in (metadata_path, args.selection, args.inventory, args.exclusions,
              args.renderer_adapter, args.upstream_tts, Path(__file__))}
    provenance = dict(dataset='OpenAssistant/oasst2', revision=REVISION, selection_sha256=hashes[str(args.selection)],
                      input_and_code_sha256=hashes, official_parquet_sha256=source_hashes)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    records = convert(selected, official, blocked, tokenizer, provenance)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / 'scripts.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    report = dict(status='format_ready_for_review_no_audio', roots=len(records),
        user_turns=sum(len(r['dialogue'])//2 for r in records), root_ids=[r['source_provenance']['root_id'] for r in records],
        scripts=str(manifest), scripts_sha256=sha256(manifest), provenance=provenance,
        casting_counts={f'{a}/{b}':sum(r['participants'][0]['gender']==a and r['participants'][1]['gender']==b for r in records) for a,b in CASTING},
        heldout_or_excluded_root_overlap=0, original_text_bytes_preserved=True, original_ids_roles_parentage_verified=True,
        no_text_generation=True, no_TTS=True, no_training=True,
        limitations=['Original human source is not fact-checked gold.',
                    'Selected root-to-completed-assistant prefixes need not be terminal leaves; future source replies are not appended.',
                    'Character/word limits do not certify rendered duration; native audio and ASR QC still required.',
                    'Renderer retains existing seven-voice allowlist; casting does not verify source speaker identity.'])
    (args.output / 'readiness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
