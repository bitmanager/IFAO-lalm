"""I/O adapter for the existing native greedy teacher runner; no training or service."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

MODEL = Path('/llm')
TOKENIZER = Path('/runs/dev-storage/ifao-data/runs/asr-short-context-v1/export/iter-26000')
NATIVE_RUNNER = Path('/runs/dev-storage/ifao-data/context-history-text-greedy-v1/runner.py')
QUARANTINE = Path('/runs/dev-storage/ifao-data/oasst-expanded-staging-v1/training-quarantine.json')
SEED = 20261006


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def cache_matches(row, item, prompt_sha256):
    if row['idx'] != item['idx'] or row['messages'] != item['messages'] or row['prompt_sha256'] != prompt_sha256:
        raise ValueError(f"Source/prompt cache conflict: {item['idx']}")
    if not isinstance(row['response'], str) or row['finish_reason'] not in ('eos', 'length', 'other'):
        raise ValueError(f"Invalid response cache: {item['idx']}")
    if row['finish_reason'] == 'eos' and (not row['response'].strip() or '<|' in row['response']):
        raise ValueError(f"Invalid completed response cache: {item['idx']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--limit', type=int, default=30, help='Balanced smoke30 or the exact full source size')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    from lhotse import CutSet
    from transformers import AutoTokenizer
    from prepare_conversation import teacher_messages
    assert sha(args.source_manifest) == args.source_sha256, 'Source manifest changed'
    cuts = list(CutSet.from_file(args.source_manifest))
    source = {c.id:c for c in cuts}
    items = [json.loads(line) for line in args.input.read_text().splitlines()]
    assert len(items) == len(source) == len(cuts) == len({x['idx'] for x in items}) > 0
    assert {x['idx'] for x in items} == set(source)
    assert args.limit in (30, len(source)), 'Only smoke30 or the complete source is supported'
    quarantined = set(json.loads(QUARANTINE.read_text())['case_ids'])
    quarantined.update(c.id for c in source.values() if re.search('[A-Za-z]', c.supervisions[0].text)
                       and not re.search('[А-Яа-яЁё]', c.supervisions[0].text))
    assert not quarantined.intersection(source), 'Apply source quarantine before teacher export'
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    prepared = {}
    for item in items:
        idx = item['idx']; cut = source[idx]
        assert idx and all(ch.isalnum() or ch in '-_.' for ch in idx), idx
        assert cut.custom.get('task', 'answer') == 'answer' and cut.custom['split'] == 'train'
        assert item['messages'] == teacher_messages(cut), f'Source/prompt mismatch: {idx}'
        assert not any('<|' in turn['content'] for turn in item['messages']), idx
        prompt = tokenizer.apply_chat_template(item['messages'], tokenize=False, add_generation_prompt=True)
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        assert tokenizer(prompt).input_ids == encoded, f'Actual native tokenizer input differs: {idx}'
        assert len(encoded) + 256 <= 8192, idx
        prepared[idx] = (item, prompt, hashlib.sha256(prompt.encode()).hexdigest())
    # Fixed before inference: 15 first-turn and 15 contextual turns; the full source is a separate opt-in.
    selected = items
    if args.limit == 30:
        order = lambda item: hashlib.sha256(f"{SEED}:{item['idx']}".encode()).hexdigest()
        selected = sum((sorted([x for x in items if bool(source[x['idx']].custom['history']) == history], key=order)[:15]
                        for history in (False, True)), [])
    assert len(selected) == args.limit
    files = [p for p in MODEL.iterdir() if p.is_file()]
    files += [TOKENIZER/name for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja')]
    provenance = dict(source_manifest=str(args.source_manifest), source_sha256=args.source_sha256,
        input_sha256=sha(args.input), model=str(MODEL), tokenizer=str(TOKENIZER),
        files_sha256={str(p):sha(p) for p in files}, adapter_sha256=sha(__file__),
        native_runner_sha256=sha(NATIVE_RUNNER), vocabulary_resize=151670,
        generation=dict(do_sample=False, max_new_tokens=256, dtype='bfloat16'),
        quarantine_sha256=sha(QUARANTINE), quarantined_ids=sorted(quarantined),
        seed_for_sample_selection=SEED, training_admitted=False,
        note='Same native runner generation; all data staging. Source quarantine applied before teacher export. No source history/transcript edits.')
    ledger = args.output/'provenance.json'
    if ledger.exists():
        assert json.loads(ledger.read_text()) == provenance, 'Source/model/adapter provenance cache conflict'
    else:
        assert not list(args.output.glob('*.json')), 'Existing response cache has no provenance'
        if not args.check_only:
            args.output.mkdir(parents=True, exist_ok=True)
            ledger.write_text(json.dumps(provenance, indent=2)+'\n')
    for path in args.output.glob('*.json'):
        if path == ledger:
            continue
        assert path.stem in prepared, f'Unknown source ID in cache: {path}'
        item, _, prompt_sha256 = prepared[path.stem]
        cache_matches(json.loads(path.read_text()), item, prompt_sha256)
    pending = [item for item in selected if not (args.output/(item['idx']+'.json')).exists()]
    if args.check_only:
        print(json.dumps(dict(status='CPU_only', source_cuts=len(source), selected=len(selected), pending=len(pending),
            selected_ids=[x['idx'] for x in selected], provenance=provenance, training_admitted=False), indent=2))
        return
    import torch
    from transformers import AutoModelForCausalLM
    assert torch.cuda.device_count() == 1, 'Expose exactly one approved GPU'
    if not pending:
        print('All selected responses already cached and source/prompt verified.')
        return
    # Unchanged native runner model construction, stock resize, prompt and generate call.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to('cuda').eval()
    model.resize_token_embeddings(151670)
    started = time.monotonic()
    with torch.inference_mode():
        for item in pending:
            idx = item['idx']; _, prompt, prompt_sha256 = prepared[idx]
            inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
            t0 = time.monotonic()
            result = model.generate(**inputs, do_sample=False, max_new_tokens=256,
                                    return_dict_in_generate=True)
            tokens = result.sequences[0, inputs.input_ids.shape[-1]:]
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            ended_eos = int(tokens[-1]) in eos if len(tokens) else False
            finish = 'eos' if ended_eos else 'length' if len(tokens) == 256 else 'other'
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            # Same fail-closed marker understood by stock prepare_conversation --responses.
            response = text if ended_eos else text+'<|truncated|>'
            row = dict(idx=idx, messages=item['messages'], response=response, finish_reason=finish,
                generated_token_ids=tokens.tolist(), input_tokens=inputs.input_ids.shape[-1],
                generated_tokens=len(tokens), prompt_sha256=prompt_sha256, seconds=time.monotonic()-t0)
            with (args.output/(idx+'.json')).open('x') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps(dict(idx=idx, finish_reason=finish, tokens=len(tokens), seconds=row['seconds'])), flush=True)
    print(json.dumps(dict(generated=len(pending), selected=len(selected), seconds=time.monotonic()-started, training_admitted=False)), flush=True)


if __name__ == '__main__':
    main()
