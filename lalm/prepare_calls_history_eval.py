"""Eval-only paired history views of a frozen original-test call selection."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from lhotse import CutSet
from lhotse.dataset import AudioSamples
from lalm_core.data_module import LALMDataset
from lalm_core.model import LALMProcessor
from prepare_calls_context import call_cuts, validate_manifest
from prepare_conversation import prepare_asr_cut


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'audio-root', 'output-dir', 'model', 'system-source'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--short-supplement', action='store_true', help='Only original <0.5s turns; stock Lhotse right-zero-padding to 0.5s')
    args = p.parse_args()
    rows = [json.loads(s) for s in args.manifest.read_text().splitlines()]
    validate_manifest(rows)
    assert len(rows) == 10 and all(r['split'] == 'test' for r in rows)
    assert rows == sorted(rows, key=lambda r: (-r['duration_seconds'], r['call_id']))
    system = json.loads(args.system_source.read_text().splitlines()[0])['messages'][0]
    assert system['role'] == 'system'
    out = args.output_dir
    out.mkdir(exist_ok=False)
    (out / 'audio').mkdir()
    processor = LALMProcessor.from_pretrained(args.model)
    views = {'with-history': [], 'no-history': []}
    excluded, ledger = [], []
    for row in rows:
        selected = list(call_cuts(row, args.audio_root, system['content'], evaluation_only=True,
                                  include_short=args.short_supplement))
        if args.short_supplement:
            selected = [c for c in selected if c.duration < 0.5]
        kept = {c.source_turn_id for c in selected}
        excluded.extend(dict(call_id=row['call_id'], **t, reason='scored_in_primary420' if args.short_supplement else 'current_duration_outside_0.5_to_30', retained_in_causal_history=True)
                        for t in row['turns'] if t['id'] not in kept)
        for original in selected:
            # Stock Lhotse channel selection/resampling and audio export only.
            audio = original.pad(duration=0.5, direction='right', preserve_id=True) if args.short_supplement else original
            cut = audio.save_audio(out / 'audio' / (original.id + '.wav'), encoding='FLOAT')
            cut.custom = copy.deepcopy(original.custom)
            for s in cut.supervisions:
                s.channel = 0
                s.duration = cut.duration
            assert np.array_equal(audio.load_audio(), cut.load_audio()), original.id
            if args.short_supplement:
                n = original.num_samples
                assert np.array_equal(original.load_audio(), cut.load_audio()[:, :n]), original.id
                assert np.count_nonzero(cut.load_audio()[:, n:]) == 0
            cut.custom.update(original_channel=original.channel, original_start=original.start,
                original_duration=original.duration, right_zero_padding_seconds=cut.duration-original.duration,
                original_audio_sha256=row['provenance']['audio_sha256'],
                reference_kind='physical_channel_RNNT_pseudo_not_human_gold',
                history_kind='previous_completed_source_RNNT_turns_not_oracle')
            record = dict(id=cut.id, call_id=row['call_id'], turn_id=cut.source_turn_id,
                audio=str(out/'audio'/(cut.id+'.wav')), audio_sha256=sha(out/'audio'/(cut.id+'.wav')),
                duration=cut.duration, original_duration=original.duration, original_channel=original.channel, original_start=original.start,
                history_turns=len(cut.history), reference=cut.supervisions[0].text)
            for mode in views:
                c = copy.deepcopy(cut)
                if mode == 'no-history':
                    c.custom['history'] = []
                c.custom['evaluation_condition'] = mode
                c = prepare_asr_cut(c, processor.tokenizer)
                assert c.conversation[-2]['content'][0]['type'] == 'audio'
                assert c.conversation[:-2] == ([system] + c.history)
                views[mode].append(c)
            ledger.append(record)
    expected = 113 if args.short_supplement else 420
    assert len(ledger) == expected and len(excluded) == 533-expected
    assert len({r['id'] for r in ledger}) == expected
    max_actual = 0
    for mode, cuts in views.items():
        # Native processor over every row: no text truncation; labels/EOS exact.
        for pos in range(0, len(cuts), 8):
            batch = LALMDataset(AudioSamples(), processor, return_cuts=True)[CutSet.from_cuts(copy.deepcopy(cuts[pos:pos+8]))]
            for c, labels, mask in zip(batch['cuts'], batch['labels'], batch['attention_mask']):
                assert processor.tokenizer.decode(labels[labels != -100], skip_special_tokens=False).strip() == c.supervisions[0].text+'<|im_end|>'
                n = int(mask.sum())
                max_actual = max(max_actual, n)
                assert n + 256 <= 32768, (c.id, n)
        manifest = out / (mode + '.jsonl.gz')
        CutSet.from_cuts(cuts).to_file(manifest)
        (out/(mode+'.yaml')).write_text(yaml.safe_dump([dict(name='ten-calls-'+('short-' if args.short_supplement else '')+mode, manifest=str(manifest))]))
    with (out/'rnnt.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['path', 'duration', 'transcription'])
        writer.writerows((r['audio'], r['duration'], r['reference']) for r in ledger)
    (out/'turn-ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2))
    (out/'excluded-turns.json').write_text(json.dumps(excluded, ensure_ascii=False, indent=2))
    summary = dict(status='PASS', evaluation_only=True, training_eligible=False,
        source_rows=10, original_turns=sum(len(r['turns']) for r in rows), scored_turns=len(ledger),
        excluded_current_turns=len(excluded), original_recording_seconds=sum(r['duration_seconds'] for r in rows),
        scored_channel_seconds=sum(r['original_duration'] for r in ledger),
        rendered_seconds_including_padding=sum(r['duration'] for r in ledger),
        short_supplement=args.short_supplement,
        reference_kind='RNNT pseudo; report agreement, not human WER',
        history_kind='source RNNT completed turns, including short turns; not oracle, not newly decoded history',
        max_actual_processor_sequence_tokens=max_actual, generation_budget=256,
        native_all_rows_labels_eos_pass=True, same_waveforms_exact=True, no_history_truncation=True,
        source_sha256=sha(args.manifest), system_source_sha256=sha(args.system_source),
        source_adapter_sha256=sha(Path(__file__).with_name('prepare_calls_context.py')),
        format_adapter_sha256=sha(__file__),
        manifest_sha256={mode:sha(out/(mode+'.jsonl.gz')) for mode in views})
    (out/'readiness.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
