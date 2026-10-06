"""Bounded answer diagnostics; select existing v3/self-teacher views, no new targets."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml
from lhotse import CutSet
from lhotse.dataset import AudioSamples
from lalm_core.data_module import LALMDataset
from lalm_core.model import LALMProcessor
from scripts.prepare_projector_semantic_v3 import MODEL, ROOT, finite_sampler_audit, sha256
from scripts.prepare_phone2_context_stage import identities
from prepare_conversation import teacher_messages

BASE = ROOT / 'runs/asr-projector-semantic-v3'
OUTPUT = ROOT / 'runs/asr-projector-answer-only-diagnostic-v1'
SELF_TEACHER_SOURCE = ROOT / 'original-script-answers-v1/source.jsonl.gz'
SELF_TEACHER_SHA = '67a54cd63909509110dea6787d1d02e276995a4bcc484c13a657c7450a9e123a'
# Parent reviewed all fixed 20 + 6 examples; exclude entire source IDs, no rewriting.
QUARANTINE = {
    'reviewed-f352a0f9fa8ab7ba7703e807-turn-0002',
    'reviewed-3d590a58734196aaf61e7a18-turn-0002',
    'reviewed-da59e91841760bc2dc43df3f-turn-0002',
    'd0e06a2fc2c869f160f9-turn-0004',
    '9299cf099f6048adb08d-turn-0000',
    '3f89512ed978a78ec9bd-turn-0000',
}


def joint_self_teacher(ready, inputs):
    """Same-ID replacement in the frozen v3 order; reuse the prepared answer ledger."""
    source = ROOT/'runs/asr-projector-self-teacher-diagnostic-v1'
    saved = json.loads((source/'readiness.json').read_text())
    assert saved['selector'] == 'self-teacher' and saved['cuts'] == 2520
    assert saved['manifest_sha256'] == '61eec1f2291a424439239c578e98459c408920678ca66f8639b153a04278e334'
    assert saved['source_provenance']['excluded_source_ids'] == sorted(QUARANTINE)
    for field in ('model_dir', 'model_file_sha256', 'heldout_manifest_sha256', 'validation_config_sha256'):
        assert saved[field] == ready[field], field
    checks = {**ready['input_sha256'], **saved['input_sha256'],
        str(source/'readiness.json'):sha256(source/'readiness.json'), saved['manifest']:saved['manifest_sha256']}
    for path, digest in checks.items():
        assert sha256(path) == digest, path
    inputs.update(checks)
    answers = {c.id:c for c in CutSet.from_file(saved['manifest'])}
    assert len(answers) == 2520
    excluded_answer = {f'{sid}_repeat{i}' for sid in QUARANTINE for i in range(3)}
    excluded_asr = {sid+'-asr' for sid in QUARANTINE}
    rows, replaced, removed, asr_hashes = [], set(), set(), {}
    for c in CutSet.from_file(BASE/'train.jsonl.gz'):
        if c.id in excluded_answer | excluded_asr:
            removed.add(c.id)
            continue
        if c.custom.get('task', 'answer') == 'answer':
            new = answers[c.id]
            assert new.recording.to_dict() == c.recording.to_dict()
            assert (new.start, new.duration, new.channel, new.supervisions[0].text) == (c.start, c.duration, c.channel, c.supervisions[0].text)
            assert all(new.custom[k] == c.custom[k] for k in ('history', 'system', 'split', 'source_group_id'))
            assert new.conversation[:-1] == c.conversation[:-1]
            replaced.add(c.id)
            c = new
        else:
            asr_hashes[c.id] = hashlib.sha256(json.dumps(c.to_dict(), sort_keys=True).encode()).hexdigest()
        rows.append(c)
    assert removed == excluded_answer | excluded_asr and replaced == set(answers)
    assert len(asr_hashes) == 19068
    provenance = dict(saved['source_provenance'], prepared_manifest=saved['manifest'],
        prepared_manifest_sha256=saved['manifest_sha256'],
        retained_asr_counts=dict(paired=6240, clean=840, phone_rnnt=5988, sova=6000),
        omitted_answer_views=sorted(excluded_answer), omitted_clean_asr_ids=sorted(excluded_asr),
        note='Replace answers only; 72 retained paired-ASR views reference three excluded answer sources. Their ASR labels remain unchanged; exclusions are not global audio quarantine.')
    return CutSet.from_cuts(rows), provenance, asr_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--self-teacher', action='store_true', help='Use saved same-Qwen clean targets, excluding six reviewed failures')
    modes.add_argument('--joint-self-teacher', action='store_true', help='Replace v3 answers with prepared self-teacher views; retain ASR except six matching clean views')
    args = parser.parse_args()
    output = ROOT/'runs/asr-projector-self-teacher-diagnostic-v1' if args.self_teacher else OUTPUT
    if args.joint_self_teacher:
        output = ROOT/'runs/asr-projector-joint-selfteacher-v1'
    updates = 500 if args.joint_self_teacher else 250
    assert not output.exists(), output
    ready = json.loads((BASE/'readiness.json').read_text())
    assert sha256(BASE/'train.jsonl.gz') == ready['manifest_sha256'] == '63ca8a9a7543bf4e72b78c916270945807527b2210afa40d3ee824fbc25ba728'
    assert ready['model_dir'] == str(MODEL) and ready['trainable_parameters'] == 10490880
    audit = json.loads((BASE/'frozen-state-audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['export_frozen_tensor_bytes_identical']
    cuts = CutSet.from_file(BASE/'train.jsonl.gz').filter(lambda c: c.custom.get('task', 'answer') == 'answer').to_eager()
    assert len(cuts) == len({c.id for c in cuts}) == 2538
    inputs = {str(BASE/'train.jsonl.gz'):ready['manifest_sha256'], str(BASE/'readiness.json'):sha256(BASE/'readiness.json')}
    provenance = ready['source_preflight']['original_answer']
    expected, unique_turns = 2538, 846
    quality = 'original_script_synthetic_not_gold'
    if args.self_teacher:
        assert sha256(SELF_TEACHER_SOURCE) == SELF_TEACHER_SHA
        source = CutSet.from_file(SELF_TEACHER_SOURCE).to_eager()
        assert len(source) == 846 and QUARANTINE <= set(source.ids)
        inputs[str(SELF_TEACHER_SOURCE)] = SELF_TEACHER_SHA
        quality = 'same_frozen_qwen_self_generated_not_gold'
        for c in source:
            directory = 'reviewed-v2-responses' if c.id.startswith('reviewed-') else 'teacher-scale512-responses'
            path = Path('/ifao-context-data')/directory/(c.id+'.json')
            response = json.loads(path.read_text())
            assert response['idx'] == c.id and response['messages'] == teacher_messages(c)
            assert response['response'] == c.supervisions[0].custom['answer']
            assert c.conversation[:-2] == response['messages'][:-1]
            assert len(c.conversation[-2]['content']) == 1 and c.conversation[-2]['content'][0]['type'] == 'audio'
            inputs[str(path)] = sha256(path)
            # The saved source retained the old Qwen target but carried metadata for
            # the later external-script replacement. Correct only this new copy.
            c.custom['original_script_join_provenance'] = c.custom['answer_provenance']
            c.custom['answer_provenance'] = dict(quality=quality, response_path=str(path),
                response_sha256=inputs[str(path)], model='Qwen/Qwen3-4B-Instruct-2507',
                model_revision='cdbee75f17c01a7cc42f958dc650907174af0554',
                messages_exact=True, generation_performed=False)
            c.custom['task'] = 'answer'
        selected = source.filter(lambda c: c.id not in QUARANTINE).to_eager()
        assert len(selected) == 840
        provenance = dict(quality=quality, source_manifest=str(SELF_TEACHER_SOURCE),
            source_sha256=SELF_TEACHER_SHA, source_turns=846, selected_turns=840,
            excluded_source_ids=sorted(QUARANTINE), repetitions=3,
            unique_audio_hours=sum(c.duration for c in selected)/3600,
            note='Existing sampled same-Qwen targets, reused audio; not factual gold. Six reviewed failures excluded.')
        cuts = selected.repeat(3, preserve_id=False).to_eager()
        expected, unique_turns = 2520, 840
    asr_hashes = {}
    if args.joint_self_teacher:
        cuts, provenance, asr_hashes = joint_self_teacher(ready, inputs)
        expected, unique_turns = 21588, 840
        quality = 'same_frozen_qwen_self_generated_not_gold'
    assert len(cuts) == len(set(cuts.ids)) == expected
    tasks = Counter(c.custom.get('task', 'answer') for c in cuts)
    assert tasks == ({'asr':19068, 'answer':2520} if args.joint_self_teacher else {'answer':expected})
    for path, digest in ready['heldout_manifest_sha256'].items():
        assert sha256(path) == digest, path
    heldout = identities([c for path in ready['heldout_manifest_sha256'] for c in CutSet.from_file(path)])
    overlap = {key:len(value & heldout[key]) for key,value in identities(cuts).items()}
    assert not any(overlap.values()), overlap
    for c in cuts:
        if c.custom.get('task', 'answer') == 'answer':
            assert c.answer_provenance['quality'] == quality
        else:
            assert c.supervisions[0].custom['answer'] == c.supervisions[0].text
        split = c.custom.get('split', c.custom.get('source', {}).get('split'))
        assert split == 'train' and not c.custom.get('source_call_id')
        assert .5 <= c.duration <= 30
        target = c.supervisions[0].custom['answer']
        assert c.conversation[-1] == {'role':'assistant', 'content':target}
        assert c.rendered_conversation.endswith(target+'<|im_end|>\n')
        # Only experiment bookkeeping changes; prompts, labels and source fields stay exact.
        if not args.joint_self_teacher:
            c.custom['training_recipe'] = output.name
    sampler = finite_sampler_audit(copy.deepcopy(cuts) if args.joint_self_teacher else cuts, required_updates=updates)
    processor = LALMProcessor.from_pretrained(MODEL)
    lengths = [(len(processor.tokenizer.encode(c.supervisions[0].custom['answer']+'<|im_end|>', add_special_tokens=False)),
                len(processor.tokenizer.encode(c.rendered_conversation.strip(), add_special_tokens=False))+round(c.duration*12.5)) for c in cuts]
    assert max(n for n,_ in lengths) <= 256 and max(n for _,n in lengths) <= 2000, 'Report token outliers; never truncate/drop silently'
    ordered = sorted(cuts, key=lambda c:(c.duration,c.id))
    sample = CutSet.from_cuts(ordered[round(i*(len(ordered)-1)/15)] for i in range(16))
    if args.joint_self_teacher:
        selected = []
        for task, count in (('asr', 12), ('answer', 4)):
            rows = [c for c in ordered if c.custom.get('task', 'answer') == task]
            selected.extend(rows[round(i*(len(rows)-1)/(count-1))] for i in range(count))
        sample = copy.deepcopy(CutSet.from_cuts(selected))
    batch = LALMDataset(AudioSamples(), processor, return_cuts=True)[sample]
    assert batch['batch_size'] == 16 and batch['asr_mask'].sum().item() == (12 if args.joint_self_teacher else 0)
    for c,labels,asr_mask in zip(batch['cuts'],batch['labels'],batch['asr_mask']):
        assert asr_mask.item() == (c.custom.get('task', 'answer') == 'asr'), c.id
        assert processor.tokenizer.decode(labels[labels != -100],skip_special_tokens=False).strip() == c.supervisions[0].custom['answer']+'<|im_end|>',c.id
    for c in cuts:
        if c.id in asr_hashes:
            assert hashlib.sha256(json.dumps(c.to_dict(), sort_keys=True).encode()).hexdigest() == asr_hashes[c.id], c.id
    output.mkdir()
    manifest = output/'train.jsonl.gz'; cuts.to_file(manifest)
    hours = sum(c.duration for c in cuts)/3600
    (output/'train.yaml').write_text(yaml.safe_dump([dict(name='joint-self-teacher' if args.joint_self_teacher else 'answer-only-diagnostic',manifest=str(manifest),hours=hours,weights=1)]))
    (output/'validation.yaml').write_bytes((BASE/'validation.yaml').read_bytes())
    result = dict(status='ready_for_review_not_launched',selector='joint-self-teacher' if args.joint_self_teacher else ('self-teacher' if args.self_teacher else 'original-script'),cuts=expected,tasks=dict(tasks),unique_original_turns=unique_turns,
        task_exposure_hours=hours,manifest=str(manifest),manifest_sha256=sha256(manifest),
        train_config_sha256=sha256(output/'train.yaml'),validation_config_sha256=sha256(output/'validation.yaml'),
        input_sha256=inputs,source_provenance=provenance,model_dir=str(MODEL),
        model_file_sha256=ready['model_file_sha256'],heldout_manifest_sha256=ready['heldout_manifest_sha256'],heldout_overlap=overlap,
        trainable_parameters=10490880,trainable_tensors=ready['trainable_tensors'],frozen_modules=ready['frozen_modules'],
        finite_sampler=sampler,native_cpu_batch=dict(cuts=16,answer=4 if args.joint_self_teacher else 16,asr=12 if args.joint_self_teacher else 0,exact_final_target_and_EOS=True,ids=[c.id for c in batch['cuts']]),
        target_token_share_answer=sum(n for c,(n,_) in zip(cuts,lengths) if c.custom.get('task','answer') == 'answer')/sum(n for n,_ in lengths),target_tokens_including_eos=sum(n for n,_ in lengths),
        max_target_tokens_including_eos=max(n for n,_ in lengths),max_estimated_total_tokens=max(n for _,n in lengths),
        updates=updates,optimizer='fresh AdamW; native cosine equal base/min LR 3e-6',training_launched=False,
        notes=['Diagnostic, not an established cure or causal ablation.',
               'Existing answer sources repeated three times; no new audio hours. See source_provenance for target origin.',
               'Self-teacher comparison changes six source exclusions and target/batch lengths; not a strict causal ablation.',
               'Retained v3 ASR rows remain exact; no new text, augmentation or architecture changes. Fixed validation unchanged.' if args.joint_self_teacher else 'No ASR views, new text, augmentation or architecture changes. Fixed validation unchanged.'])
    if args.joint_self_teacher:
        result['retained_asr_exact_rows'] = len(asr_hashes)
        result['retained_asr_ids_sha256'] = hashlib.sha256('\n'.join(sorted(asr_hashes)).encode()).hexdigest()
    (output/'readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
