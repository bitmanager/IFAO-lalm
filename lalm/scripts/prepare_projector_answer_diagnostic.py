"""Answer-only bounded diagnostic; filter existing v3 views, no new targets."""
import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-teacher', action='store_true', help='Use saved same-Qwen clean targets, excluding six reviewed failures')
    args = parser.parse_args()
    output = ROOT/'runs/asr-projector-self-teacher-diagnostic-v1' if args.self_teacher else OUTPUT
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
    assert len(cuts) == len(set(cuts.ids)) == expected
    for path, digest in ready['heldout_manifest_sha256'].items():
        assert sha256(path) == digest, path
    heldout = identities([c for path in ready['heldout_manifest_sha256'] for c in CutSet.from_file(path)])
    overlap = {key:len(value & heldout[key]) for key,value in identities(cuts).items()}
    assert not any(overlap.values()), overlap
    for c in cuts:
        assert c.answer_provenance['quality'] == quality
        assert c.custom['split'] == 'train' and not c.custom.get('source_call_id')
        assert .5 <= c.duration <= 30
        target = c.supervisions[0].custom['answer']
        assert c.conversation[-1] == {'role':'assistant', 'content':target}
        assert c.rendered_conversation.endswith(target+'<|im_end|>\n')
        # Only experiment bookkeeping changes; prompts, labels and source fields stay exact.
        c.custom['training_recipe'] = output.name
    sampler = finite_sampler_audit(cuts, required_updates=250)
    processor = LALMProcessor.from_pretrained(MODEL)
    lengths = [(len(processor.tokenizer.encode(c.supervisions[0].custom['answer']+'<|im_end|>', add_special_tokens=False)),
                len(processor.tokenizer.encode(c.rendered_conversation.strip(), add_special_tokens=False))+round(c.duration*12.5)) for c in cuts]
    assert max(n for n,_ in lengths) <= 256 and max(n for _,n in lengths) <= 2000, 'Report token outliers; never truncate/drop silently'
    ordered = sorted(cuts, key=lambda c:(c.duration,c.id))
    sample = CutSet.from_cuts(ordered[round(i*(len(ordered)-1)/15)] for i in range(16))
    batch = LALMDataset(AudioSamples(), processor, return_cuts=True)[sample]
    assert batch['batch_size'] == 16 and not batch['asr_mask'].any()
    for c,labels in zip(batch['cuts'],batch['labels']):
        assert processor.tokenizer.decode(labels[labels != -100],skip_special_tokens=False).strip() == c.supervisions[0].custom['answer']+'<|im_end|>',c.id
    output.mkdir()
    manifest = output/'train.jsonl.gz'; cuts.to_file(manifest)
    hours = sum(c.duration for c in cuts)/3600
    (output/'train.yaml').write_text(yaml.safe_dump([dict(name='answer-only-diagnostic',manifest=str(manifest),hours=hours,weights=1)]))
    (output/'validation.yaml').write_bytes((BASE/'validation.yaml').read_bytes())
    result = dict(status='ready_for_review_not_launched',selector='self-teacher' if args.self_teacher else 'original-script',cuts=expected,tasks={'answer':expected},unique_original_turns=unique_turns,
        task_exposure_hours=hours,manifest=str(manifest),manifest_sha256=sha256(manifest),
        train_config_sha256=sha256(output/'train.yaml'),validation_config_sha256=sha256(output/'validation.yaml'),
        input_sha256=inputs,source_provenance=provenance,model_dir=str(MODEL),
        model_file_sha256=ready['model_file_sha256'],heldout_manifest_sha256=ready['heldout_manifest_sha256'],heldout_overlap=overlap,
        trainable_parameters=10490880,trainable_tensors=ready['trainable_tensors'],frozen_modules=ready['frozen_modules'],
        finite_sampler=sampler,native_cpu_batch=dict(cuts=16,answer=16,asr=0,exact_final_target_and_EOS=True),
        target_token_share_answer=1.0,target_tokens_including_eos=sum(n for n,_ in lengths),
        max_target_tokens_including_eos=max(n for n,_ in lengths),max_estimated_total_tokens=max(n for _,n in lengths),
        updates=250,optimizer='fresh AdamW; native cosine equal base/min LR 3e-6',training_launched=False,
        notes=['Diagnostic, not an established cure or causal ablation.',
               'Existing answer sources repeated three times; no new audio hours. See source_provenance for target origin.',
               'Self-teacher comparison changes six source exclusions and target/batch lengths; not a strict causal ablation.',
               'No ASR views, new text, augmentation or architecture changes. Fixed validation unchanged.'])
    (output/'readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
