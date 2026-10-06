"""Answer-only bounded diagnostic; filter existing v3 views, no new targets."""
import json
from pathlib import Path

import yaml
from lhotse import CutSet
from lhotse.dataset import AudioSamples
from lalm_core.data_module import LALMDataset
from lalm_core.model import LALMProcessor
from scripts.prepare_projector_semantic_v3 import MODEL, ROOT, finite_sampler_audit, sha256
from scripts.prepare_phone2_context_stage import identities

BASE = ROOT / 'runs/asr-projector-semantic-v3'
OUTPUT = ROOT / 'runs/asr-projector-answer-only-diagnostic-v1'


def main():
    assert not OUTPUT.exists(), OUTPUT
    ready = json.loads((BASE/'readiness.json').read_text())
    assert sha256(BASE/'train.jsonl.gz') == ready['manifest_sha256'] == '63ca8a9a7543bf4e72b78c916270945807527b2210afa40d3ee824fbc25ba728'
    assert ready['model_dir'] == str(MODEL) and ready['trainable_parameters'] == 10490880
    audit = json.loads((BASE/'frozen-state-audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['export_frozen_tensor_bytes_identical']
    cuts = CutSet.from_file(BASE/'train.jsonl.gz').filter(lambda c: c.custom.get('task', 'answer') == 'answer').to_eager()
    assert len(cuts) == len({c.id for c in cuts}) == 2538
    heldout = identities([c for path in ready['heldout_manifest_sha256'] for c in CutSet.from_file(path)])
    overlap = {key:len(value & heldout[key]) for key,value in identities(cuts).items()}
    assert not any(overlap.values()), overlap
    for c in cuts:
        assert c.answer_provenance['quality'] == 'original_script_synthetic_not_gold'
        assert c.custom['split'] == 'train' and not c.custom.get('source_call_id')
        assert .5 <= c.duration <= 30
        target = c.supervisions[0].custom['answer']
        assert c.conversation[-1] == {'role':'assistant', 'content':target}
        assert c.rendered_conversation.endswith(target+'<|im_end|>\n')
        # Only experiment bookkeeping changes; prompts, labels and source fields stay exact.
        c.custom['training_recipe'] = 'projector-answer-only-diagnostic-v1'
    sampler = finite_sampler_audit(cuts, required_updates=250)
    processor = LALMProcessor.from_pretrained(MODEL)
    ordered = sorted(cuts, key=lambda c:(c.duration,c.id))
    sample = CutSet.from_cuts(ordered[round(i*(len(ordered)-1)/15)] for i in range(16))
    batch = LALMDataset(AudioSamples(), processor, return_cuts=True)[sample]
    assert batch['batch_size'] == 16 and not batch['asr_mask'].any()
    for c,labels in zip(batch['cuts'],batch['labels']):
        assert processor.tokenizer.decode(labels[labels != -100],skip_special_tokens=False).strip() == c.supervisions[0].custom['answer']+'<|im_end|>',c.id
    OUTPUT.mkdir()
    manifest = OUTPUT/'train.jsonl.gz'; cuts.to_file(manifest)
    hours = sum(c.duration for c in cuts)/3600
    (OUTPUT/'train.yaml').write_text(yaml.safe_dump([dict(name='answer-only-diagnostic',manifest=str(manifest),hours=hours,weights=1)]))
    (OUTPUT/'validation.yaml').write_bytes((BASE/'validation.yaml').read_bytes())
    result = dict(status='ready_for_review_not_launched',cuts=2538,tasks={'answer':2538},unique_original_turns=846,
        task_exposure_hours=hours,manifest=str(manifest),manifest_sha256=sha256(manifest),
        train_config_sha256=sha256(OUTPUT/'train.yaml'),validation_config_sha256=sha256(OUTPUT/'validation.yaml'),
        input_sha256={str(BASE/'train.jsonl.gz'):ready['manifest_sha256'],str(BASE/'readiness.json'):sha256(BASE/'readiness.json')},
        source_provenance=ready['source_preflight']['original_answer'],model_dir=str(MODEL),
        model_file_sha256=ready['model_file_sha256'],heldout_manifest_sha256=ready['heldout_manifest_sha256'],heldout_overlap=overlap,
        trainable_parameters=10490880,trainable_tensors=ready['trainable_tensors'],frozen_modules=ready['frozen_modules'],
        finite_sampler=sampler,native_cpu_batch=dict(cuts=16,answer=16,asr=0,exact_final_target_and_EOS=True),
        target_token_share_answer=1.0,target_tokens_including_eos=83016,
        updates=250,optimizer='fresh AdamW; native cosine equal base/min LR 3e-6',training_launched=False,
        notes=['Diagnostic, not an established cure or causal ablation.',
               'Exact existing original-script answers repeated three times: 846 original turns, not new audio hours.',
               'No ASR views, new text, augmentation or architecture changes. Fixed validation unchanged.'])
    (OUTPUT/'readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
