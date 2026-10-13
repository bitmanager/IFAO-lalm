"""Shifted response KD, unequal prefixes, native teacher rendering and freeze guards."""
import copy
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from lalm_core.response_kl import validate_response_targets
from lalm_core.task_loss import response_kl_loss
from test_asr_readout import model


def inputs(model):
    model.asr_head.requires_grad_(False)
    model.asr_norm.requires_grad_(False)
    model.config.audio_token_id = 39
    model.encode_audio = lambda audio_features, feature_lens: audio_features
    model._get_audio_output_lengths = lambda lengths: lengths
    ids = torch.tensor([[2, 39, 3, 4, 5, 9, 0], [6, 39, 7, 8, 9, 0, 0]])
    labels = torch.tensor([[-100, -100, -100, -100, 5, 9, -100],
                           [-100, -100, -100, 8, 9, -100, -100]])
    teacher_ids = torch.tensor([[2, 10, 11, 3, 4, 5, 9], [6, 7, 8, 9, 0, 0, 0]])
    teacher_labels = torch.tensor([[-100, -100, -100, -100, -100, 5, 9],
                                   [-100, -100, 8, 9, -100, -100, -100]])
    return dict(input_ids=ids, attention_mask=(ids != 0).long(), labels=labels,
                asr_mask=torch.tensor([False, False]), response_kl=True,
                audio_features=torch.randn(2, 4, 8), feature_lens=torch.tensor([4, 4]),
                teacher_inputs=dict(input_ids=teacher_ids, labels=teacher_labels,
                                    attention_mask=(teacher_ids != 0).long()))


def test_packed_kl_matches_independent_responses_and_projector_only_gradients(model):
    args = inputs(model)
    text_ids = torch.tensor([[2, 10, 11]])
    with torch.no_grad():
        text_before = model.language_model(input_ids=text_ids).logits.clone()
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters()
                     if not n.startswith('projector.')}
    model.train()
    calls = []
    hook = model.language_model.register_forward_pre_hook(
        lambda m, a: calls.append((torch.is_grad_enabled(), m.training)))
    head_hook = model.asr_head.register_forward_pre_hook(
        lambda *a: pytest.fail("KL must not evaluate the ASR head"))
    output = model(**args)
    hook.remove(); head_hook.remove()
    assert calls == [(True, False), (False, False)]
    assert output.response_target_tokens.item() == 4
    assert output.packed_labels[0, 6] == -100  # next packed segment start
    embed = model.language_model.get_input_embeddings()(args['input_ids'])
    embed = model._merge_input_ids_with_audio_features(
        args['input_ids'], embed, args['audio_features'], args['feature_lens'])
    terms = []
    for i in range(2):
        n = args['attention_mask'][i].sum()
        student = model.language_model(inputs_embeds=embed[i:i+1, :n]).logits
        t = args['teacher_inputs']; tn = t['attention_mask'][i].sum()
        with torch.no_grad():
            teacher = model.language_model(input_ids=t['input_ids'][i:i+1, :tn]).logits
        sm = args['labels'][i, 1:n] != -100; tm = t['labels'][i, 1:tn] != -100
        terms.append(F.kl_div(F.log_softmax(student[0, :-1][sm].float()/2, -1),
                              F.softmax(teacher[0, :-1][tm].float()/2, -1), reduction='sum'))
    expected = sum(terms) / 4
    torch.testing.assert_close(output.loss, expected, atol=1e-7, rtol=1e-4)
    grads = torch.autograd.grad(expected, tuple(model.projector.parameters()), retain_graph=True)
    output.loss.backward()
    for p, expected_grad in zip(model.projector.parameters(), grads):
        torch.testing.assert_close(p.grad, expected_grad, atol=1e-7, rtol=1e-4)
    assert model.projector.linear1.weight.grad.norm() > 0
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith('projector.'))
    with torch.no_grad():
        for p in model.projector.parameters():
            p.add_(p.grad, alpha=-.01)
        torch.testing.assert_close(model.language_model(input_ids=text_ids).logits, text_before, atol=0, rtol=0)
    assert all(torch.equal(p, frozen_before[n]) for n, p in model.named_parameters()
               if n in frozen_before)


def test_shift_includes_first_answer_and_eos_excludes_padding_and_after_eos():
    student = torch.randn(1, 7, 11, requires_grad=True)
    teacher = torch.randn(1, 9, 11, requires_grad=True)
    sl = torch.tensor([[-100, -100, 3, 4, 9, -100, -100]])
    tl = torch.tensor([[-100, -100, -100, -100, 3, 4, 9, -100, -100]])
    loss, numerator, count = response_kl_loss(student, teacher, sl, tl)
    assert count == 3 and numerator.dtype == torch.float32
    loss.backward()
    assert (student.grad.norm(dim=-1) > 0).tolist() == [[False, True, True, True, False, False, False]]
    assert teacher.grad is None


def test_pairwise_check_rejects_equal_flat_counts_but_different_targets():
    left = torch.tensor([[-100, 1, 2], [-100, 3, 4]])
    right = torch.tensor([[-100, 1, 3], [-100, 2, 4]])
    with pytest.raises(ValueError, match='within an example'):
        validate_response_targets(left, right, torch.ones_like(left), torch.ones_like(right))
    with pytest.raises(ValueError, match='padding'):
        validate_response_targets(left, left, torch.zeros_like(left), torch.ones_like(left))


@pytest.mark.parametrize('field', ['asr', 'mixed', 'llm', 'head', 'tower', 'projector', 'ce'])
def test_response_kl_rejects_wrong_routes_or_freeze(model, field):
    args = inputs(model)
    if field == 'asr': args['asr_mask'][:] = True
    elif field == 'mixed': args['asr_mask'][0] = True
    elif field == 'llm': model.language_model.requires_grad_(True)
    elif field == 'head': model.asr_head.requires_grad_(True)
    elif field == 'tower': model.audio_tower = torch.nn.Linear(8, 8)
    elif field == 'projector': model.projector.requires_grad_(False)
    else: args['answer_loss_weight'] = 1.
    with pytest.raises(ValueError, match='Response KL'):
        model(**args)


def test_native_processor_teacher_preserves_history_policy_and_suffix(tmp_path):
    native = Path('/runs/dev-storage/ifao-data/runs/asr-short-context-v1/export/iter-26000')
    if not native.exists():
        pytest.skip('Existing native tokenizer only; never download')
    import numpy as np
    import soundfile as sf
    from lhotse import CutSet
    from lhotse.dataset import AudioSamples
    from lalm_core.model import LALMProcessor
    from lalm_core.data_module import LALMDataset
    from prepare_conversation import prepare_cut
    from test_paired_tasks import source
    proc = LALMProcessor.from_pretrained(native)
    cut = source(); wav = tmp_path/'source.wav'
    sf.write(wav, np.sin(np.arange(16000)*.05).astype('float32')*.1, 16000)
    cut.recording.sources[0].source = str(wav)
    cut = prepare_cut(cut, proc.tokenizer, instruction='Учитывай собеседника из истории.', system='Система')
    original = copy.deepcopy(cut.to_dict())
    batch = LALMDataset(AudioSamples(), proc, response_kl=True)[CutSet.from_cuts([cut])]
    teacher = batch['teacher_inputs']; teacher_text = proc.tokenizer.decode(teacher['input_ids'][0])
    assert 'Текущий вопрос Учитывай собеседника из истории.' in teacher_text
    assert 'История' in teacher_text and 'Прошлый ответ' in teacher_text and 'Система' in teacher_text
    assert '<|audio|>' not in teacher_text
    assert proc.tokenizer.decode(teacher['labels'][0][teacher['labels'][0] != -100]) == 'Точный ответ<|im_end|>'
    assert cut.to_dict() == original
    cut.custom['task'] = 'asr'
    with pytest.raises(ValueError, match='answer-only'):
        LALMDataset(AudioSamples(), proc, response_kl=True)[CutSet.from_cuts([cut])]
    cut.custom.pop('task'); cut.custom['rendered_conversation'] += 'wrong'
    with pytest.raises(ValueError, match='rendering'):
        LALMDataset(AudioSamples(), proc, response_kl=True)[CutSet.from_cuts([cut])]
