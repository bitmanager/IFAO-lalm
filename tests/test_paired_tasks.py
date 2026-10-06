"""Atomic linked views, missing labels, native packing and independently normalized CE."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from lhotse import AudioSource, CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset.sampling.base import TokenConstraint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from prepare_paired_tasks import prepare_paired_cut
from lalm_core.paired_tasks import expand_task_views
from lalm_core.data_module import estimate_cut_tokens, LALMDataset
from lalm_core.task_loss import separate_task_ce
from lalm_core.trainer import task_validation_values
from test_asr_readout import model


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=text.split())


def source(transcript="Текущий вопрос", answer="Точный ответ"):
    recording = Recording(id='audio', sampling_rate=16000, num_samples=16000, duration=1,
        sources=[AudioSource(type='file', channels=[0], source='/same.wav')])
    sup = SupervisionSegment(id='sup', recording_id='audio', start=0, duration=1,
        channel=0, text=transcript, language='ru', custom={'answer': answer})
    return MonoCut(id='source', start=0, duration=1, channel=0, recording=recording,
        supervisions=[sup], custom={'history': [{'role': 'user', 'content': 'История'},
            {'role': 'assistant', 'content': 'Прошлый ответ'}]})


def test_pair_preserves_audio_history_and_does_not_leak_current_targets():
    original = source()
    before = copy.deepcopy(original.to_dict())
    unit = prepare_paired_cut(original, Tokenizer(), 'Система')
    views = list(expand_task_views(CutSet.from_cuts([unit])))
    assert {v.task for v in views} == {'asr', 'answer'}
    assert original.to_dict() == before
    assert all(v.recording.to_dict() == original.recording.to_dict() for v in views)
    assert all(v.history == original.history and v.task_pair_id == original.id for v in views)
    for view in views:
        prefix = str(view.conversation[:-1])
        assert 'Текущий вопрос' not in prefix and 'Точный ответ' not in prefix
        assert view.rendered_conversation.endswith(view.supervisions[0].custom['answer'] + '<|im_end|>\n')
    assert views[0].supervisions[0].custom['answer'] == 'Текущий вопрос'
    assert views[1].supervisions[0].custom['answer'] == 'Точный ответ'


@pytest.mark.parametrize('transcript,answer,tasks', [
    ('Текст', None, ['asr']), ('Текст', '', ['asr']),
    (None, 'Ответ', ['answer']), ('', 'Ответ', ['answer']),
])
def test_missing_labels_never_create_false_tasks(transcript, answer, tasks):
    unit = prepare_paired_cut(source(transcript, answer), Tokenizer())
    assert [v.task for v in expand_task_views(CutSet.from_cuts([unit]))] == tasks


def test_no_labels_and_already_task_adapted_input_are_rejected():
    with pytest.raises(ValueError, match='neither'):
        prepare_paired_cut(source(None, None), Tokenizer())
    cut = source()
    cut.custom['task'] = 'asr'
    with pytest.raises(ValueError, match='base cuts'):
        prepare_paired_cut(cut, Tokenizer())


def test_pair_cannot_silently_lose_one_view_or_change_asr_target():
    unit = prepare_paired_cut(source(), Tokenizer())
    class DropOne:
        def __call__(self, cuts):
            return torch.zeros(1, 2), torch.ones(1), CutSet.from_cuts([next(iter(cuts))])
    with pytest.raises(ValueError, match='dropped'):
        LALMDataset(DropOne(), None, paired_tasks=True)[CutSet.from_cuts([unit])]
    unit.task_views[0]['target'] = 'Другой текст'
    unit.task_views[0]['conversation'][-1]['content'] = 'Другой текст'
    with pytest.raises(ValueError, match='explicit transcript'):
        expand_task_views(CutSet.from_cuts([unit]))


def test_native_two_rank_sampler_keeps_pairs_atomic_and_budgets_both_views(tmp_path):
    unit = prepare_paired_cut(source(), Tokenizer())
    with pytest.raises(ValueError, match='paired_tasks'):
        estimate_cut_tokens(unit, 12.5)
    estimate_cut_tokens(unit, 12.5, paired_tasks=True)
    assert unit.num_tokens == sum(v['num_text_tokens'] + 12 for v in unit.task_views)
    units = CutSet.from_cuts([copy.deepcopy(unit).with_id(f'unit{i}') for i in range(20)])
    path = tmp_path/'pairs.jsonl.gz'
    units.to_file(path)
    seen = set()
    for rank in (0, 1):
        cuts = CutSet.from_file(path).repeat(1).map(lambda c: estimate_cut_tokens(c, 12.5, True))
        sampler = DynamicBucketingSampler(cuts, constraint=TokenConstraint(max_tokens=400),
            num_buckets=2, shuffle=False, drop_last=False, world_size=2, rank=rank)
        for batch in sampler:
            expanded = list(expand_task_views(batch))
            for c in batch:
                seen.add(c.id)
                pair = [v for v in expanded if v.task_pair_id == c.id]
                assert {v.task for v in pair} == {'asr', 'answer'}
                assert len(pair) == 2
    assert seen == set(units.repeat(1).ids)
    plain = source(); plain.custom.update(num_text_tokens=20, task='asr')
    assert estimate_cut_tokens(plain, 12.5).num_tokens == 32
    assert next(iter(expand_task_views(CutSet.from_cuts([plain])))).to_dict() == plain.to_dict()


def tensors(model):
    ids = torch.tensor([[2, 3, 4, 5, 0], [6, 7, 8, 9, 10]])
    mask = ids != 0
    labels = ids.masked_fill(~mask, -100)
    labels[:, :2] = -100
    return ids, mask.long(), labels, torch.tensor([True, False])


@pytest.mark.parametrize('weight', [0., .5, 2.])
def test_separate_ce_matches_independent_native_heads_and_lambda_gradient(model, weight):
    ids, mask, labels, tasks = tensors(model)
    embeds = model.language_model.get_input_embeddings()(ids).detach().requires_grad_()
    out, _ = model._forward_packed(embeds, mask, labels, asr_mask=tasks, answer_loss_weight=weight)
    asr = model._asr_language_model()(inputs_embeds=embeds[:1, :4], labels=labels[:1, :4]).loss
    answer = model.language_model(inputs_embeds=embeds[1:], labels=labels[1:]).loss
    torch.testing.assert_close(out.loss, asr + weight * answer)
    expected_grad = torch.autograd.grad(asr + weight * answer, embeds, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(out.loss, embeds, retain_graph=True)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)
    out.loss.backward()
    assert model.asr_head.weight.grad.norm() > 0
    assert all(p.grad is None for p in model.language_model.parameters())
    assert (actual_grad[1].norm() > 0).item() == (weight > 0)
    assert out.task_token_counts.tolist() == [2, 3]


def test_optin_all_ignored_zero_and_packed_boundaries(model):
    ids, mask, labels, tasks = tensors(model)
    embeds = model.language_model.get_input_embeddings()(ids).detach().requires_grad_()
    # A deliberately unmasked first token of the second segment must not become
    # supervision for the end of the previous segment.
    labels[1, 0] = 6
    out, packed = model._forward_packed(embeds, mask, labels, asr_mask=tasks, answer_loss_weight=1.)
    assert packed[0, 4] == -100 and out.task_token_counts.tolist() == [2, 3]
    out, _ = model._forward_packed(embeds, mask, torch.full_like(labels, -100),
        asr_mask=tasks, answer_loss_weight=1.)
    assert out.loss.item() == 0 and torch.isfinite(out.loss)
    out.loss.backward()
    assert torch.equal(embeds.grad, torch.zeros_like(embeds))
    assert out.task_token_counts.tolist() == [0, 0]


def test_default_pooled_ce_and_logits_are_unchanged(model):
    ids, mask, labels, tasks = tensors(model)
    legacy = model(input_ids=ids, attention_mask=mask, labels=labels, asr_mask=tasks)
    opted = model(input_ids=ids, attention_mask=mask, labels=labels, asr_mask=tasks, answer_loss_weight=1.)
    torch.testing.assert_close(legacy.logits, opted.logits, atol=0, rtol=0)
    expected = F.cross_entropy(legacy.logits[:, :-1].reshape(-1, 40), legacy.packed_labels[:, 1:].reshape(-1))
    torch.testing.assert_close(legacy.loss, expected)


@pytest.mark.parametrize('tasks', [[True, False], [False, False], [True, True]])
def test_projector_backward_through_frozen_backbone_and_empty_asr_route(model, tasks):
    ids, mask, labels, _ = tensors(model)
    model.config.audio_token_id = 39
    ids[:, 1] = 39
    model.encode_audio = lambda audio_features, feature_lens: audio_features
    model._get_audio_output_lengths = lambda lengths: lengths
    output = model(input_ids=ids, attention_mask=mask, labels=labels,
        audio_features=torch.randn(2, 4, 8), feature_lens=torch.tensor([4, 4]),
        asr_mask=torch.tensor(tasks), answer_loss_weight=2.)
    output.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.projector.parameters())
    assert model.projector.linear1.weight.grad.norm() > 0
    assert model.asr_head.weight.grad is not None
    assert model.asr_norm.weight.grad is not None
    assert (model.asr_head.weight.grad.norm() > 0).item() == any(tasks)
    assert all(p.grad is None for p in model.language_model.parameters())


def test_real_native_processor_pair_labels_masks_audio_and_eos(tmp_path):
    native = Path('/runs/dev-storage/ifao-data/runs/asr-short-context-v1/export/iter-26000')
    if not native.exists():
        pytest.skip('Optional CPU integration requires the existing native tokenizer, no download')
    import numpy as np
    import soundfile as sf
    from lhotse.dataset import AudioSamples
    from lalm_core.model import LALMProcessor
    processor = LALMProcessor.from_pretrained(native)
    original = source()
    wav = tmp_path/'same.wav'
    sf.write(wav, np.sin(np.arange(16000)*.05).astype('float32')*.1, 16000)
    original.recording.sources[0].source = str(wav)
    unit = prepare_paired_cut(original, processor.tokenizer, 'Ты голосовой ассистент.')
    dataset = LALMDataset(AudioSamples(), processor, return_cuts=True, paired_tasks=True)
    batch = dataset[CutSet.from_cuts([unit])]
    assert batch['batch_size'] == 2
    assert batch['asr_mask'].sum() == 1
    for cut, labels, route in zip(batch['cuts'], batch['labels'], batch['asr_mask']):
        assert route.item() == (cut.task == 'asr')
        decoded = processor.tokenizer.decode(labels[labels != -100], skip_special_tokens=False).strip()
        assert decoded == cut.supervisions[0].custom['answer'] + '<|im_end|>'
        assert 'Прошлый ответ' not in decoded
    lens = batch['feature_lens'].tolist()
    assert lens[0] == lens[1]
    torch.testing.assert_close(batch['features'][..., :lens[0]], batch['features'][..., lens[0]:])


def test_task_validation_uses_summed_numerators_and_denominators():
    assert task_validation_values(dict(asr_nll=12, asr_target_tokens=4,
        answer_nll=2, answer_target_tokens=1), 3) == dict(asr_ce=3, answer_ce=2, loss=9)
    assert task_validation_values(dict(asr_nll=0, asr_target_tokens=0,
        answer_nll=0, answer_target_tokens=0), 3)['loss'] == 0


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError, match='finite'):
        separate_task_ce(torch.zeros(1, 2, 3), torch.tensor([[-100, 1]]),
            torch.tensor([True, True]), weight)
