"""Intermediate readout uses native Qwen decoding and isolated packed tasks."""
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from lalm_core.model import LALMConfig, LALMForConditionalGeneration


@pytest.fixture
def model():
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=40, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=3, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16, tie_word_embeddings=True)
    lm = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    model = LALMForConditionalGeneration(
        LALMConfig(text_config=config, audio_dim=8, text_dim=32),
        language_model=lm, audio_tower=torch.nn.Identity(),
    ).eval()
    model.enable_asr(2)
    return model


@pytest.mark.parametrize("tasks", [[False, True], [False, False], [True, True]])
def test_packed_matches_native_generation_view_and_gradients(model, tasks):
    ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    labels = ids.clone()
    labels[:, :2] = -100
    labels[mask == 0] = -100
    embeds = model.language_model.get_input_embeddings()(ids).detach().requires_grad_()
    output, packed_labels = model._forward_packed(embeds, mask, labels, asr_mask=torch.tensor(tasks))
    expected = []
    for i, task in enumerate(tasks):
        lm = model._asr_language_model() if task else model.language_model
        expected.append(lm(inputs_embeds=embeds[i:i+1, :mask[i].sum()]).logits)
    torch.testing.assert_close(output.logits, torch.cat(expected, dim=1), atol=1e-6, rtol=1e-5)
    output.loss.backward()
    assert torch.isfinite(embeds.grad).all() and embeds.grad.norm() > 0
    assert model.asr_head.weight.grad is not None  # DDP-safe even for an answer-only batch
    assert torch.isfinite(model.asr_head.weight.grad).all()
    assert (model.asr_head.weight.grad.norm() > 0).item() == any(tasks)
    assert all(p.grad is None for p in model.language_model.parameters())
    assert model.asr_head.weight.data_ptr() != model.language_model.lm_head.weight.data_ptr()
    assert packed_labels.shape == (1, 7)


def test_view_never_changes_agent_and_uses_native_generate(model):
    layers, norm, head = model.language_model.model.layers, model.language_model.model.norm, model.language_model.lm_head
    view = model._asr_language_model()
    assert len(view.model.layers) == 2
    assert view.model.layers[0] is layers[0]
    assert view.model.norm is model.asr_norm and view.lm_head is model.asr_head
    view.generate(input_ids=torch.tensor([[2, 3]]), max_new_tokens=2, do_sample=False, pad_token_id=0)
    assert model.language_model.model.layers is layers and len(layers) == 3
    assert model.language_model.model.norm is norm and model.language_model.lm_head is head
    assert model.language_model.config.num_hidden_layers == 3
    assert model.language_model.model.config.num_hidden_layers == 3


def test_packed_examples_do_not_leak_targets(model):
    ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    labels = ids.clone()
    labels[:, :2] = -100
    kwargs = dict(attention_mask=torch.ones_like(ids), labels=labels, asr_mask=torch.tensor([False, True]))
    first = model(input_ids=ids, **kwargs).logits.detach()
    ids[1] = torch.tensor([10, 11, 12, 13])
    second = model(input_ids=ids, **kwargs).logits.detach()
    torch.testing.assert_close(first[:, :4], second[:, :4])
    ids[0, 3] = 14
    third = model(input_ids=ids, **kwargs).logits.detach()
    torch.testing.assert_close(second[:, :3], third[:, :3])
