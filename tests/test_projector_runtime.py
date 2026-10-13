"""Portable trainer lifecycle and hybrid packing without external model weights."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from lalm_core.model import LALMConfig, LALMForConditionalGeneration
from projector_runtime import install_eval_checkpointing
from train_projector import ProjectorTrainer


def tiny_model():
    torch.manual_seed(17)
    config = Qwen3_5TextConfig(
        vocab_size=40, hidden_size=64, intermediate_size=96,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, use_cache=False, linear_num_key_heads=1,
        linear_num_value_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4,
        full_attention_interval=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 1., "mrope_section": [11, 11, 10],
                         "mrope_interleaved": True},
    )
    lm = Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
    model = LALMForConditionalGeneration(
        LALMConfig(text_config=config, audio_dim=8, text_dim=64,
                   projector_hidden_size=80, projector_downsample_rate=2,
                   audio_token_id=39),
        language_model=lm, audio_tower=torch.nn.Identity(),
    ).eval()
    model.encode_audio = lambda audio_features, feature_lens: audio_features
    model._get_audio_output_lengths = lambda feature_lens: feature_lens
    return model


def test_hybrid_packing_preserves_logits_gradients_and_recomputation():
    model = tiny_model()
    install_eval_checkpointing(model.language_model)
    ids = torch.tensor([[2, 39, 39, 3, 4, 5, 6], [7, 39, 39, 8, 9, 0, 0]])
    mask = (ids != 0).long()
    labels = ids.clone()
    labels[:, :3] = -100
    labels[mask == 0] = -100
    raw = torch.randn(2, 4, 8)
    embedded = model._merge_input_ids_with_audio_features(
        ids, model.language_model.get_input_embeddings()(ids), raw, torch.tensor([4, 4])
    )
    with patch("torch.cuda.is_current_stream_capturing", return_value=False):
        packed, _ = model._forward_packed(embedded, mask, labels)
        independent = [model.language_model(
            inputs_embeds=embedded[i:i + 1, :n], labels=labels[i:i + 1, :n],
            use_cache=False,
        ) for i, n in enumerate(mask.sum(1).tolist())]
        torch.testing.assert_close(packed.logits, torch.cat([o.logits for o in independent], 1),
                                   atol=1e-6, rtol=1e-5)
        counts = (labels[:, 1:] != -100).sum(1)
        expected = sum(o.loss * n for o, n in zip(independent, counts)) / counts.sum()
        torch.testing.assert_close(packed.loss, expected, atol=1e-6, rtol=1e-5)
        parameters = tuple(model.projector.parameters())
        actual_grads = torch.autograd.grad(packed.loss, parameters, retain_graph=True)
        reference_grads = torch.autograd.grad(expected, parameters)
        for actual, reference in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
        assert sum(g.norm() for g in actual_grads) > 0
        assert all(p.grad is None for p in model.language_model.parameters())
        for altered in (0, 1):
            changed = embedded.detach().clone()
            changed[altered] += .5 * torch.randn_like(changed[altered])
            output, _ = model._forward_packed(changed, mask, labels)
            start, end = (7, 12) if altered == 0 else (0, 7)
            torch.testing.assert_close(output.logits[:, start:end], packed.logits[:, start:end],
                                       atol=1e-6, rtol=1e-5)


def test_portable_trainer_epoch_updates_only_projector_and_exports(tmp_path):
    model = tiny_model()
    install_eval_checkpointing(model.language_model)
    ids = torch.tensor([[2, 39, 39, 3, 4, 5]])
    labels = torch.tensor([[-100, -100, -100, -100, 4, 5]])
    teacher_ids = torch.tensor([[2, 10, 11, 3, 4, 5]])
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                 features=torch.randn(1, 4, 8), feature_lens=torch.tensor([4]),
                 asr_mask=torch.tensor([False]), batch_size=1,
                 teacher_inputs=dict(input_ids=teacher_ids, labels=labels,
                                     attention_mask=torch.ones_like(ids)))
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "lalm/configs/train_projector_27b.yaml")
    cfg.exp_dir = str(tmp_path)
    cfg.trainer.tensorboard = False
    cfg.trainer.optimizer.fused = False
    cfg.trainer.mixed_precision = None
    cfg.trainer.num_epochs = 1
    cfg.trainer.average_period = 1
    cfg.model.language_model = cfg.model.tokenizer = "tiny-test-model"
    cfg.model.gigaam_checkpoint = "synthetic-encoder"
    cfg.data.train_data_config = cfg.data.valid_data_config = "synthetic-data"
    data = SimpleNamespace(train_dl=[batch], valid_names=["tiny"], valid_dls=[[batch]])
    frozen = {name: p.detach().clone() for name, p in model.named_parameters()
              if not name.startswith("projector.")}
    before = model.projector.linear1.weight.detach().clone()
    trainer = ProjectorTrainer(cfg, model, data)
    trainer.device = torch.device("cpu")
    with patch("torch.cuda.is_current_stream_capturing", return_value=False):
        trainer.validate(0)
        trainer.run()
    assert trainer.global_step == 1
    assert not torch.equal(before, model.projector.linear1.weight)
    assert all(torch.equal(p, frozen[name]) for name, p in model.named_parameters()
               if name in frozen)
    checkpoint = torch.load(tmp_path / "epoch-1.pt", weights_only=True)
    assert checkpoint["epoch"] == checkpoint["step"] == 1
    assert set(checkpoint) == {"projector", "projector_ema", "optimizer", "scheduler",
                               "scaler", "step", "epoch", "config"}
    for name, p in model.projector.state_dict().items():
        torch.testing.assert_close(checkpoint["projector"][name], p)
    assert not list(tmp_path.glob("checkpoint-*.pt"))
    assert not list(tmp_path.glob("*.tmp"))
