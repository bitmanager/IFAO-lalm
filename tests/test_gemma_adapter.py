"""CPU-only contracts; no downloaded weights or accelerator allocation."""
import sys
from pathlib import Path

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lalm'))
from lalm_core.model.gemma_adapter import GemmaAudioConfig, GemmaAudioTower, gemma_output_length
from lalm_core.model.configuration_lalm import LALMConfig
from lalm_core.model.modeling_lalm import LALMProjector
from lalm_core.model.processing_lalm import make_audio_token_length_fn


def tower():
    cfg = Gemma4AudioConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                           output_proj_dims=24, subsampling_conv_channels=[128, 32])
    return GemmaAudioTower(GemmaAudioConfig(), Gemma4AudioModel(cfg), Gemma4AudioFeatureExtractor())


@pytest.mark.parametrize('lengths', [[8000, 16000], [15999, 16001], [1441, 1601]])
def test_packed_lengths_padding_and_projector_gradients(lengths):
    torch.manual_seed(1)
    enc = tower()
    enc.train()  # parent training must not enable frozen encoder dropout
    lengths = torch.tensor(lengths)
    waves = torch.randn(1, int(lengths.sum())) * 0.01
    output = enc(waves, lengths)
    actual_lengths = gemma_output_length(lengths)
    assert output.shape[-1] == 16
    assert not output.requires_grad and not enc.model.training
    for i, n in enumerate(actual_lengths):
        assert torch.count_nonzero(output[i, n:]) == 0
    # A short item must not change when batched with a longer waveform.
    alone = enc(waves[:, :lengths[0]], lengths[:1])
    torch.testing.assert_close(output[0, :actual_lengths[0]], alone[0, :actual_lengths[0]], atol=1e-5, rtol=1e-4)
    projector = LALMProjector(16, 8, downsample_rate=2)
    projector(output).square().mean().backward()
    assert all(p.grad is None for p in enc.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters())
    token_length = make_audio_token_length_fn('embeddinggemma_audio', 2)
    assert [token_length(n) for n in lengths.tolist()] == (actual_lengths // 2).tolist()


def test_config_roundtrip():
    cfg = LALMConfig(audio_config=GemmaAudioConfig(), audio_dim=1024, text_dim=8)
    restored = LALMConfig.from_dict(cfg.to_dict())
    assert isinstance(restored.audio_config, GemmaAudioConfig)
    assert restored.audio_config.model_id == 'google/embeddinggemma-2'
