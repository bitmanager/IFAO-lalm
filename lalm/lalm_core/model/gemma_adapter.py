"""Frozen EmbeddingGemma 2 audio tower for the existing packed-waveform interface."""

import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel, PretrainedConfig


def gemma_output_length(samples):
    # Semicausal 320-sample windows, hop 160; two stride-2 convolutions.
    return (((samples - 161) // 160 + 1) + 3) // 4


class GemmaAudioConfig(PretrainedConfig):
    model_type = "embeddinggemma_audio"

    def __init__(self, model_id="google/embeddinggemma-2", **kwargs):
        self.model_id = model_id
        super().__init__(**kwargs)


class GemmaAudioTower(nn.Module):
    def __init__(self, config, model=None, feature_extractor=None):
        super().__init__()
        self.config = config
        if model is None:
            full = AutoModel.from_pretrained(
                config.model_id, dtype=torch.bfloat16, vision_config=None
            )
            model = full.audio_tower
        self.model = model
        # Expose the final Conformer state, not its Gemma-specific output projection.
        self.model.output_proj = nn.Identity()
        self.audio_dim = self.model.config.hidden_size
        self.feature_extractor = feature_extractor or AutoFeatureExtractor.from_pretrained(config.model_id)
        fe = self.feature_extractor
        if (fe.sampling_rate, fe.frame_length, fe.hop_length) != (16000, 320, 160):
            raise ValueError("Unsupported Gemma audio timing configuration")
        self.requires_grad_(False).eval()

    @classmethod
    def from_pretrained(cls, model_id):
        return cls(GemmaAudioConfig(model_id=model_id))

    @torch.no_grad()
    def forward(self, packed_waveforms, lengths):
        self.model.eval()
        waveforms = [x.float().cpu().numpy() for x in packed_waveforms.squeeze(0).split(lengths.tolist())]
        features = self.feature_extractor(
            waveforms, sampling_rate=16000, truncation=False,
            padding=True, return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        encoded = self.model(
            features.input_features.to(device),
            attention_mask=features.input_features_mask.to(device),
            return_dict=True,
        )
        valid = encoded.attention_mask.bool()
        if not torch.equal(valid.sum(-1), gemma_output_length(lengths).to(device)):
            raise ValueError("Gemma lengths do not match audio placeholders")
        return encoded.last_hidden_state.masked_fill(~valid[..., None], 0)
