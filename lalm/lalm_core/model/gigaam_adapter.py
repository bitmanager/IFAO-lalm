# PR provenance (implementation origin; this update adds comments only):
# PR #1: Entire adapter introduced in 4aac0de; encoder/loading/forward use official GigaAM.
# PR #19: Copied unchanged from #1 to make this PR self-contained on main.
# https://github.com/bitmanager/IFAO-lalm/pull/1
# https://github.com/bitmanager/IFAO-lalm/pull/19

"""GigaAM's official batched forward adapted to the packed IFAO audio interface."""

# PR #1 integration glue: main's HF encoder routes do not load GigaAM's native
# OmegaConf/.ckpt format. This wrapper delegates loading and encoding to GigaAM;
# it converts IFAO packed waveforms (1, sum(samples)) to a padded waveform batch
# and encoder output (batch, channels, frames) to (batch, frames, channels).
# The timing formula/guard is specific to the historical encoder checkpoint.
# Upstream implementation used by this wrapper:
# https://github.com/salute-developers/GigaAM/blob/7447938d791c4f3e643386ee22c33777004293a5/gigaam/model.py

import torch
from torch import nn
from transformers import PretrainedConfig


def gigaam_output_length(samples):
    # PR #1: predict audio placeholder count, then verify official output lengths.
    # Historical timing: window 320, hop 160, center=False, subsampling 4;
    # __init__ rejects incompatible configs rather than guessing frame counts.
    return (((samples - 320) // 160 + 1) + 3) // 4


class GigaAMConfig(PretrainedConfig):
    # PR #1: expose native GigaAM config through LALM's HF config contract.
    model_type = "gigaam"

    def __init__(self, gigaam_config=None, **kwargs):
        self.gigaam_config = gigaam_config
        super().__init__(**kwargs)


class GigaAMAudioTower(nn.Module):
    def __init__(self, config, model=None):
        super().__init__()
        from gigaam import GigaAM
        from omegaconf import OmegaConf

        self.config = config
        cfg = OmegaConf.create(config.gigaam_config)
        p, e = cfg.preprocessor, cfg.encoder
        if (p.sample_rate, p.win_length, p.hop_length, p.center, e.subsampling_factor) != (16000, 320, 160, False, 4):
            raise ValueError("Unsupported GigaAM timing configuration")
        self.model = model if model is not None else GigaAM(cfg)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        import gigaam
        from omegaconf import OmegaConf

        # PR #1: official loader; use encoder features and discard the ASR head.
        model = gigaam.load_model(checkpoint, device="cuda", fp16_encoder=True, use_flash=True)
        del model.head
        config = GigaAMConfig(gigaam_config=OmegaConf.to_container(model.cfg, resolve=True))
        return cls(config, model)

    def forward(self, packed_waveforms, lengths):
        # PR #1: undo IFAO waveform packing for official GigaAM batched forward.
        waveforms = nn.utils.rnn.pad_sequence(
            packed_waveforms.squeeze(0).split(lengths.tolist()), batch_first=True
        )
        # Keep the official preprocessor in FP32 and the frozen encoder in eval mode.
        self.model.eval()
        self.model.preprocessor.float()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            # Official preprocessor/encoder implementation, frozen for this task.
            encoded, encoded_lengths = self.model(waveforms.float(), lengths)
        if not torch.equal(encoded_lengths, gigaam_output_length(lengths)):
            raise ValueError("GigaAM lengths do not match the processor audio placeholders")
        valid = torch.arange(encoded.size(2), device=encoded.device)[None] < encoded_lengths[:, None]
        # PR #1: projector consumes time-major features; discard padded frames.
        return encoded.transpose(1, 2).masked_fill(~valid[..., None], 0)
