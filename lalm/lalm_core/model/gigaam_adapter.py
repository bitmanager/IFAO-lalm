"""GigaAM's official batched forward adapted to the packed IFAO audio interface."""

import torch
from torch import nn
from transformers import PretrainedConfig


def gigaam_output_length(samples):
    # Our v3 checkpoint: uncentered 320-sample window, hop 160, subsampling 4.
    return (((samples - 320) // 160 + 1) + 3) // 4


class GigaAMConfig(PretrainedConfig):
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

        model = gigaam.load_model(checkpoint, device="cuda", fp16_encoder=True, use_flash=True)
        del model.head
        config = GigaAMConfig(gigaam_config=OmegaConf.to_container(model.cfg, resolve=True))
        return cls(config, model)

    def forward(self, packed_waveforms, lengths):
        waveforms = nn.utils.rnn.pad_sequence(
            packed_waveforms.squeeze(0).split(lengths.tolist()), batch_first=True
        )
        # Keep the official preprocessor in FP32 and the frozen encoder in eval mode.
        self.model.eval()
        self.model.preprocessor.float()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            encoded, encoded_lengths = self.model(waveforms.float(), lengths)
        if not torch.equal(encoded_lengths, gigaam_output_length(lengths)):
            raise ValueError("GigaAM lengths do not match the processor audio placeholders")
        valid = torch.arange(encoded.size(2), device=encoded.device)[None] < encoded_lengths[:, None]
        return encoded.transpose(1, 2).masked_fill(~valid[..., None], 0)
