"""Warm-start an auxiliary Qwen ASR readout from a native trainer checkpoint."""
import argparse
from pathlib import Path

import torch
from transformers.modeling_utils import no_init_weights

from lalm_core.model import LALMConfig, LALMForConditionalGeneration, LALMProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    hf = args.checkpoint.parent / "hf"
    config = LALMConfig.from_pretrained(hf)
    if config.asr_layer is not None:
        raise ValueError("Checkpoint already contains an ASR readout")
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    with no_init_weights():
        model = LALMForConditionalGeneration(config).to(torch.bfloat16)
    model.load_state_dict(state["model"], strict=True)
    model.enable_asr(config.text_config.num_hidden_layers - 1)
    assert torch.equal(model.projector.linear1.weight, state["model"]["projector.linear1.weight"].to(torch.bfloat16))
    model.save_pretrained(args.output_dir)
    LALMProcessor.from_pretrained(hf).save_pretrained(args.output_dir)
    print({"source": str(args.checkpoint), "source_step": state.get("batch_idx_train"),
           "asr_layer": model.config.asr_layer,
           "asr_parameters": sum(p.numel() for m in (model.asr_norm, model.asr_head) for p in m.parameters()),
           "optimizer": "new stage; weights preserved, optimizer initialized afresh"})


if __name__ == "__main__":
    main()
