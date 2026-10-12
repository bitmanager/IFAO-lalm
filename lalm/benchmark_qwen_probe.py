"""Run a prepared probe using the checkpoint owner's existing inference code."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import soundfile as sf
    import torch
    import kl_runtime as runtime
    from evaluate_kl import generated, tokens
    from lalm_core.model.gigaam_adapter import gigaam_output_length

    spec = json.loads(args.manifest.read_text())
    # The user explicitly reassigned this GPU from SGLang/vLLM to this probe.
    gpu = "GPU-de629798-46d4-ac37-e28e-d73f59094ab7"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == gpu
    runtime.GPUS = [gpu]
    checkpoint = Path(spec["checkpoint"])
    assert runtime.digest(checkpoint) == spec["checkpoint_sha256"]
    audio = Path(spec["audio_path"])
    assert runtime.digest(audio) == spec["audio_sha256"]
    wave, rate = sf.read(audio, dtype="float32")
    assert rate == 16000 and wave.ndim == 1
    wave = torch.from_numpy(wave.copy())
    assert torch.isfinite(wave).all()
    print("Loading specified KL1532 checkpoint on GPU0", flush=True)
    model, processor = runtime.load_model()
    model.projector.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["projector"])
    model.requires_grad_(False).eval()
    model.language_model.gradient_checkpointing_disable()
    tokenizer = processor.tokenizer
    count = int(gigaam_output_length(torch.tensor(wave.numel()))) // 2
    result = dict(checkpoint_sha256=spec["checkpoint_sha256"], audio_sha256=spec["audio_sha256"], variants=[])
    for variant in spec["variants"]:
        history = [{"role": h["role"], "content": h["content"]} for h in variant["history"]]
        messages = [{"role": "system", "content": spec["system_prompt"]}] + history
        messages.append({"role": "user", "content": "<|audio|>" * count + "\n" + spec["user_suffix"]})
        prefix = tokens(tokenizer, messages)
        assert prefix.count(tokenizer.convert_tokens_to_ids("<|audio|>")) == count
        answer = generated(model, tokenizer, prefix, wave)
        result["variants"].append(dict(name=variant["name"], history_messages=len(history), history_truncated=False,
                                       prompt_tokens=len(prefix), output=answer))
        runtime.atomic(args.output, result)
        print(json.dumps(result["variants"][-1], ensure_ascii=False), flush=True)
    assert runtime.digest(checkpoint) == spec["checkpoint_sha256"]
    result["status"] = "complete"
    runtime.atomic(args.output, result)


if __name__ == "__main__":
    main()
