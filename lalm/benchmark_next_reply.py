"""Same-history next-reply controls using the owner's unchanged inference."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    import torch
    import soundfile as sf
    import gigaam
    import kl_runtime as runtime
    from evaluate_kl import generated, tokens
    from lalm_core.model.gigaam_adapter import gigaam_output_length

    spec = json.loads(args.manifest.read_text())
    assert os.environ["CUDA_VISIBLE_DEVICES"] == spec["gpu_uuid"]
    runtime.GPUS = [spec["gpu_uuid"]]
    assert torch.cuda.mem_get_info()[0] >= 70 * 1024**3, "Selected GPU lacks room; do not disturb existing jobs"
    assert runtime.digest(spec["checkpoint"]) == spec["checkpoint_sha256"]
    model, proc = runtime.load_model()
    model.projector.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)["projector"])
    model.requires_grad_(False).eval()
    model.language_model.gradient_checkpointing_disable()
    asr = gigaam.load_model(runtime.CKPT, device="cuda", fp16_encoder=False, use_flash=False).requires_grad_(False).eval()
    asr.preprocessor.float()
    result = dict(status="running", cases=[], checkpoint_sha256=spec["checkpoint_sha256"],
                  giga_checkpoint_sha256=runtime.digest(runtime.CKPT), system_prompt=spec["system_prompt"])
    for case in spec["cases"]:
        assert runtime.digest(case["audio_path"]) == case["audio_sha256"]
        wave, rate = sf.read(case["audio_path"], dtype="float32")
        assert rate == 16000 and wave.ndim == 1
        wave = torch.from_numpy(wave.copy())
        assert torch.isfinite(wave).all()
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            length = torch.tensor([wave.numel()], device="cuda")
            encoded, elen = asr(wave.unsqueeze(0).cuda(), length)
            transcript = asr._decode(encoded, elen, length, word_timestamps=False)[0][0]
        count = int(gigaam_output_length(torch.tensor(wave.numel()))) // 2
        history = [{"role": h["role"], "content": h["content"]} for h in case["history"]]
        messages = [dict(role="system", content=spec["system_prompt"])] + history
        row = dict(id=case["id"], history=history, reference=case["reference"],
                   offline_giga_text=transcript, outputs={})
        variants = dict(reference=case["reference"], offline_giga=transcript,
                        audio="<|audio|>" * count)
        if "online_text" in case:
            variants["online"] = case["online_text"]
            row["online_text"] = case["online_text"]
        for name, content in variants.items():
            prefix = tokens(proc.tokenizer, messages + [dict(role="user", content=content)])
            row["outputs"][name] = generated(model, proc.tokenizer, prefix, wave if name == "audio" else None)
        result["cases"].append(row)
        runtime.atomic(args.output, result)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    result["status"] = "complete"
    runtime.atomic(args.output, result)


if __name__ == "__main__":
    main()
