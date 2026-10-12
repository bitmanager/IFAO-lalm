"""Call-level adapter around the existing KL1532 loader and greedy inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    import soundfile as sf
    import torch
    import kl_runtime as runtime
    from evaluate_kl import generated, tokens
    from lalm_core.model.gigaam_adapter import gigaam_output_length

    spec = json.loads(args.manifest.read_text())
    gpu = "GPU-de629798-46d4-ac37-e28e-d73f59094ab7"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == gpu
    runtime.GPUS = [gpu]
    checkpoint = Path(spec["checkpoint"])
    assert runtime.digest(checkpoint) == spec["checkpoint_sha256"]
    args.output.mkdir(parents=True, exist_ok=True)
    model, processor = runtime.load_model()
    model.projector.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["projector"])
    model.requires_grad_(False).eval()
    model.language_model.gradient_checkpointing_disable()
    tokenizer = processor.tokenizer
    recipe = {k: spec[k] for k in ("checkpoint_sha256", "system_prompt", "user_suffix")}
    recipe["adapter_sha256"] = runtime.digest(Path(__file__))
    recipe["native_inference_sha256"] = runtime.digest(Path(__import__("evaluate_kl").__file__))
    calls = spec["calls"][:args.limit] if args.limit else spec["calls"]
    for call_index, call in enumerate(calls):
        assert call["segments"], "No speech segments: cannot mark a call transcribed"
        key = fingerprint(dict(recipe=recipe, call=call))
        target = args.output / (key + ".json")
        result = json.loads(target.read_text()) if target.exists() else dict(
            key=key, recipe=recipe, call_id=call["call_id"], media_id=call["media_id"],
            environment=call["environment"], audio_sha256=call["user_sha256"],
            provenance={k: v for k, v in call.items() if k != "segments"}, segments=[], status="running")
        assert result["key"] == key
        audio = Path(call["user_audio_path"])
        assert runtime.digest(audio) == call["user_sha256"]
        if result["status"] == "complete":
            continue
        info = sf.info(audio)
        assert info.samplerate == 16000 and info.channels == 1
        for index, segment in enumerate(call["segments"]):
            if index < len(result["segments"]):
                assert result["segments"][index]["source"] == segment
                continue
            start, end = round(segment["start"] * 16000), round(segment["end"] * 16000)
            assert 0 <= start < end <= info.frames and end - start <= 30 * 16000
            wave, rate = sf.read(audio, start=start, stop=end, dtype="float32")
            wave = torch.from_numpy(wave.copy())
            assert rate == 16000 and torch.isfinite(wave).all()
            count = int(gigaam_output_length(torch.tensor(wave.numel()))) // 2
            history = []
            for h in segment["history"]:
                assert h["source_at_seconds"] < segment["start"]
                assert h["role"] in ("user", "assistant") and "<|" not in h["content"]
                history.append({"role": h["role"], "content": h["content"]})
            user = {"role": "user", "content": "<|audio|>" * count + "\n" + spec["user_suffix"]}
            while True:
                prefix = tokens(tokenizer, [{"role": "system", "content": spec["system_prompt"]}] + history + [user])
                if len(prefix) <= 8192:
                    break
                assert history, "Audio/system prompt exceeds inference context budget"
                history.pop(0)
            assert prefix.count(tokenizer.convert_tokens_to_ids("<|audio|>")) == count
            answer = generated(model, tokenizer, prefix, wave)
            result["segments"].append(dict(source=segment, output=answer, prompt_tokens=len(prefix),
                                           history_used=len(history), history_dropped=len(segment["history"]) - len(history)))
            runtime.atomic(target, result)
            print(json.dumps(dict(call=call_index + 1, calls=len(calls), segment=index + 1,
                                  segments=len(call["segments"]), generation_complete=answer["complete"])), flush=True)
        assert runtime.digest(audio) == call["user_sha256"]
        result["status"] = "complete" if all(s["output"]["complete"] for s in result["segments"]) else "truncated"
        result["transcript"] = "\n".join(s["output"]["text"] for s in result["segments"])
        result["human_gold"] = False
        runtime.atomic(target, result)
    assert runtime.digest(checkpoint) == spec["checkpoint_sha256"]


if __name__ == "__main__":
    main()
