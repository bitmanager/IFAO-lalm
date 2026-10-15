"""Call-level adapter around the existing KL1532 loader and greedy inference."""
import argparse
import hashlib
import json
import inspect
import marshal
import os
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def inference_messages(system, history, user, mode):
    if mode == "no_history":
        return [{"role": "system", "content": system}, user]
    if mode == "reference_history_v3":
        context = "\n".join(h["role"] + ": " + h["content"] for h in history)
        current = dict(user)
        if context:
            current["content"] = ("Справка о прошлых репликах:\n" + context
                                  + "\nКонец справки. ТЕКУЩЕЕ АУДИО:\n" + user["content"])
        return [{"role": "system", "content": system}, current]
    if mode == "reference_history_v2":
        # Context is evidence, not a conversation the model should continue.
        if history:
            system += ("\nИстория ниже — только справка. Не копируй её; расшифруй новый звук.\n"
                       + "\n".join(h["role"] + ": " + h["content"] for h in history))
        return [{"role": "system", "content": system}, user]
    if mode != "self_transcripts_and_agent":
        raise ValueError("Unknown history format")
    return [{"role": "system", "content": system}] + history + [user]


def context_copy_flag(text, history):
    # A review signal only: genuine repetitions are possible. Never edit output.
    normalized = " ".join(text.casefold().split()).strip(" .,!?:;—-")
    return len(normalized) >= 20 and any(
        normalized == " ".join(h["content"].casefold().split()).strip(" .,!?:;—-")
        for h in history)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sglang-url", help="Use a validated external-embedding SGLang server")
    parser.add_argument("--sglang-embedding-format", choices=("json", "bytes"), default="json")
    args = parser.parse_args()
    import soundfile as sf
    import torch
    import kl_runtime as runtime
    from evaluate_kl import generated, tokens
    from lalm_core.model.gigaam_adapter import gigaam_output_length

    spec = json.loads(args.manifest.read_text())
    encoder_name = spec.get("encoder_name", "gigaam")
    if encoder_name != "gigaam" and not args.sglang_url:
        raise ValueError("Alternative audio encoders require the SGLang frontend")
    assert spec["history_mode"] in ("self_transcripts_and_agent", "reference_history_v2", "reference_history_v3", "no_history")
    gpu = spec.get("gpu_uuid", "GPU-de629798-46d4-ac37-e28e-d73f59094ab7")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == gpu
    if not args.sglang_url:
        assert torch.cuda.mem_get_info()[0] >= 70 * 1024**3, "Selected GPU lacks room; do not disturb existing jobs"
    runtime.GPUS = [gpu]
    checkpoint = Path(spec["checkpoint"])
    assert runtime.digest(checkpoint) == spec["checkpoint_sha256"]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.sglang_url:
        import benchmark_sglang_client as sglang_client
        model = sglang_client.SGLangClient(
            checkpoint, args.sglang_url, args.sglang_embedding_format,
            encoder_name=encoder_name, encoder_path=spec.get("encoder_path"))
        audio_output_length = model.output_length
        tokenizer = model.tokenizer
        generated = sglang_client.generated
    else:
        audio_output_length = gigaam_output_length
        model, processor = runtime.load_model()
        model.projector.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["projector"])
        model.requires_grad_(False).eval()
        model.language_model.gradient_checkpointing_disable()
        tokenizer = processor.tokenizer
    recipe = {k: spec[k] for k in ("checkpoint_sha256", "system_prompt", "user_suffix", "history_mode")}
    recipe["adapter_sha256"] = runtime.digest(Path(__file__))
    recipe["native_inference_sha256"] = runtime.digest(Path(__import__("evaluate_kl").__file__))
    recipe["native_runtime_sha256"] = runtime.digest(Path(runtime.__file__))
    recipe["model_runtime_sha256"] = runtime.digest(Path(__import__("model_runtime").__file__))
    recipe["base_model_manifest_sha256"] = runtime.digest(runtime.ROOT / "model_download.json")
    recipe["encoder_name"] = encoder_name
    if encoder_name == "gigaam":
        recipe["gigaam_checkpoint_sha256"] = runtime.digest(Path(runtime.CKPT))
    if args.sglang_url:
        recipe["backend"] = "sglang-0.5.21-qwen-input-embeds"
        recipe["sglang_adapter_sha256"] = runtime.digest(Path(sglang_client.__file__))
        recipe["sglang_server_args"] = model.server_recipe
    recipe["tokenizer_files"] = {p.name: runtime.digest(p) for p in sorted((runtime.ROOT / "data/tokenizer").iterdir())
                                 if p.is_file()}
    recipe["loaded_model_methods"] = {
        name: hashlib.sha256(marshal.dumps(inspect.unwrap(getattr(runtime.LALMForConditionalGeneration, name)).__code__)).hexdigest()
        for name in ("__init__", "forward", "generate")}
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
        if result["status"] in ("complete", "truncated"):
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
            count = int(audio_output_length(torch.tensor(wave.numel()))) // 2
            history = []
            for h in segment["history"]:
                assert h["source_at_seconds"] < segment["start"]
                assert h["role"] == "assistant" and "<|" not in h["content"]
                history.append(dict(h))
            for prior in result["segments"]:
                assert prior["source"]["end"] <= segment["start"], "Overlapping speech segments"
                assert prior["output"]["complete"], "Cannot continue history after truncated generation"
                history.append(dict(role="user", content=prior["output"]["text"],
                                    source_at_seconds=prior["source"]["end"]))
            history.sort(key=lambda h: h["source_at_seconds"])
            history = [{"role": h["role"], "content": h["content"]} for h in history]
            if spec["history_mode"] == "no_history":
                history = []
            history_total = len(history)
            user = {"role": "user", "content": "<|audio|>" * count + "\n" + spec["user_suffix"]}
            while True:
                prefix = tokens(tokenizer, inference_messages(spec["system_prompt"], history, user, spec["history_mode"]))
                if len(prefix) <= 8192:
                    break
                assert history, "Audio/system prompt exceeds inference context budget"
                history.pop(0)
            assert prefix.count(tokenizer.convert_tokens_to_ids("<|audio|>")) == count
            answer = generated(model, tokenizer, prefix, wave)
            result["segments"].append(dict(source=segment, output=answer, prompt_tokens=len(prefix),
                                           history_used=len(history), history_dropped=history_total - len(history),
                                           history_sent=history,
                                           possible_context_copy=context_copy_flag(answer["text"], history)))
            runtime.atomic(target, result)
            print(json.dumps(dict(call=call_index + 1, calls=len(calls), segment=index + 1,
                                  segments=len(call["segments"]), generation_complete=answer["complete"])), flush=True)
            if not answer["complete"]:
                break
        assert runtime.digest(audio) == call["user_sha256"]
        result["status"] = "complete" if len(result["segments"]) == len(call["segments"]) and all(
            s["output"]["complete"] for s in result["segments"]) else "truncated"
        result["transcript"] = "\n".join(s["output"]["text"] for s in result["segments"])
        result["human_gold"] = False
        result["quality_review"] = dict(
            possible_context_copy_segments=[s["source"]["id"] for s in result["segments"]
                                            if s.get("possible_context_copy")],
            human_verified=False)
        runtime.atomic(target, result)
    assert runtime.digest(checkpoint) == spec["checkpoint_sha256"]


if __name__ == "__main__":
    main()
