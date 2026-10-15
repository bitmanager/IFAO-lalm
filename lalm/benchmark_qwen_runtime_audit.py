"""Read-only controls around the checkpoint owner's native generation path."""
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
    from benchmark_qwen_calls import inference_messages
    from lalm_core.model.gigaam_adapter import gigaam_output_length

    spec = json.loads(args.manifest.read_text())
    assert os.environ["CUDA_VISIBLE_DEVICES"] == spec["gpu_uuid"]
    runtime.GPUS = [spec["gpu_uuid"]]
    assert runtime.digest(spec["checkpoint"]) == spec["checkpoint_sha256"]
    model, processor = runtime.load_model()
    model.projector.load_state_dict(torch.load(
        spec["checkpoint"], map_location="cpu", weights_only=True)["projector"])
    model.requires_grad_(False).eval()
    model.language_model.gradient_checkpointing_disable()
    tok = processor.tokenizer
    report = dict(checkpoint_sha256=spec["checkpoint_sha256"], cases=[], status="running")
    original_generate = model.language_model.generate
    original_merge = model._merge_input_ids_with_audio_features
    merge_checks = []

    def checked_merge(ids, embeds, features, feature_lens=None):
        merged = original_merge(ids, embeds, features, feature_lens=feature_lens)
        mask = ids == model.config.audio_token_id
        assert torch.equal(merged[~mask], embeds[~mask]), "Non-audio embeddings changed"
        assert torch.isfinite(merged).all(), "Nonfinite embeddings"
        assert not torch.equal(merged[mask], embeds[mask]), "Audio not inserted"
        merge_checks.append(dict(audio_positions=int(mask.sum()),
                                 audio_embedding_std=float(merged[mask].float().std())))
        return merged

    model._merge_input_ids_with_audio_features = checked_merge
    for case in spec["cases"]:
        assert runtime.digest(case["audio_path"]) == case["audio_sha256"]
        wave, sr = sf.read(case["audio_path"], dtype="float32")
        assert sr == 16000 and wave.ndim == 1
        wave = torch.from_numpy(wave.copy())
        assert torch.isfinite(wave).all()
        count = int(gigaam_output_length(torch.tensor(wave.numel()))) // 2
        user = dict(role="user", content="<|audio|>" * count + "\n" + spec["user_suffix"])
        prefixes = {mode: tokens(tok, inference_messages(
            spec["system_prompt"], case["history"], user, mode))
            for mode in ("reference_history_v3", "no_history")}
        assert all(p.count(model.config.audio_token_id) == count for p in prefixes.values())
        row = dict(id=case["id"], prior_output=case["prior_output"], controls={})
        report["cases"].append(row)
        for name, mode, silence in (
            ("history", "reference_history_v3", False),
            ("no_history", "no_history", False),
            ("silence_history", "reference_history_v3", True),
            ("history_repeat", "reference_history_v3", False),
        ):
            def capture_generation(**kwargs):
                if name != "history":
                    return original_generate(**kwargs)
                cached = original_generate(**kwargs, return_dict_in_generate=True, output_scores=True)
                # HF forces cache on for inputs_embeds generation. Test actual
                # uncached forward on the identical generated token prefixes.
                embeds = kwargs["inputs_embeds"]
                continuation = model.language_model.get_input_embeddings()(cached.sequences[:, :-1])
                joined = torch.cat((embeds, continuation), dim=1)
                n = cached.sequences.shape[1]
                uncached = model.language_model(
                    inputs_embeds=joined, attention_mask=torch.ones(joined.shape[:2], device=joined.device, dtype=torch.long),
                    use_cache=False, logits_to_keep=n).logits.float()
                uncached[:, :, kwargs["suppress_tokens"]] = -torch.inf
                row["uncached_forward_check"] = dict(
                    tokens=n, argmax_matches=int((uncached.argmax(-1) == cached.sequences).sum()),
                    first_token_matches=bool(uncached[0, 0].argmax() == cached.sequences[0, 0]))
                return cached.sequences
            model.language_model.generate = capture_generation
            merge_checks.clear()
            answer = generated(model, tok, prefixes[mode], torch.zeros_like(wave) if silence else wave)
            row["controls"][name] = dict(output=answer, merge_checks=list(merge_checks))
            runtime.atomic(args.output, report)
            print(json.dumps(dict(case=case["id"], control=name, **answer), ensure_ascii=False), flush=True)
        row["repeat_identical"] = row["controls"]["history"]["output"]["text"] == row["controls"]["history_repeat"]["output"]["text"]
        model.language_model.generate = original_generate
        # Independent ASR hypothesis is a text-path control, never human gold.
        if case.get("text_control"):
            text_user = dict(role="user", content=case["text_control"] + "\n" + spec["user_suffix"])
            row["text_control_input"] = case["text_control"]
            for mode in ("reference_history_v3", "no_history"):
                prefix = tokens(tok, inference_messages(spec["system_prompt"], case["history"], text_user, mode))
                row["controls"]["text_" + mode] = dict(output=generated(model, tok, prefix))
            runtime.atomic(args.output, report)
    model.language_model.generate = original_generate
    assert runtime.digest(spec["checkpoint"]) == spec["checkpoint_sha256"]
    report["status"] = "complete"
    runtime.atomic(args.output, report)


if __name__ == "__main__":
    main()
