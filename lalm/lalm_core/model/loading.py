"""Load the original language weights, including text-only Qwen3.8 checkpoints."""

from transformers import AutoConfig, AutoModelForCausalLM


def load_language_model(model_path, torch_dtype=None, **kwargs):
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type == "qwen3_5":
        # The official checkpoint also contains a vision tower. Preserve the
        # original text weights and vocab while loading only the text backbone.
        model, info = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config.text_config,
            dtype=torch_dtype,
            key_mapping={r"^model\.language_model\.": "model."},
            output_loading_info=True,
            **kwargs,
        )
        if info.get("missing_keys") or info.get("mismatched_keys"):
            raise RuntimeError("Incomplete language weights: " + str(info))
        unexpected = [
            k
            for k in info.get("unexpected_keys", [])
            if not k.startswith(("model.visual.", "visual.", "mtp."))
        ]
        if unexpected:
            raise RuntimeError(
                "Unexpected language weight keys: " + str(unexpected[:20])
            )
        return model
    return AutoModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype, **kwargs)
