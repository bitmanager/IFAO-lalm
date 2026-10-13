"""Generate frozen-Qwen answers and prepare native answer-only Lhotse cuts."""

import argparse
from pathlib import Path

import torch
from lalm_core.model.loading import load_language_model
from lhotse import CutSet
from prepare_conversation import prepare_cut, teacher_messages
from projector_runtime import install_rendering
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, help="Pinned local HF snapshot")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--reuse-answers", action="store_true")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|audio|>"]})
    install_rendering()
    model = None
    if not args.reuse_answers:
        model = (
            load_language_model(
                args.model,
                torch_dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
                attn_implementation="sdpa",
            )
            .eval()
            .requires_grad_(False)
        )
    with CutSet.open_writer(args.output, overwrite=False) as writer:
        for cut in CutSet.from_file(args.input):
            if len(cut.supervisions) != 1:
                raise ValueError(f"{cut.id}: expected one foreground transcript")
            sup = cut.supervisions[0]
            if not sup.text or not sup.text.strip():
                raise ValueError(f"{cut.id}: empty transcript")
            if model is not None:
                prompt = tokenizer.apply_chat_template(
                    teacher_messages(cut),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=False
                ).to("cuda:0")
                with torch.inference_mode():
                    result = model.generate(
                        **inputs, do_sample=False, max_new_tokens=args.max_new_tokens
                    )
                generated = result[0, inputs.input_ids.shape[1] :]
                if generated[-1].item() != tokenizer.eos_token_id:
                    raise ValueError(
                        f"{cut.id}: incomplete teacher answer; raise token budget"
                    )
                answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
                sup.custom = {**(sup.custom or {}), "answer": answer}
            answer = (sup.custom or {}).get("answer")
            if not answer or any(
                token in answer for token in ("<|", "<think>", "</think>")
            ):
                raise ValueError(f"{cut.id}: invalid no-thinking answer")
            cut.custom = {**(cut.custom or {}), "task": "answer", "thinking": False}
            writer.write(prepare_cut(cut, tokenizer))


if __name__ == "__main__":
    main()
