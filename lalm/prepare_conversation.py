"""Offline prepare cut.conversation for LALM training.

This script reads a Lhotse CutSet manifest, builds a conversation object for each
cut, writes it to ``cut.custom["conversation"]`` (accessible as ``cut.conversation``),
and saves a new manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lhotse import CutSet
from tqdm import tqdm
from transformers import AutoTokenizer

def _build_conversation(cut, instruction=None, system=None):
    supervision = cut.supervisions[0]
    response = (supervision.custom or {}).get("answer", supervision.text)
    if "history" in (cut.custom or {}) and "answer" not in (supervision.custom or {}):
        raise ValueError(f"Cut {cut.id}: contextual training requires a teacher answer")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"Cut {cut.id}: empty target answer")

    audio_source = cut.recording.sources[0].source

    messages = teacher_messages(cut, system)[:-1]

    user_content = [{"type": "audio", "audio": audio_source}]
    if instruction:
        user_content.append({"type": "text", "text": str(instruction)})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": response})
    return messages


def teacher_messages(cut, system=None):
    """Teacher and student share history; only the current modality differs."""
    metadata = cut.custom or {}
    recorded_system = metadata.get("system")
    if system and recorded_system and system != recorded_system:
        raise ValueError(f"Cut {cut.id}: system prompt differs from the recorded prompt")
    messages = []
    if recorded_system or system:
        messages.append({"role": "system", "content": recorded_system or system})
    for turn in metadata.get("history", []):
        if turn["role"] not in ("user", "assistant") or not isinstance(turn["content"], str):
            raise ValueError(f"Cut {cut.id}: history must contain text user/assistant turns")
        if "<|" in turn["content"]:
            raise ValueError(f"Cut {cut.id}: history contains reserved chat tokens")
        messages.append(dict(turn))
    messages.append({"role": "user", "content": cut.supervisions[0].text})
    return messages


def _render_conversation(
    conversation: list[dict], tokenizer, audio_token: str = "<|audio|>"
) -> str:
    """Render conversation to Qwen-style chat text with im tags.

    Example:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "audio"}, {"type": "text", "text": "Please transcribe."}],
            },
            {"role": "assistant", "content": "Trading has almost stalled."},
        ]

        rendered_text =
            <|im_start|>user
            <|audio|>Please transcribe.<|im_end|>
            <|im_start|>assistant
            Trading has almost stalled.<|im_end|>
    """
    chunks = []
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if isinstance(content, list):
            parts = []
            for item in content:
                item_type = item.get("type")
                if item_type == "audio":
                    parts.append(audio_token)
                elif item_type == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    raise ValueError(f"Unknown content type: {item_type!r}")
            rendered_content = "".join(parts)
        else:
            rendered_content = str(content)
        chunks.append({"role": role, "content": rendered_content})
    return tokenizer.apply_chat_template(chunks, tokenize=False, add_generation_prompt=False)


def _estimate_text_tokens(text: str, tokenizer) -> int:
    """Estimate text tokens with a real tokenizer."""
    text = text.strip()
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def prepare_cut(cut, tokenizer, instruction=None, system=None):
    if cut.custom is None:
        cut.custom = {}
    conversation = _build_conversation(cut, instruction, system)
    rendered_conversation = _render_conversation(conversation, tokenizer)
    cut.custom["conversation"] = conversation
    cut.custom["rendered_conversation"] = rendered_conversation
    cut.custom["num_text_tokens"] = _estimate_text_tokens(
        rendered_conversation,
        tokenizer,
    )
    return cut


def main():
    parser = argparse.ArgumentParser(
        description="Prepare cut.conversation offline for a CutSet."
    )
    parser.add_argument(
        "--input_manifest",
        required=True,
        help="Input CutSet manifest path (e.g. *.jsonl.gz).",
    )
    parser.add_argument(
        "--output_manifest",
        required=True,
        help="Output CutSet manifest path.",
    )
    parser.add_argument(
        "--instruction",
        help="Additional user prompt",
    )
    parser.add_argument(
        "--system",
        help="System for the conversation.",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name or path used to estimate num_text_tokens.",
    )
    parser.add_argument("--teacher-input", help="Export contextual messages for the upstream teacher generator")
    parser.add_argument("--responses", type=Path, help="Directory produced by the upstream teacher generator")
    args = parser.parse_args()

    out_path = Path(args.output_manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    cuts = CutSet.from_file(args.input_manifest)
    if args.teacher_input or args.responses:
        if args.instruction:
            raise ValueError("Contextual distillation does not accept a student-only instruction")
    if args.teacher_input:
        with open(args.teacher_input, "x", encoding="utf-8") as stream:
            for cut in cuts:
                stream.write(json.dumps({"idx": cut.id, "messages": teacher_messages(cut, args.system)}, ensure_ascii=False) + "\n")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with CutSet.open_writer(out_path, overwrite=False) as writer:
        for cut in tqdm(cuts, desc="Preparing conversations"):
            if args.responses:
                result = json.loads((args.responses / f"{cut.id}.json").read_text())
                if result["messages"] != teacher_messages(cut, args.system):
                    raise ValueError(f"Cut {cut.id}: teacher history/system/transcript mismatch")
                answer = result["response"]
                if not answer.strip() or "<|" in answer:
                    raise ValueError(f"Cut {cut.id}: empty, truncated or control-token teacher response")
                cut.supervisions[0].custom = {**(cut.supervisions[0].custom or {}), "answer": answer}
            writer.write(prepare_cut(cut, tokenizer, instruction=args.instruction, system=args.system))
    print(f"Saved prepared CutSet to: {out_path}")


if __name__ == "__main__":
    main()
