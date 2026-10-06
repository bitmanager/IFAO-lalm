"""Prepare linked ASR/answer views from explicit original labels; no teacher calls."""
import argparse
import copy
from pathlib import Path

from lhotse import CutSet, MonoCut
from lalm_core.model import LALMProcessor
from prepare_conversation import prepare_cut, prepare_asr_cut, _render_conversation, _estimate_text_tokens
from lalm_core.paired_tasks import task_views


def replace_mixed_answer(answer_view, foreground, original_answer, tokenizer):
    """Bridge an existing mixture to an exact original foreground answer.

    ``foreground`` is the immutable cut saved by the mixer's source provenance;
    ``original_answer`` is the separately verified original-script answer cut.
    The current user instruction/policy and all history remain unchanged.
    """
    if answer_view.custom.get("task") != "answer":
        raise ValueError("Expected an existing answer view")
    if not (answer_view.foreground_source_cut_id == foreground.id == original_answer.id):
        raise ValueError("Foreground source IDs differ")
    for key in ("recording", "start", "duration", "channel"):
        if foreground.to_dict()[key] != original_answer.to_dict()[key]:
            raise ValueError(f"Original foreground audio differs: {key}")
    if not (foreground.supervisions[0].text == original_answer.supervisions[0].text
            == answer_view.supervisions[0].text):
        raise ValueError("Original foreground transcript differs")
    for key in ("history", "system", "split"):
        if not (foreground.custom.get(key) == original_answer.custom.get(key)
                == answer_view.custom.get(key)):
            raise ValueError(f"Original foreground {key} differs")
    target = (original_answer.supervisions[0].custom or {}).get("answer")
    if not isinstance(target, str) or not target.strip() or "<|" in target:
        raise ValueError("Missing explicit original answer")
    if original_answer.conversation[-1] != {"role": "assistant", "content": target}:
        raise ValueError("Original answer and prepared target differ")
    if answer_view.conversation[:-2] != original_answer.conversation[:-2]:
        raise ValueError("Prepared system/history differs")
    result = copy.deepcopy(answer_view)
    result.supervisions[0].custom["answer"] = target
    result.custom["conversation"][-1] = {"role": "assistant", "content": target}
    result.custom["rendered_conversation"] = _render_conversation(result.conversation, tokenizer)
    result.custom["num_text_tokens"] = _estimate_text_tokens(result.rendered_conversation, tokenizer)
    return result


def pack_existing_task_views(asr, answer, tokenizer):
    """Pack prepared corresponding views, preserving their policy and targets."""
    for cut, task in ((asr, "asr"), (answer, "answer")):
        if not isinstance(cut, MonoCut) or len(cut.supervisions) != 1:
            raise ValueError("Expected one-supervision MonoCut views")
        if cut.custom.get("task") != task or cut.custom.get("task_views"):
            raise ValueError("Expected separate existing ASR and answer views")
        if not cut.id.endswith("-" + task):
            raise ValueError("Expected native task-suffixed IDs")
        target = (cut.supervisions[0].custom or {}).get("answer")
        if not isinstance(target, str) or not target.strip() or "<|" in target:
            raise ValueError("Missing explicit view target")
        if cut.conversation[-1] != {"role": "assistant", "content": target}:
            raise ValueError("Prepared view target differs")
        if _render_conversation(cut.conversation, tokenizer) != cut.rendered_conversation:
            raise ValueError("Prepared rendered conversation differs")
        if cut.num_text_tokens < _estimate_text_tokens(cut.rendered_conversation, tokenizer):
            raise ValueError("Stored token budget underestimates prepared view")
    if asr.id[:-4] != answer.id[:-7]:
        raise ValueError("Task view IDs are not a corresponding pair")
    # Compare every other field, including audio, history, split and mixture QC.
    def common(cut):
        value = copy.deepcopy(cut.to_dict())
        value.pop("id")
        for key in ("task", "conversation", "rendered_conversation", "num_text_tokens", "num_tokens"):
            value["custom"].pop(key, None)
        value["supervisions"][0].get("custom", {}).pop("answer", None)
        return value
    if common(asr) != common(answer):
        raise ValueError("Task views differ in audio/history/source metadata")
    if asr.conversation[:-2] != answer.conversation[:-2]:
        raise ValueError("Prepared task history differs")
    asr_user, answer_user = asr.conversation[-2], answer.conversation[-2]
    if asr_user["role"] != "user" or answer_user["role"] != "user":
        raise ValueError("Expected native current user turns")
    a, b = asr_user["content"], answer_user["content"]
    instruction = "Дословно расшифруй текущую аудиозапись. Выведи только её текст."
    if not (len(a) == 2 and len(b) in (1, 2) and a[0] == b[0]
            and a[0].get("type") == "audio"):
        raise ValueError("Current audio/policy structure differs")
    expected = instruction + ("\n" + b[1]["text"] if len(b) == 2 else "")
    if a[1] != {"type": "text", "text": expected}:
        raise ValueError("ASR and answer foreground policies differ")
    unit = copy.deepcopy(answer)
    unit.id = answer.id[:-7]
    for key in ("task", "conversation", "rendered_conversation", "num_tokens"):
        unit.custom.pop(key, None)
    unit.custom["task_views"] = [dict(task=task,
        target=cut.supervisions[0].custom["answer"], **{k: copy.deepcopy(cut.custom[k])
        for k in ("conversation", "rendered_conversation", "num_text_tokens")})
        for cut, task in ((asr, "asr"), (answer, "answer"))]
    unit.custom["num_text_tokens"] = sum(v["num_text_tokens"] for v in unit.task_views)
    task_views(unit)
    return unit


def prepare_paired_cut(source, tokenizer, system=None):
    if not isinstance(source, MonoCut) or len(source.supervisions) != 1:
        raise ValueError(f"{source.id}: expected one current-utterance MonoCut")
    if (source.custom or {}).get("task") or (source.custom or {}).get("task_views"):
        raise ValueError(f"{source.id}: supply base cuts before task adapters")
    cut = copy.deepcopy(source)
    cut.custom = cut.custom or {}
    sup = cut.supervisions[0]
    if abs(sup.start) > .001 or abs(sup.duration-cut.duration) > .001:
        raise ValueError(f"{cut.id}: labels must cover the current audio cut")
    transcript, answer = sup.text, (sup.custom or {}).get("answer")
    prepared = []
    for task, target in (("asr", transcript), ("answer", answer)):
        if target is None or (isinstance(target, str) and not target.strip()):
            continue  # Missing labels are absent tasks, never transcript-as-answer.
        if not isinstance(target, str) or "<|" in target:
            raise ValueError(f"{cut.id}: expected a plain {task} target")
        if system:
            if cut.custom.get("system", system) != system:
                raise ValueError(f"{cut.id}: system mismatch")
            cut.custom["system"] = system
        view = prepare_asr_cut(cut, tokenizer) if task == "asr" else prepare_cut(copy.deepcopy(cut), tokenizer)
        prepared.append(dict(task=task, target=target, **{k: view.custom[k] for k in (
            "conversation", "rendered_conversation", "num_text_tokens")}))
    if not prepared:
        raise ValueError(f"{cut.id}: neither transcript nor answer is labelled")
    cut.custom["task_views"] = prepared
    cut.custom["num_text_tokens"] = sum(v["num_text_tokens"] for v in prepared)
    return cut


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--model-dir", required=True, help="Native HF export with its audio token")
    parser.add_argument("--system-file", type=Path)
    args = parser.parse_args()
    processor = LALMProcessor.from_pretrained(args.model_dir)
    system = args.system_file.read_text().strip() if args.system_file else None
    seen = set()
    with CutSet.open_writer(args.output_manifest, overwrite=False) as writer:
        for cut in CutSet.from_file(args.input_manifest):
            if cut.id in seen:
                raise ValueError(f"Duplicate source ID: {cut.id}")
            seen.add(cut.id)
            writer.write(prepare_paired_cut(cut, processor.tokenizer, system))


if __name__ == "__main__":
    main()
