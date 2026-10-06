"""Prepare linked ASR/answer views from explicit original labels; no teacher calls."""
import argparse
import copy
from pathlib import Path

from lhotse import CutSet, MonoCut
from lalm_core.model import LALMProcessor
from prepare_conversation import prepare_cut, prepare_asr_cut


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
