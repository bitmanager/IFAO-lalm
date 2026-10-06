"""Fixed heldout history-flip diagnostic; reuse 36 immutable mixture WAVs."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from lhotse import CutSet, MonoCut
from lhotse.dataset.input_strategies import AudioSamples

from evaluate_qa import _strip_last_assistant
from lalm_core.data_module import LALMDataset
from lalm_core.model.processing_lalm import LALMProcessor
from prepare_context_overlap import task_views

PAIR_IDS = tuple(f"ctxmix-validation-{n:04d}" for n in (162, 163, 164, 165, 166, 168))
QC_PAIR_ID = "ctxmix-validation-0165"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor", required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.source.resolve()):
        raise ValueError("Diagnostic output must be separate from the frozen source")
    tracked = [args.source / p for p in (
        "validation.jsonl.gz", "source/recipe.jsonl",
        "source/original-cuts.jsonl", "mixed/pilot.jsonl")]
    before = {str(p): digest(p) for p in tracked}
    recipe = {r["id"]: r for r in rows(args.source / "source/recipe.jsonl")}
    sources = {(r["pair_id"], r["role"]): MonoCut.from_dict(copy.deepcopy(r["cut"]))
               for r in rows(args.source / "source/original-cuts.jsonl")}
    examples = [r for r in rows(args.source / "mixed/pilot.jsonl")
                if r["source_group_id"] in PAIR_IDS and r["condition"] == "mixture"]
    if len(examples) != 36:
        raise ValueError("Expected six existing mixtures per predeclared pair")
    masks = {c.mixture_pair_id: c.clean_source_qc
             for c in CutSet.from_file(args.source / "validation.jsonl.gz")}
    if not masks[QC_PAIR_ID]["clean_asr_exact_both"]:
        raise ValueError("Predeclared clean-source stratum changed")
    processor = LALMProcessor.from_pretrained(args.processor)
    cuts, index = [], []
    audio_hashes = {}
    for pair_id in PAIR_IDS:
        if recipe[pair_id]["split"] != "validation":
            raise ValueError("Diagnostic sources must retain their frozen heldout split")
        a, b = [sources[(pair_id, role)] for role in ("foreground", "background")]
        if not a.history or not b.history or a.history == b.history or a.system != b.system:
            raise ValueError("Both distinct original histories and the same system are required")
        pair_examples = [e for e in examples if e["source_group_id"] == pair_id]
        if {(e["onset_order"], e["snr_db_requested"]) for e in pair_examples} != {
                (order, snr) for order in ("foreground-first", "background-first") for snr in (3, 0, -3)}:
            raise ValueError("Unexpected existing mixture conditions")
        for example in pair_examples:
            path = example["audio_path"]
            audio_hashes[path] = digest(path)
            views = []
            for cue, role, source in (("A", "foreground", a), ("B", "background", b)):
                # Existing adapter delegates ASR labels and rendering to prepare_asr_cut.
                cut = task_views(source, example, processor.tokenizer)[1]
                cut.id += "-history-" + cue
                cut.custom.update(
                    diagnostic="same_waveform_history_flip", history_cue=cue,
                    target_original_role=role, original_mixture_id=example["id"],
                    original_foreground_source_cut_id=a.id,
                    original_background_source_cut_id=b.id,
                    original_onset_order=example["onset_order"],
                    original_foreground_snr_db=example["snr_db_requested"],
                    clean_source_qc=copy.deepcopy(masks[pair_id]),
                    training_eligible=False, main_training_connected=False,
                )
                if cue == "B":
                    cut.custom["onset_order"] = (
                        "background-first" if example["onset_order"] == "foreground-first"
                        else "foreground-first")
                    cut.custom["snr_db"] = -example["snr_db_requested"]
                assert cut.history == source.history and cut.system == source.system
                assert cut.conversation[-1]["content"] == source.supervisions[0].text
                assert cut.supervisions[0].custom["answer"] == source.supervisions[0].text
                prompt = _strip_last_assistant(cut.rendered_conversation)
                assert source.supervisions[0].text not in prompt
                assert cut.task == "asr" and cut.custom["split"] == "validation"
                views.append(cut)
                cuts.append(cut)
            assert views[0].recording.sources == views[1].recording.sources
            assert views[0].conversation[-2] == views[1].conversation[-2]
            assert _strip_last_assistant(views[0].rendered_conversation) != _strip_last_assistant(views[1].rendered_conversation)
            audio = views[0].load_audio()
            assert audio.shape == (1, views[0].num_samples) and np.isfinite(audio).all()
            index.append({"pair_id": pair_id, "mixture_id": example["id"],
                          "audio_path": path, "audio_sha256": audio_hashes[path],
                          "A_cut_id": views[0].id, "B_cut_id": views[1].id,
                          "A_source_id": a.id, "B_source_id": b.id,
                          "A_reference": a.supervisions[0].text,
                          "B_reference": b.supervisions[0].text,
                          "original_onset_order": example["onset_order"],
                          "original_foreground_snr_db": example["snr_db_requested"]})
    assert len(cuts) == len({c.id for c in cuts}) == 72
    assert len(audio_hashes) == 36
    dataset = LALMDataset(AudioSamples(fault_tolerant=False), processor, return_cuts=True)
    for start in range(0, len(cuts), 12):
        batch = dataset[CutSet.from_cuts(cuts[start:start + 12])]
        assert batch["asr_mask"].all()
        for i, cut in enumerate(batch["cuts"]):
            labels = batch["labels"][i]
            text = processor.tokenizer.decode(labels[labels != -100])
            assert text == cut.conversation[-1]["content"] + "<|im_end|>"
    assert {str(p): digest(p) for p in tracked} == before
    assert all(digest(p) == sha for p, sha in audio_hashes.items())
    args.output.mkdir(parents=True, exist_ok=False)
    all_cuts = CutSet.from_cuts(cuts)
    all_cuts.to_file(args.output / "all6.jsonl.gz")
    qc = all_cuts.filter(lambda c: c.mixture_pair_id == QC_PAIR_ID).to_eager()
    assert len(qc) == 12
    qc.to_file(args.output / "qc0165.jsonl.gz")
    for name in ("all6", "qc0165"):
        (args.output / ("eval-" + name + ".yaml")).write_text(yaml.safe_dump([
            {"name": "context-history-flip-" + name,
             "manifest": str(args.output / (name + ".jsonl.gz"))}]))
    (args.output / "paired-audio-index.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in index))
    summary = {
        "status": "CPU_QA_passed_no_model_inference", "cuts": 72, "audio_files": 36,
        "pairs": list(PAIR_IDS), "history_cue_counts": dict(Counter(c.history_cue for c in cuts)),
        "qc_stratum": {"pair_id": QC_PAIR_ID, "cuts": 12,
                       "selection": "Predeclared independent clean-source QC, not predictions"},
        "source_hashes_before_and_after": before,
        "manifest_sha256": {name: digest(args.output / (name + ".jsonl.gz"))
                            for name in ("all6", "qc0165")},
        "all_72_prompt_labels_history_EOS_checked": True,
        "same_audio_bytes_for_A_and_B": True, "primary_validation_unchanged": True,
        "target_text_in_inference_prompt": False, "new_audio_written": False,
        "new_history_or_targets": False, "no_gpu_or_model_loaded": True,
        "training_eligible": False, "main_training_connected": False,
        "onset_and_snr_metadata": "Target-relative; original mixture values retained separately",
        "clean_source_qc_role_basis": "QC retains original foreground A/background B roles. For history B, target error count is background_word_errors, not foreground_word_errors.",
        "limitations": "Six pairs reuse seven source utterances; only pair 0165 has exact independent clean QC for both sources. Text history is a semantic cue, not acoustic enrollment.",
    }
    (args.output / "readiness.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
