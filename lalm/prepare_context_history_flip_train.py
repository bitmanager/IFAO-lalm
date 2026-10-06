"""Format-only TRAIN views: two original histories over each audited existing mix."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import yaml
import numpy as np
from gigaam.utils import normalize_raw_text
from lhotse import CutSet, MonoCut
from lhotse.dataset.input_strategies import AudioSamples

from evaluate_qa import _strip_last_assistant
from lalm_core.data_module import LALMDataset
from lalm_core.model.processing_lalm import LALMProcessor
from prepare_context_history_flip import digest, rows
from prepare_context_overlap import task_views

INVENTORY_SHA256 = "77546ebe32485688a3e225808c5e1ad3dccf8aeba57f395fa2090cf8b206b787"
EXPECTED = {"context-target-v1": 63, "context-target-older441-v1": 441,
            "context-target-reserve29-v1": 16}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--processor", required=True)
    args = p.parse_args()
    assert digest(args.inventory) == INVENTORY_SHA256
    audit = json.loads(args.inventory.read_text())
    assert audit["status"] == "read_only_inventory_complete"
    assert audit["checked_mixtures"] == 3120 and not audit["cuda_initialized"]
    assert not args.output.exists()
    for name in EXPECTED:
        assert not args.output.resolve().is_relative_to((args.root / name).resolve())
    tracked = dict(audit["heldout_manifest_sha256"])
    for name, data in audit["datasets"].items():
        assert name in EXPECTED
        tracked.update({str(args.root / name / f): h for f, h in data["source_hashes"].items()})
    assert all(digest(f) == h for f, h in tracked.items())
    processor = LALMProcessor.from_pretrained(args.processor)
    cuts, index, audio_hashes = [], [], {}
    for name, expected in EXPECTED.items():
        root = args.root / name
        selected = [r for r in audit["pairs"] if r["dataset"] == name and r["eligible"]]
        assert len(selected) == expected
        recipe = {r["id"]: r for r in rows(root / "source/recipe.jsonl")}
        sources = {(r["pair_id"], r["role"]): MonoCut.from_dict(copy.deepcopy(r["cut"]))
                   for r in rows(root / "source/original-cuts.jsonl")}
        examples = rows(root / "mixed/pilot.jsonl")
        audio_qc = {(r["pair_id"], r["level"]): r
                    for r in json.loads((root / "mixed/audio-qc.json").read_text())}
        masks = {c.mixture_pair_id: c.clean_source_qc for c in CutSet.from_file(root / "train.jsonl.gz")}
        for item in selected:
            pair = item["pair_id"]
            assert item["metadata_pass"] and item["both_full_sources_retained"]
            assert not item["reasons"] and not item["prompt_leaks"]
            assert recipe[pair]["split"] == masks[pair]["split"] == "train"
            assert masks[pair]["clean_asr_exact_both"]
            assert len(set(item["source_groups"])) == 2
            a, b = [sources[pair, role] for role in ("foreground", "background")]
            assert a.id == item["A_source_id"] and b.id == item["B_source_id"]
            assert a.history and b.history and a.history != b.history and a.system == b.system
            assert a.custom["split"] == b.custom["split"] == "train"
            assert masks[pair]["foreground_source_cut_id"] == a.id
            assert masks[pair]["background_source_cut_id"] == b.id
            pair_examples = [e for e in examples if e["source_group_id"] == pair and e["condition"] == "mixture"]
            assert len(pair_examples) == 6
            assert {(e["onset_order"], e["snr_db_requested"]) for e in pair_examples} == {
                (order, snr) for order in ("foreground-first", "background-first") for snr in (3, 0, -3)}
            for example in pair_examples:
                assert example["target_policy"] == "context_target"
                path = example["audio_path"]
                qc = audio_qc[pair, Path(path).parent.name]
                assert digest(path) == qc["file_sha256"]["mix"]
                audio_hashes[path] = qc["file_sha256"]["mix"]
                views = []
                for cue, role, source in (("A", "foreground", a), ("B", "background", b)):
                    cut = task_views(source, example, processor.tokenizer)[1]
                    cut.id = name + "-" + cut.id + "-history-" + cue
                    cut.custom.update(
                        diagnostic="same_waveform_original_history_train", history_cue=cue,
                        target_original_role=role, original_mixture_id=example["id"],
                        original_foreground_source_cut_id=a.id, original_background_source_cut_id=b.id,
                        original_onset_order=example["onset_order"],
                        original_foreground_snr_db=example["snr_db_requested"],
                        target_source_cut_id=source.id,
                        clean_source_qc=copy.deepcopy(masks[pair]),
                        clean_source_qc_role_basis="original foreground A / background B",
                        full_source_retention_audit_sha256=INVENTORY_SHA256,
                        audio_sha256=audio_hashes[path], split="train",
                        training_eligible=False, main_training_connected=False,
                    )
                    if cue == "B":
                        cut.custom["onset_order"] = ("background-first" if example["onset_order"] == "foreground-first"
                                                     else "foreground-first")
                        cut.custom["snr_db"] = -example["snr_db_requested"]
                    reference = source.supervisions[0].text
                    assert cut.history == source.history and cut.system == source.system
                    assert cut.supervisions[0].text == cut.supervisions[0].custom["answer"] == reference
                    assert cut.conversation[-1] == {"role": "assistant", "content": reference}
                    assert cut.rendered_conversation.endswith(reference + "<|im_end|>\n")
                    assert cut.task == "asr" and cut.custom["split"] == "train"
                    prompt = _strip_last_assistant(cut.rendered_conversation)
                    for ref in (a.supervisions[0].text, b.supervisions[0].text):
                        assert ref and ref not in prompt
                        assert normalize_raw_text(ref) not in normalize_raw_text(prompt)
                    views.append(cut)
                    cuts.append(cut)
                assert views[0].recording == views[1].recording
                assert views[0].conversation[-2] == views[1].conversation[-2]
                assert _strip_last_assistant(views[0].rendered_conversation) != _strip_last_assistant(views[1].rendered_conversation)
                index.append(dict(dataset=name, pair_id=pair, mixture_id=example["id"],
                    audio_path=path, audio_sha256=audio_hashes[path],
                    A_cut_id=views[0].id, B_cut_id=views[1].id,
                    A_source_id=a.id, B_source_id=b.id,
                    A_reference=a.supervisions[0].text, B_reference=b.supervisions[0].text))
    assert len(cuts) == len({c.id for c in cuts}) == 6240
    assert len(index) == len(audio_hashes) == 3120
    # Stock final-assistant masking for every row; audio expansion is checked
    # separately by the real-audio batch below, without decoding every WAV twice.
    lengths = []
    for start in range(0, len(cuts), 32):
        group = cuts[start:start + 32]
        expanded = processor.replace_multimodal_special_tokens(
            [c.rendered_conversation for c in group],
            iter(processor.audio_token_length_fn(c.num_samples) for c in group))
        # Tokenize already expanded text directly: passing it through processor
        # again would expand each placeholder a second time.
        text_batch = processor.tokenizer(expanded, return_tensors="pt", padding=True)
        text_batch["labels"] = processor._prepare_labels(
            text_batch["input_ids"], text_batch["attention_mask"])
        for cut, labels, mask in zip(group, text_batch["labels"], text_batch["attention_mask"]):
            assert processor.tokenizer.decode(labels[labels != -100]) == cut.supervisions[0].text + "<|im_end|>"
            assert cut.sampling_rate == 16000
            lengths.append(dict(cut_id=cut.id, total_tokens=int(mask.sum()),
                                target_tokens=int((labels != -100).sum()),
                                audio_tokens=processor.audio_token_length_fn(cut.num_samples)))
    # Eight duration-spread mixtures, each with A and B: all 16 use native loss masks.
    pairs = sorted(zip(cuts[::2], cuts[1::2]), key=lambda pair: (pair[0].duration, pair[0].id))
    sample = [c for i in range(8) for c in pairs[round(i * (len(pairs) - 1) / 7)]]
    dataset = LALMDataset(AudioSamples(fault_tolerant=False), processor, return_cuts=True)
    batch = dataset[CutSet.from_cuts(sample)]
    assert batch["asr_mask"].all() and batch["batch_size"] == 16
    for cut, labels in zip(batch["cuts"], batch["labels"]):
        assert processor.tokenizer.decode(labels[labels != -100]) == cut.supervisions[0].text + "<|im_end|>"
    assert all(digest(f) == h for f, h in tracked.items())
    assert all(digest(f) == h for f, h in audio_hashes.items())
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / "train.jsonl.gz"
    CutSet.from_cuts(cuts).to_file(manifest)
    (args.output / "train.yaml").write_text(yaml.safe_dump([
        {"name": "original-history-flip-train", "manifest": str(manifest)}]))
    (args.output / "paired-audio-index.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in index))
    (args.output / "sequence-lengths.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in lengths))
    length_stats = {key: dict(zip(("p50", "p90", "p95", "p99", "max"),
        np.percentile([r[key] for r in lengths], [50, 90, 95, 99, 100]).tolist()))
        for key in ("total_tokens", "target_tokens", "audio_tokens")}
    summary = dict(status="ready_staging_only", cuts=len(cuts), pairs=520, audio_files=len(audio_hashes),
        task_exposure_hours=sum(c.duration for c in cuts) / 3600,
        dataset_eligible_pairs=EXPECTED, history_cue_counts=dict(Counter(c.history_cue for c in cuts)),
        manifest=str(manifest), manifest_sha256=digest(manifest), inventory_sha256=INVENTORY_SHA256,
        source_and_heldout_hashes_before_and_after=tracked,
        adapter_sha256=digest(__file__), task_views_module_sha256=digest(Path(__file__).with_name("prepare_context_overlap.py")),
        all_6240_prompt_history_labels_EOS_checked=True, all_6240_native_text_loss_masks_checked=True,
        sequence_lengths=length_stats,
        over_2000_tokens=sum(r["total_tokens"] > 2000 for r in lengths),
        over_8192_tokens=sum(r["total_tokens"] > 8192 for r in lengths),
        truncation_or_length_filter=False,
        native_CPU_batch=dict(cuts=16, history_A=8, history_B=8, exact_labels_and_EOS=True),
        new_audio=False, new_history_or_targets=False, model_inference=False,
        task_view_note="B supervision over the mixture is a new view; history and transcript are existing original source fields, not generated text",
        training_eligible=False, main_training_connected=False,
        audio_retention="Prior frozen full-stem audit of all 3120 mixes, bound by source metadata and audio hashes",
        clean_source_qc_role_basis="Original foreground A/background B; B target QC is background_word_errors",
        limitations="Synthetic original histories; text is a semantic cue, not acoustic enrollment. No human listening claim.")
    (args.output / "readiness.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
