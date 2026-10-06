"""Export original synthetic script answers in the stock teacher-response format.

Select already admitted train user cuts only. No audio rendering, generation,
new alignment, role inference or admission of additional source utterances.
Use prepare_conversation.py --responses on the exported source manifest.
"""

import argparse
import hashlib
import json
from collections import Counter
from itertools import groupby
from pathlib import Path

from lhotse import CutSet

from prepare_conversation import teacher_messages
from prepare_multitalk_context import dialogue_cuts


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_keys(cut):
    return {("group", cut.custom["source_group_id"]),
            ("recording", cut.recording.id),
            *(('audio_path', s.source) for s in cut.recording.sources)}


def original_answer(cut, metadata_path, metadata, native_cuts):
    """Join the existing role-group index; validate it against native conversion."""
    if cut.id not in native_cuts:
        raise ValueError(f"{cut.id}: missing original native user cut")
    original = native_cuts[cut.id]
    if (cut.channel != original.channel or abs(cut.start - original.start) > .001
            or abs(cut.duration - original.duration) > .001
            or cut.supervisions[0].text != original.supervisions[0].text
            or any(cut.custom[k] != original.custom[k]
                   for k in ("history", "system", "source_group_id", "split"))):
        raise ValueError(f"{cut.id}: source transcript, bounds, channel or history changed")
    roles = {p["name"]: p["role"] for p in metadata["source_script"]["participants"]}
    groups = [(role, list(turns)) for role, turns in groupby(
        metadata["segments"], key=lambda segment: roles[segment["participant"]])]
    scripted = [(role, " ".join(t["text"].strip() for t in turns))
                for role, turns in groupby(metadata["source_script"]["dialogue"],
                                           key=lambda turn: roles[turn["speaker"]])]
    if scripted != [(role, " ".join(t["text"].strip() for t in turns)) for role, turns in groups]:
        raise ValueError(f"{cut.id}: rendered role/text groups differ from the original script")
    if any(t["channel"] != metadata["speaker_to_channel"][t["participant"]]
           for t in metadata["segments"]):
        raise ValueError(f"{cut.id}: rendered channel map differs from original participants")
    index = int(cut.id.rsplit("-turn-", 1)[1])
    if groups[index][0] != "user" or index + 1 >= len(groups) or groups[index + 1][0] != "assistant":
        raise ValueError(f"{cut.id}: missing unambiguous following assistant group")
    following = groups[index + 1][1]
    answer = " ".join(turn["text"].strip() for turn in following)
    start, end = following[0]["start"], max(t["end"] for t in following)
    if not answer or "<|" in answer or not cut.start <= start < end <= cut.recording.duration + .001:
        raise ValueError(f"{cut.id}: invalid following assistant text or bounds")
    if end <= cut.end or any(t["channel"] == cut.channel for t in following):
        raise ValueError(f"{cut.id}: following assistant does not follow on its own channel")
    provenance = {
        "quality": "original_script_synthetic_not_gold",
        "metadata_path": str(metadata_path), "metadata_sha256": sha256(metadata_path),
        "source_group_id": cut.source_group_id, "user_role_group_index": index,
        "answer_role_group_index": index + 1, "answer_start": start, "answer_end": end,
        "answer_channels": sorted({t["channel"] for t in following}),
        "answer_onset_overlap_seconds": max(0., cut.end - start),
        "prior_history_unchanged": True, "generation_performed": False,
    }
    return {"idx": cut.id, "messages": teacher_messages(cut), "response": answer}, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cuts", type=int, required=True)
    args = parser.parse_args()
    heldout = set()
    heldout_ids = set()
    for path in args.heldout_manifests:
        for cut in CutSet.from_file(path):
            heldout_ids.add(cut.id)
            # Some ASR-only heldout cuts have no conversational group.
            if (cut.custom or {}).get("source_group_id"):
                heldout.update(source_keys(cut))
            else:
                heldout.add(("recording", cut.recording.id))
                heldout.update(("audio_path", s.source) for s in cut.recording.sources)
    selected = list(CutSet.from_file(args.input_manifest).filter(
        lambda c: (c.custom or {}).get("boundary_source") == "synthetic_rendered_turn"))
    if len(selected) != args.expected_cuts or len({c.id for c in selected}) != len(selected):
        raise ValueError("Unexpected or duplicate admitted synthetic source count")
    cache, rows, counts = {}, [], Counter()
    for cut in selected:
        if (cut.custom.get("split") != "train" or cut.custom.get("source_call_id")
                or cut.custom.get("task", "answer") != "answer" or len(cut.supervisions) != 1
                or Path(cut.id).name != cut.id):
            raise ValueError(f"{cut.id}: expected admitted train-only synthetic answer cut")
        if cut.id in heldout_ids or source_keys(cut) & heldout:
            raise ValueError(f"{cut.id}: source intersects supplied heldout")
        path = Path(cut.recording.sources[0].source).with_suffix(".json")
        if path not in cache:
            metadata = json.loads(path.read_text())
            native = {c.id: c for c in dialogue_cuts(path, cut.custom["system"])}
            cache[path] = metadata, native
        response, provenance = original_answer(cut, path, *cache[path])
        cut.custom["answer_provenance"] = provenance
        rows.append((cut, response, provenance))
        counts["reviewed_v2" if cut.id.startswith("reviewed-") else "older_synthetic"] += 1
    # Validate every source before creating any export artifact.
    args.output.mkdir(parents=True, exist_ok=False)
    responses = args.output / "responses"
    responses.mkdir()
    with CutSet.open_writer(args.output / "source.jsonl.gz") as writer:
        for cut, response, _ in rows:
            writer.write(cut)
            (responses / (cut.id + ".json")).write_text(
                json.dumps(response, ensure_ascii=False, indent=2) + "\n")
    (args.output / "answer-provenance.jsonl").write_text("".join(
        json.dumps({"id": cut.id, **provenance}, ensure_ascii=False) + "\n"
        for cut, _, provenance in rows))
    summary = {
        "quality": "original_script_synthetic_not_gold", "cuts": len(rows), "sources": counts,
        "source_ids": [cut.id for cut, _, _ in rows],
        "source_groups": len({cut.source_group_id for cut, _, _ in rows}),
        "unique_user_audio_hours": sum(c.duration for c, _, _ in rows) / 3600,
        "input_sha256": {str(p): sha256(p) for p in [args.input_manifest, *args.heldout_manifests]},
        "adapter_sha256": sha256(__file__), "source_manifest_sha256": sha256(args.output / "source.jsonl.gz"),
        "responses_sha256": {p.name: sha256(p) for p in sorted(responses.glob("*.json"))},
        "heldout_id_group_recording_path_intersections": 0,
        "history_system_ids_audio_splits_preserved": True, "new_audio_or_teacher_generation": False,
        "training_connected": False, "native_preparation_and_batch_qa": "pending",
        "limitation": "Preserves existing admitted groups; synthetic source text is not factual gold. Audio identity beyond source ID/path requires the existing corpus QA.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
