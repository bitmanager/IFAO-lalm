"""Adapt fixed existing contextual cuts through the existing Lhotse overlap mixer.

No TTS, teacher inference, trainer, decoder or model changes. Both task views
preserve foreground history; only the final assistant target differs by task.
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import yaml
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from transformers import AutoTokenizer

from prepare_conversation import prepare_asr_cut, prepare_cut
from remix_foreground_pilot import digest, read_jsonl, save_json


FIRST_VOICE_POLICY = (
    "В текущей аудиозаписи пользователь — собеседник, начавший говорить первым. "
    "Учитывай только его речь; одновременная посторонняя речь на фоне не является его репликой."
)


def validate_pairs(recipe, cuts):
    seen, split_keys = set(), {}
    for row in recipe:
        if row["id"] in seen or Path(row["id"]).name != row["id"]:
            raise ValueError("Duplicate or unsafe pair ID")
        seen.add(row["id"])
        split = row["split"]
        if split not in ("train", "validation"):
            raise ValueError("Preserve the existing train/validation split")
        if row["foreground_voice"] == row["background_voice"]:
            raise ValueError("Pair requires distinct voice IDs")
        if row["foreground_topic"] == row["background_topic"]:
            raise ValueError("Pair requires distinct reviewed topics")
        groups = []
        for role in ("foreground", "background"):
            cut = cuts[row[role + "_cut_id"]]
            if not isinstance(cut, MonoCut) or len(cut.supervisions) != 1:
                raise ValueError("Expected one isolated native foreground/background cut")
            sup = cut.supervisions[0]
            if abs(sup.start) > .001 or abs(sup.duration - cut.duration) > .001:
                raise ValueError("Transcript must cover the complete source cut")
            if sup.language not in ("ru", "Russian") or not sup.text or "<|" in sup.text:
                raise ValueError("Expected a plain Russian source transcript")
            if cut.custom.get("split") != split or not cut.custom.get("source_group_id"):
                raise ValueError("Source split/group differs from recipe")
            group = cut.custom["source_group_id"]
            if row.get(role + "_source_group", group) != group:
                raise ValueError("Declared source group differs from the original cut")
            metadata_path = row.get(role + "_voice_metadata_path")
            if metadata_path:
                if digest(metadata_path) != row[role + "_voice_metadata_sha256"]:
                    raise ValueError("Voice provenance metadata changed")
                metadata = json.loads(Path(metadata_path).read_text())
                participant = next(k for k, v in metadata["speaker_to_channel"].items() if v == cut.channel)
                if metadata["voice"][participant] != row[role + "_voice"]:
                    raise ValueError("Voice ID differs from the original render metadata")
            groups.append(group)
            for key in (("source_group", group), ("recording", cut.recording.id),
                        ("text", sup.text.casefold())):
                if split_keys.setdefault(key, split) != split:
                    raise ValueError("Source group, recording or text crosses splits")
            if role == "foreground":
                if not cut.custom.get("history") or not (sup.custom or {}).get("answer"):
                    raise ValueError("Foreground requires existing history and teacher answer")
                if cut.custom.get("task", "answer") != "answer":
                    raise ValueError("Foreground must be an existing answer task")
        if groups[0] == groups[1]:
            raise ValueError("Foreground and background share a source group")
    if not seen:
        raise ValueError("Empty pair recipe")


def task_views(foreground, example, tokenizer):
    recording = Recording.from_file(example["audio_path"], recording_id=example["id"])
    if recording.num_channels != 1 or abs(recording.duration - example["duration"]) > .001:
        raise ValueError("Mixed audio differs from the existing mixer manifest")
    cut = recording.to_cut()
    source_sup = foreground.supervisions[0]
    cut.supervisions = [SupervisionSegment(
        id=cut.id, recording_id=recording.id, start=0, duration=cut.duration,
        channel=0, text=source_sup.text, language=source_sup.language,
        custom=copy.deepcopy(source_sup.custom),
    )]
    cut.custom = {k: copy.deepcopy(v) for k, v in foreground.custom.items()
                  if k not in ("conversation", "rendered_conversation", "num_text_tokens", "task")}
    cut.custom.update(
        condition=example["condition"], mixture_pair_id=example["source_group_id"],
        target_policy="first_active_voice" if example["condition"] == "mixture" else "ordinary_clean",
        snr_db=example.get("snr_db_requested"), training_eligible=False,
        main_training_connected=False, foreground_source_cut_id=foreground.id,
    )
    policy = FIRST_VOICE_POLICY if example["condition"] == "mixture" else None
    answer = copy.deepcopy(cut)
    answer.id += "-answer"
    answer.custom["task"] = "answer"
    answer = prepare_cut(answer, tokenizer, instruction=policy)
    asr = prepare_asr_cut(cut, tokenizer)
    if policy:
        # Retain the exact existing ASR task instruction, then qualify whose
        # utterance it refers to. The common policy itself never asks for ASR.
        original = asr.conversation[-2]["content"][1]["text"]
        asr = prepare_cut(asr, tokenizer, instruction=original + "\n" + policy)
    return answer, asr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    cuts = {}
    for manifest in args.source_manifests:
        for cut in CutSet.from_file(manifest):
            if cut.id in cuts:
                raise ValueError("Duplicate source cut ID")
            cuts[cut.id] = cut
    recipe = read_jsonl(args.recipe)
    validate_pairs(recipe, cuts)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    source_root = args.output / "source"
    source_root.mkdir()
    (source_root / "audio").mkdir()
    staged, mixer_recipe, originals = [], [], []
    source_hash_splits = {}
    for row in recipe:
        mix_row = {**row, "category": "unrelated_context_topics"}
        for role in ("foreground", "background"):
            cut = cuts[row[role + "_cut_id"]]
            path = source_root / "audio" / (row["id"] + "-" + role + ".wav")
            samples = cut.resample(16000).load_audio()
            if samples.shape[0] != 1:
                raise ValueError("Expected the selected source channel only")
            sf.write(path, samples[0], 16000, subtype="FLOAT")
            sha = digest(path)
            # FLOAT WAV headers can carry varying PEAK-chunk timestamps.
            # Split deduplication must compare samples, not container bytes.
            pcm_sha = hashlib.sha256(samples.astype("<f4", copy=False).tobytes()).hexdigest()
            if source_hash_splits.setdefault(pcm_sha, row["split"]) != row["split"]:
                raise ValueError("Identical source waveform crosses splits")
            text = cut.supervisions[0].text
            mix_row[role + "_text"] = text
            staged.append({"pair_id": row["id"], "role": role, "split": row["split"],
                "path": str(path), "wav_sha256": sha, "pcm_float32_le_sha256": pcm_sha,
                "duration": samples.shape[1] / 16000,
                "text": text, "voice": row[role + "_voice"], "source_group_id": cut.source_group_id,
                "source_cut_id": cut.id, "source_start": cut.start, "source_channel": cut.channel})
            originals.append({"pair_id": row["id"], "role": role, "cut": cut.to_dict()})
        mixer_recipe.append(mix_row)
    for name, rows in [("sources.jsonl", staged), ("recipe.jsonl", mixer_recipe),
                       ("original-cuts.jsonl", originals)]:
        (source_root / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    save_json(source_root / "provenance.json", {
        "source_manifests": {str(p): digest(p) for p in args.source_manifests},
        "recipe_sha256": digest(args.recipe), "adapter_sha256": digest(__file__),
        "voice_bank_revision": None, "source_kind": "Existing native contextual audio cuts",
        "label_quality": "Foreground transcript/history/teacher answer preserved; no new teacher labels; not human acoustic gold",
        "new_tts_requests": 0, "speaker_disjoint": False, "training_eligible": False,
    })
    mixed_root = args.output / "mixed"
    subprocess.run([sys.executable, str(Path(__file__).with_name("remix_foreground_pilot.py")),
        "--source-pilot", str(source_root), "--output", str(mixed_root),
        "--snr-db", "3", "0", "-3", "--split-by-source-group"], check=True)
    lookup = {r["id"]: cuts[r["foreground_cut_id"]] for r in recipe}
    groups = defaultdict(list)
    for example in read_jsonl(mixed_root / "pilot.jsonl"):
        foreground = lookup[example["source_group_id"]]
        for cut in task_views(foreground, example, tokenizer):
            groups[cut.custom["split"]].append(cut)
    for split, rows in groups.items():
        CutSet.from_cuts(rows).to_file(args.output / (split + ".jsonl.gz"))
    for task in ("asr", "answer"):
        config = []
        for condition in ("clean", "snr+3", "snr+0", "snr-3"):
            selected = [c for c in groups["validation"] if c.task == task and
                ("clean" if c.condition == "clean" else f"snr{c.snr_db:+g}") == condition]
            name = f"context-overlap-{condition}-{task}"
            path = args.output / (name + ".jsonl.gz")
            CutSet.from_cuts(selected).to_file(path)
            config.append({"name": name, "manifest": str(path)})
        (args.output / ("eval-" + task + ".yaml")).write_text(yaml.safe_dump(config))
    save_json(args.output / "summary.json", {
        "pairs": len(recipe), "source_fg_seconds": sum(c.duration for c in {c.id: c for c in lookup.values()}.values()),
        "split_examples": {s: len(rs) for s, rs in groups.items()},
        "task_examples": sum(len(rs) for rs in groups.values()),
        "task_exposure_hours": sum(c.duration for rs in groups.values() for c in rs) / 3600,
        "training_eligible": False, "main_training_connected": False,
        "speaker_disjoint": False, "source_group_disjoint": True,
        "new_tts_requests": 0, "new_teacher_answers": 0, "gpu_inference": "not run",
    })


if __name__ == "__main__":
    main()
