"""Adapt official OpenSTT phone training CSV/audio/text to the existing ASR task."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from lhotse import CutSet, Recording, SupervisionSegment
from transformers import AutoTokenizer

from prepare_asr_manifest import asr_cut


def source_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe OpenSTT source: {relative}")
    return path


def convert(manifest, audio_root, output, tokenizer, system, excluded_ids=()):
    counts, seen, seconds = Counter(), {}, 0.0
    with open(manifest, newline="") as stream, CutSet.open_writer(output, overwrite=False) as writer:
        for audio_rel, text_rel, duration in csv.reader(stream):
            # Validation repackaged as a 'train' split must never enter this adapter.
            subset = Path(audio_rel).parts[0]
            if subset not in ("asr_public_phone_calls_1", "asr_public_phone_calls_2"):
                raise ValueError(f"Not an official phone training subset: {subset}")
            if Path(text_rel).with_suffix("") != Path(audio_rel).with_suffix(""):
                raise ValueError(f"Unpaired audio/transcript: {audio_rel}")
            key = Path(audio_rel).with_suffix("").as_posix().replace("/", "-")
            if key in seen:
                if seen[key] != (text_rel, duration):
                    raise ValueError(f"Conflicting duplicate recording: {key}")
                counts["duplicate"] += 1
                continue
            seen[key] = (text_rel, duration)
            if Path(audio_rel).stem in excluded_ids:
                counts["heldout_audio"] += 1
                continue
            if not 0.5 <= float(duration) <= 30:
                counts["duration"] += 1
                continue
            audio = source_path(audio_root, audio_rel)
            text = source_path(audio_root, text_rel).read_text().strip()
            if not text or not re.search("[А-Яа-яЁё]", text) or "<|" in text:
                counts["invalid_text"] += 1
                continue
            recording = Recording.from_file(audio, recording_id=key)
            if recording.num_channels != 1:
                raise ValueError(f"Expected mono audio: {audio}")
            if abs(recording.duration - float(duration)) > 0.1:
                raise ValueError(f"Duration disagrees with official CSV: {audio}")
            if not 0.5 <= recording.duration <= 30:
                counts["duration"] += 1
                continue
            cut = recording.to_cut()
            cut.supervisions = [SupervisionSegment(
                id=key, recording_id=key, start=0, duration=recording.duration,
                channel=0, text=text, language="ru",
            )]
            cut.custom = {"source_dataset": subset, "label_source": "OpenSTT automatic ASR"}
            writer.write(asr_cut(cut, tokenizer, system))
            counts["accepted"] += 1
            seconds += cut.duration
            if counts["accepted"] % 10000 == 0:
                print(json.dumps({"counts": counts, "hours": seconds / 3600}), flush=True)
    if not counts["accepted"]:
        raise ValueError("No admitted phone training audio")
    return {"counts": counts, "hours": seconds / 3600, "history": False,
            "labels": "automatic ASR, not human gold", "manifest": str(manifest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[])
    args = parser.parse_args()
    excluded = set()
    for manifest in args.exclude_csv:
        with manifest.open() as stream:
            excluded.update(Path(row[0]).stem for row in csv.reader(stream))
    summary = convert(args.manifest, args.audio_root, args.output,
                      AutoTokenizer.from_pretrained(args.tokenizer),
                      args.system_file.read_text().strip(), excluded)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
