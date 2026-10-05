"""Adapt complete two-channel IVR calls to contextual Lhotse cuts.

Keep the original train/dev split and speaker-relative roles. History contains
only whole turns completed before the current turn starts. RNNT boundaries
are approximate ASR emission times, not production VAD/commit events.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from lhotse import CutSet, MonoCut, Recording, SupervisionSegment


def plain_text(value):
    if not isinstance(value, str) or not value.strip() or "<|" in value:
        raise ValueError("Expected nonempty plain text without reserved chat tokens")
    return value.strip()


def validate_manifest(rows):
    seen = set()
    for row in rows:
        if row["language"] != "ru":
            raise ValueError("Expected Russian calls")
        split = row["split"]
        if split not in ("train", "dev", "test", "control") or row.get("original_split", split) != split:
            raise ValueError("Expected original train/dev/test/control splits, not remapped splits")
        for field, value in [("call_id", row["call_id"]),
                             ("source_sha256", row["provenance"]["source_sha256"]),
                             ("audio_sha256", row["provenance"]["audio_sha256"])]:
            pattern = r"[A-Za-z0-9][A-Za-z0-9_-]*" if field == "call_id" else r"[0-9a-f]{64}"
            if not isinstance(value, str) or not re.fullmatch(pattern, value):
                raise ValueError(f"Invalid {field}")
            if (field, value) in seen:
                raise ValueError(f"Duplicate {field}: source calls must not repeat within or across splits")
            seen.add((field, value))


def call_cuts(row, audio_root, system):
    if row["split"] not in ("train", "dev"):
        raise ValueError("Held-out test/control calls must not be exported")
    path = audio_root / Path(row["audio_path"]).name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != row["provenance"]["audio_sha256"]:
        raise ValueError(f"{row['call_id']}: audio SHA256 mismatch")
    recording = Recording.from_file(path.resolve(), recording_id=row["call_id"])
    duration = float(row["duration_seconds"])
    permutation = row["provenance"]["speaker_channel_permutation"]
    if recording.channel_ids != [0, 1] or sorted(permutation) != [1, 2]:
        raise ValueError(f"{row['call_id']}: expected stereo audio and a two-speaker permutation")
    if not math.isfinite(duration) or abs(duration - recording.duration) > 0.001:
        raise ValueError(f"{row['call_id']}: recording duration differs from manifest")
    channels = {f"speaker_{speaker}": channel for channel, speaker in enumerate(permutation)}
    turns, ids = row["turns"], set()
    if not turns or turns != sorted(turns, key=lambda turn: turn["start"]):
        raise ValueError(f"{row['call_id']}: missing or unordered turns")
    for turn in turns:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", turn["id"]) or turn["id"] in ids:
            raise ValueError(f"{row['call_id']}: invalid or duplicate turn id")
        ids.add(turn["id"])
        plain_text(turn["text"])
        if turn["speaker"] not in channels or not 0 <= turn["start"] <= turn["end"] <= recording.duration + 0.001:
            raise ValueError(f"{row['call_id']}: invalid speaker or turn bounds")
    for current in turns:
        start, end = current["start"], min(current["end"], recording.duration)
        if not 0.5 <= end - start <= 30:
            continue
        history = [{"role": "user" if past["speaker"] == current["speaker"] else "assistant",
                    "content": plain_text(past["text"])}
                   for past in turns if past["id"] != current["id"] and past["end"] <= start]
        cut_id, channel = f"{row['call_id']}--{current['id']}", channels[current["speaker"]]
        yield MonoCut(
            id=cut_id, start=start, duration=end - start, channel=channel, recording=recording,
            supervisions=[SupervisionSegment(
                id=cut_id, recording_id=recording.id, start=0, duration=end - start,
                channel=channel, text=plain_text(current["text"]), language="Russian",
            )],
            custom={"history": history, "system": system, "source_group_id": row["call_id"],
                    "source_call_id": row["call_id"], "source_sha256": row["provenance"]["source_sha256"],
                    "original_split": row["split"], "split": "validation" if row["split"] == "dev" else "train",
                    "source_turn_id": current["id"], "boundary_source": "rnnt_emission_approximate",
                    "role_mapping": "current_speaker_user_other_speaker_assistant"},
        ).resample(16000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    validate_manifest(rows)
    system = plain_text(args.system_file.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts, seconds, skipped = Counter(), Counter(), Counter()
    with ExitStack() as stack:
        writers = {split: stack.enter_context(CutSet.open_writer(args.output_dir / f"{split}.jsonl.gz"))
                   for split in ("train", "validation")}
        for row in rows:
            if row["split"] not in ("train", "dev"):
                continue
            exported = 0
            for cut in call_cuts(row, args.audio_root, system):
                split = cut.custom["split"]
                writers[split].write(cut)
                counts[split] += 1
                seconds[split] += cut.duration
                exported += 1
            skipped[row["split"]] += len(row["turns"]) - exported
        if not counts["train"] or not counts["validation"]:
            raise ValueError("Both train and validation require eligible turns")
    print(json.dumps({"cuts": counts, "hours": {k: v / 3600 for k, v in seconds.items()},
                      "excluded_duration": skipped}, ensure_ascii=False))


if __name__ == "__main__":
    main()
