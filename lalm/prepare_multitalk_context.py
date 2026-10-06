"""Adapt existing MultiTalk rendered dialogues to contextual Lhotse cuts.

Only turn-level bounds are required. No word alignment, re-synthesis or new VAD.
These synthetic bounds are not represented as production commit events.
"""

import argparse
import json
from collections import Counter
from itertools import groupby
from pathlib import Path

from lhotse import CutSet, MonoCut, Recording, SupervisionSegment


def dialogue_cuts(metadata_path, system):
    data = json.loads(metadata_path.read_text())
    recording = Recording.from_file(metadata_path.with_suffix(".wav"), recording_id=data["id"])
    roles = {p["name"]: p["role"] for p in data["source_script"]["participants"]}
    if set(roles.values()) != {"user", "assistant"} or len(roles) != 2:
        raise ValueError(f"{metadata_path}: V1 expects one user and one assistant")
    segments = data["segments"]
    if not segments or segments != sorted(segments, key=lambda s: s["start"]):
        raise ValueError(f"{metadata_path}: missing or unordered segments")
    history = []
    past_end = 0.0
    for index, (role, group) in enumerate(groupby(segments, key=lambda s: roles[s["participant"]])):
        turns = list(group)
        start, end = turns[0]["start"], max(t["end"] for t in turns)
        text = " ".join(t["text"].strip() for t in turns)
        if not text or "<|" in text or not 0 <= start < end <= recording.duration + 0.001:
            raise ValueError(f"{metadata_path}: invalid turn {index}")
        if role == "user":
            if past_end > end:
                raise ValueError(f"{metadata_path}: history extends beyond current audio")
            channel = turns[0]["channel"]
            if any(t["channel"] != channel for t in turns):
                raise ValueError(f"{metadata_path}: user fragments span multiple channels")
            duration = min(end, recording.duration) - start
            if not 0.5 <= duration <= 30:
                raise ValueError(f"{metadata_path}: turn duration {duration} exceeds V1 bounds")
            cut_id = f"{data['id']}-turn-{index:04d}"
            yield MonoCut(
                id=cut_id, start=start, duration=duration, channel=channel,
                recording=recording,
                supervisions=[SupervisionSegment(
                    id=cut_id, recording_id=recording.id, start=0, duration=duration,
                    channel=channel, text=text, language="Russian",
                )],
                custom={
                    "history": list(history), "system": system,
                    "source_group_id": data["source_script"]["source_group_id"],
                    "split": data["split"], "boundary_source": "synthetic_rendered_turn",
                },
            ).resample(16000)
        history.append({"role": role, "content": text})
        past_end = max(past_end, end)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    system = args.system_file.read_text().strip()
    if not system or "<|" in system:
        raise ValueError("A nonempty plain-text production system prompt is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts, seconds, groups = Counter(), Counter(), {}
    from contextlib import ExitStack
    with ExitStack() as stack:
        writers = {split: stack.enter_context(CutSet.open_writer(args.output_dir / f"{split}.jsonl.gz"))
                   for split in ("train", "validation", "test")}
        for audio_path in sorted(args.metadata_root.glob("**/multichannel_wavs/*/*.wav")):
            path = audio_path.with_suffix(".json")
            for cut in dialogue_cuts(path, system):
                split = cut.custom["split"]
                split = "validation" if split == "val" else split
                group = cut.source_group_id
                if groups.setdefault(group, split) != split:
                    raise ValueError(f"Source group {group} crosses dataset splits")
                writers[split].write(cut)
                counts[split] += 1
                seconds[split] += cut.duration
        if not counts:
            raise ValueError("No rendered dialogues found")
    print(json.dumps({"cuts": counts, "hours": {k: v / 3600 for k, v in seconds.items()},
                      "source_groups": len(groups)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
