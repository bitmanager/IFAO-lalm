"""Adapt local Balalaika FLAC/JSON shards to ASR cuts, excluding held-out sources."""
import argparse
import json
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path

from lhotse import CutSet, Recording, SupervisionSegment
from transformers import AutoTokenizer

from prepare_asr_manifest import asr_cut


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    args = parser.parse_args()
    excluded = {row["voice"].rsplit(":", 1)[0] for row in json.loads(args.heldout.read_text())}
    if not excluded:
        raise ValueError("Expected nonempty source-level held-out list")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    system = args.system_file.read_text().strip()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts, seen, seconds = Counter(), set(), 0.0
    with CutSet.open_writer(args.output_dir / "train-asr.jsonl.gz", overwrite=False) as writer:
        for shard in sorted(args.shards.glob("*.tar")):
            with tarfile.open(shard) as archive:
                for member in archive.getmembers():
                    if not member.isfile() or not member.name.endswith(".json"):
                        continue
                    metadata = json.load(archive.extractfile(member))
                    if str(metadata["podcast_id"]) in excluded:
                        counts["heldout_source"] += 1
                        continue
                    text = metadata.get("punct", "").strip()
                    if not text or not re.search("[А-Яа-яЁё]", text) or "<|" in text:
                        counts["invalid_text"] += 1
                        continue
                    key = Path(member.name).stem
                    if key in seen:
                        raise ValueError(f"Duplicate source key: {key}")
                    seen.add(key)
                    path = args.output_dir / "audio" / shard.stem / f"{key}.flac"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(str(Path(member.name).with_suffix(".flac"))) as source, path.open("xb") as target:
                        shutil.copyfileobj(source, target)
                    recording = Recording.from_file(path, recording_id=f"youtube-{key}")
                    if recording.num_channels != 1 or not 0.5 <= recording.duration <= 30:
                        raise ValueError(f"Unexpected audio shape or duration: {path}")
                    cut = recording.to_cut()
                    cut.supervisions = [SupervisionSegment(
                        id=recording.id, recording_id=recording.id, start=0,
                        duration=recording.duration, channel=0, text=text, language="ru",
                    )]
                    cut.custom = {"source_dataset": "lab260/youtube_balalaika",
                                  "source_recording": str(metadata["podcast_id"]),
                                  "label_source": "Balalaika punctuated ASR consensus"}
                    writer.write(asr_cut(cut, tokenizer, system))
                    seconds += cut.duration
                    counts["accepted"] += 1
            print(json.dumps({"shard": shard.name, "counts": counts, "hours": seconds / 3600}), flush=True)
    if not counts["accepted"]:
        raise ValueError("No admitted YouTube audio")
    (args.output_dir / "summary.json").write_text(json.dumps({
        "counts": counts, "hours": seconds / 3600, "history": False,
        "excluded_source_ids": sorted(excluded), "labels": "automatic ASR, not human gold",
    }, indent=2))


if __name__ == "__main__":
    main()
