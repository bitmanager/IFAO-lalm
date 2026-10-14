"""Adapt existing foreground-mixture JSONL to native IFAO evaluation cuts."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml
from lhotse import CutSet, Recording, SupervisionSegment
from transformers import AutoTokenizer

from prepare_asr_manifest import asr_cut


FOREGROUND_INSTRUCTION = (
    "Расшифровывай только собеседника, который начал говорить первым. "
    "Одновременная речь другого человека на фоне не относится к его реплике."
)


def convert(source, output, audio_root, tokenizer, system):
    groups, seen = defaultdict(list), set()
    for line in source.read_text().splitlines():
        row = json.loads(line)
        if row["split"] not in ("validation", "test"):
            raise ValueError("This adapter exports held-out evaluation only")
        if row["id"] in seen or row["target_policy"] != "first_active_voice":
            raise ValueError("Duplicate ID or unsupported target policy")
        seen.add(row["id"])
        relative = Path(row["audio_path"]).relative_to("/mnt/local/drive1")
        path = (audio_root / relative).resolve()
        if not path.is_relative_to(audio_root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe source audio: {path}")
        recording = Recording.from_file(path, recording_id=row["id"])
        if recording.num_channels != 1 or abs(recording.duration - row["duration"]) > 0.001:
            raise ValueError("Source audio differs from the reviewed manifest")
        cut = recording.to_cut()
        cut.supervisions = [SupervisionSegment(
            id=cut.id, recording_id=recording.id, start=0, duration=cut.duration,
            channel=0, text=row["target_text"], language="ru",
        )]
        cut.custom = {"source_group_id": row["source_group_id"], "split": row["split"],
                      "label_source": "saved TTS script", "target_policy": row["target_policy"],
                      "background_text": row["background_text"]}
        condition = row["condition"]
        if condition == "mixture":
            condition += f'-snr{row["snr_db_requested"]:+g}'
        elif condition != "clean":
            raise ValueError("Unsupported foreground evaluation condition")
        # Compare ordinary transcription with the explicit target-speaker task.
        for policy, prompt in (("default", system),
                               ("first-voice", system + "\n" + FOREGROUND_INSTRUCTION)):
            groups[f"foreground-{condition}-{policy}"].append(asr_cut(cut, tokenizer, prompt))
    if not seen:
        raise ValueError("Empty foreground evaluation")
    output.mkdir(parents=True, exist_ok=False)
    config = []
    for name, cuts in groups.items():
        path = output / (name + ".jsonl.gz")
        CutSet.from_cuts(cuts).to_file(path)
        config.append({"name": name, "manifest": str(path),
                       "hours": sum(c.duration for c in cuts) / 3600})
    (output / "eval.yaml").write_text(yaml.safe_dump(config, allow_unicode=True))
    (output / "summary.json").write_text(json.dumps({
        "source": str(source), "unique_audio_examples": len(seen),
        "conditions": {name: len(cuts) for name, cuts in groups.items()},
        "scope": "Synthetic diagnostic, not human gold or proven unseen speaker identities",
        "main_training_connected": False,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, default=Path("/runs/dev-storage"))
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source_manifest, args.output_dir, args.audio_root,
            AutoTokenizer.from_pretrained(args.tokenizer), args.system_file.read_text().strip())
