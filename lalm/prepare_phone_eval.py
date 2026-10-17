"""Adapt the frozen phone-test TSV to native IFAO ASR evaluation cuts only."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from lhotse import CutSet, Recording, SupervisionSegment
from transformers import AutoTokenizer

from prepare_asr_manifest import asr_cut


DATASET = "Malecc/asr_calls_2_val"
REVISION = "36f9769d1548bfa9282b615737e5c29e29149198"
TSV_SHA256 = "5698520cf90eaa412cff62cf124d2fb63b023d5f5129882e7d015a5d865864bd"


def phone_eval_cut(row, index, audio_root, tokenizer, system):
    relative = Path(row["path"])
    root = audio_root.resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Expected an existing relative audio path inside audio_root")
    text = row["transcription"]
    if not text.strip() or "<|" in text:
        raise ValueError("Expected a nonempty plain reference transcript")
    expected_duration = float(row["duration"])
    if not math.isfinite(expected_duration) or expected_duration <= 0:
        raise ValueError("Invalid TSV duration")
    recording = Recording.from_file(path, recording_id=f"phone-test-{index:06d}")
    if recording.num_channels != 1:
        raise ValueError("Expected a mono phone-test recording")
    if abs(recording.duration - expected_duration) > 0.001:
        raise ValueError("Audio duration differs from the frozen TSV")
    cut = recording.to_cut().resample(16000)
    cut.supervisions = [SupervisionSegment(
        id=cut.id, recording_id=recording.id, start=0,
        duration=cut.duration, channel=0, text=text, language="ru",
    )]
    cut.custom = {
        "source_dataset": DATASET, "source_revision": REVISION,
        "source_split": "test", "source_row": index,
        "source_relative_path": row["path"],
        "source_audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "label_source": "Frozen dataset reference; human annotation provenance not independently verified",
        "evaluation_only": True, "training_eligible": False,
    }
    # Deliberately preserve sub-0.5 s eval items; the training CLI's duration
    # filter is not part of this frozen test protocol.
    return asr_cut(cut, tokenizer, system)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.tsv.read_bytes()).hexdigest() != TSV_SHA256:
        raise ValueError("TSV differs from the frozen phone-test reference")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    system = args.system_file.read_text().strip()
    if not system or "<|" in system:
        raise ValueError("Expected a nonempty plain system prompt")
    with args.tsv.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["path", "duration", "transcription"]:
            raise ValueError("Unexpected phone-test TSV schema")
        rows = list(reader)
    if len(rows) != 1943:
        raise ValueError("Expected all 1943 frozen phone-test rows; no filtering")
    if len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate audio path in phone-test TSV")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir / "phone-test.jsonl.gz"
    seconds, hashes = 0.0, set()
    with CutSet.open_writer(manifest, overwrite=False) as writer:
        for index, row in enumerate(rows):
            cut = phone_eval_cut(row, index, args.audio_root, tokenizer, system)
            writer.write(cut)
            seconds += cut.duration
            hashes.add(cut.source_audio_sha256)
    (args.output_dir / "eval.yaml").write_text(
        f"- name: phone1943-canonic\n  manifest: {json.dumps(str(manifest.resolve()))}\n"
    )
    report = {
        "dataset": DATASET, "revision": REVISION, "split": "test",
        "cuts": len(rows), "hours": seconds / 3600,
        "unique_source_audio_hashes": len(hashes),
        "tsv_sha256": hashlib.sha256(args.tsv.read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tokenizer": args.tokenizer, "system": system,
        "normalization": "References preserved unchanged; native IFAO and GigaAM metrics must be compared under the same normalizer",
        "evaluation_only": True, "training_eligible": False,
        "gpu_inference": "not run",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
