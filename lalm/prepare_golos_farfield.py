"""Stage pinned Golos farfield train rows or the complete eval-only test split.

No shuffled selection, transcript normalization, transcription, or training admission.
Original JSONL manifests, when supplied, are compared verbatim by original ID.
"""
import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

REPO = "Sh1man/golos_opus"
REVISION = "931f5c412a8045e0b03b9e5584072f0810c41dcc"
LIMIT = 1000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pcm_sha256(audio):
    return hashlib.sha256(audio.astype("<f4", copy=False).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--original-manifest", type=Path, action="append", default=[])
    parser.add_argument("--heldout-manifest", type=Path, action="append", required=True)
    parser.add_argument("--heldout-index", type=Path, action="append", required=True)
    args = parser.parse_args()
    expected_count = 124003 if args.split == "train" else 1916
    if not 1 <= args.limit <= expected_count:
        raise ValueError(f"Expected a positive limit no larger than {expected_count}")
    if args.split == "test" and args.limit != expected_count:
        raise ValueError("Official test requires all 1916 source rows; no subset")
    for path in (args.output_dir, args.cache_dir):
        if not any(path.resolve().is_relative_to(root) for root in
                   (Path("/runs/dev-storage"), Path("/mnt/local/drive1"))):
            raise ValueError(f"Use dev storage for output and cache: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for key, suffix in (("HF_HOME", "home"), ("HF_HUB_CACHE", "hub"),
                        ("HF_DATASETS_CACHE", "datasets")):
        os.environ[key] = str(args.cache_dir / suffix)
    # Import after setting caches: no default exp-root HF cache writes.
    import datasets
    import lhotse
    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset
    from lhotse import CutSet, Recording, SupervisionSegment
    from transformers import AutoTokenizer
    from prepare_asr_manifest import asr_cut

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    system = args.system_file.read_text().strip()
    if not system or "<|" in system:
        raise ValueError("Expected the existing plain system prompt")
    inputs = args.original_manifest + args.heldout_manifest + args.heldout_index + [args.system_file]
    provenance = {str(path): sha256(path) for path in inputs}
    excluded_ids, excluded_files, excluded_pcm = set(), set(), set()
    heldout_counts = Counter()
    for path in args.heldout_index:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                for key in ("cut_id", "utt_id", "original_audio_filename"):
                    if row.get(key):
                        excluded_ids.add(Path(row[key]).stem)
                for key in ("source_audio_sha256", "derived_query_sha256"):
                    if row.get(key):
                        excluded_files.add(row[key])
                heldout_counts["index_rows"] += 1
    for path in args.heldout_manifest:
        for cut in CutSet.from_file(path):
            cut = cut.resample(16000)
            audio = cut.load_audio()
            if audio is None or not np.isfinite(audio).all():
                raise ValueError(f"Invalid heldout audio: {cut.id}")
            excluded_ids.update((cut.id, cut.recording_id))
            excluded_pcm.add((cut.sampling_rate, tuple(audio.shape), pcm_sha256(audio)))
            heldout_counts["decoded_cuts"] += 1
    if not heldout_counts["decoded_cuts"] or not heldout_counts["index_rows"]:
        raise ValueError("Expected both heldout native audio and original source indexes")

    originals = {}
    for path in args.original_manifest:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                prefix = "farfield/" if args.split == "train" else "files/"
                if not row["audio_filepath"].startswith(prefix):
                    continue
                if row["id"] in originals and originals[row["id"]] != row:
                    raise ValueError(f"Conflicting original manifest ID: {row['id']}")
                originals[row["id"]] = row
    source = load_dataset(REPO, "farfield", split=args.split, revision=REVISION,
                          streaming=True, cache_dir=str(args.cache_dir / "datasets"))
    source = source.cast_column("opus", Audio(decode=False))
    counts, seen, seconds = Counter(), set(), 0.0
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir()
    manifest_name = "pilot-asr.jsonl.gz" if args.split == "train" else "eval-asr.jsonl.gz"
    corpus_id = "golos-farfield" if args.split == "train" else "golos-farfield-test"
    with CutSet.open_writer(args.output_dir / manifest_name, overwrite=False) as writer, \
            (args.output_dir / "source-index.jsonl").open("x") as index, \
            (args.output_dir / "unresolved.jsonl").open("x") as unresolved:
        for number, row in enumerate(source.take(args.limit)):
            metadata = row["json"]
            key, text = metadata["id"], metadata["text"]
            if (not re.fullmatch(r"[0-9a-f]{28,64}", key) or key != row["__key__"]
                    or key in seen or f"farfield/{args.split}/" not in row["__url__"]):
                raise ValueError(f"Unexpected {args.split} source identity at row {number}")
            seen.add(key)
            path = audio_dir / f"{key}.opus"
            payload = row["opus"]["bytes"]
            if not payload:
                raise ValueError(f"Missing original OPUS bytes: {key}")
            with path.open("xb") as stream:
                stream.write(payload)
            info = sf.info(path)
            original_recording = Recording.from_file(path, recording_id=f"{corpus_id}-{key}")
            recording = original_recording.resample(16000)
            audio = recording.load_audio()
            if recording.num_channels != 1 or not np.isfinite(audio).all():
                raise ValueError(f"Invalid mono audio: {key}")
            item = dict(source_row=number, original_id=key, original_metadata=metadata,
                        source_dataset=REPO, revision=REVISION, config="farfield", split=args.split,
                        source_shard=row["__url__"], audio_filepath=str(path),
                        source_audio_sha256=sha256(path), pcm_sha256=pcm_sha256(audio),
                        original_audio_format=info.format, original_audio_subtype=info.subtype,
                        original_sampling_rate=original_recording.sampling_rate,
                        sampling_rate=recording.sampling_rate, duration=recording.duration,
                        history_available=False, speaker_identity_available=False,
                        training_eligible=False)
            if args.split == "test":
                item["evaluation_only"] = True
            original = originals.get(key)
            item["original_manifest_match"] = None
            if original:
                item["original_manifest_match"] = (original["text"] == text and
                    abs(float(original["duration"]) - float(metadata["duration"])) <= 0.001)
                item["original_manifest_record"] = original
            reasons = []
            if not isinstance(text, str) or not text.strip() or "<|" in text:
                reasons.append("empty_or_invalid_reference")
            if abs(recording.duration - float(metadata["duration"])) > 0.05:
                reasons.append("declared_duration_mismatch")
            if args.split == "train" and not 0.5 <= recording.duration <= 30:
                reasons.append("outside_native_duration_range")
            if original and not item["original_manifest_match"]:
                reasons.append("original_manifest_mismatch")
            if (key in excluded_ids or recording.id in excluded_ids or
                    item["source_audio_sha256"] in excluded_files or
                    (recording.sampling_rate, tuple(audio.shape), item["pcm_sha256"]) in excluded_pcm):
                if args.split == "train":
                    reasons.append("heldout_identity_or_exact_audio")
                else:
                    item["other_eval_identity_or_exact_audio_overlap"] = True
                    counts["other_eval_overlap_rows"] += 1
            item["unresolved_reasons"] = reasons
            counts["source_rows"] += 1
            counts["original_labels_verified" if original and item["original_manifest_match"]
                   else "original_labels_unverified"] += 1
            index.write(json.dumps(item, ensure_ascii=False) + "\n")
            if reasons:
                counts["unresolved_rows"] += 1
                counts.update(reasons)
                unresolved.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue
            cut = recording.to_cut()
            cut.supervisions = [SupervisionSegment(id=recording.id, recording_id=recording.id,
                start=0, duration=recording.duration, channel=0, text=text, language="ru")]
            cut.custom = {**item, "label_source": "Golos OPUS mirror transcript; verification recorded separately"}
            writer.write(asr_cut(cut, tokenizer, system))
            counts["native_staging_cuts"] += 1
            seconds += recording.duration
    if counts["source_rows"] != args.limit:
        raise ValueError(f"Expected exactly {args.limit} source rows: {counts}")
    summary = dict(counts=counts, unique_native_audio_hours=seconds / 3600,
        source_dataset=REPO, revision=REVISION, config="farfield", split=args.split,
        selection=f"First {args.limit} streaming {args.split} rows before any checks; no shuffle",
        input_sha256=provenance, adapter_sha256=sha256(__file__), heldout_counts=heldout_counts,
        datasets_version=datasets.__version__, lhotse_version=lhotse.__version__,
        training_eligible=False, label_text_unchanged=True,
        pcm_comparison="Stock Lhotse 16 kHz mono float32 little-endian; original OPUS bytes preserved",
        license="Primary custom Golos license; HF CC-BY-NC tag does not match primary PDF",
        limitations=["Exact PCM checks cannot rule out recoded or cropped duplicates",
                      "No speaker-disjoint claim; no future validation identities provided",
                      "Unverified original-manifest labels remain staging, not admitted train"],
        output_sha256={p.name: sha256(p) for p in args.output_dir.iterdir() if p.is_file()})
    if args.split == "test":
        summary["evaluation_only"] = True
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
