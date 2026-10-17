"""Export staged SOVA train WAV bytes and wrap utterances in the existing ASR task.

No decoding/re-encoding, segmentation, history, or new labels are introduced.
Exclusions are an audited JSON object with texts, ids, sha256, and sources lists.
"""
import argparse
import hashlib
import io
import json
import re
import wave
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from lhotse import CutSet, Recording, SupervisionSegment
from transformers import AutoTokenizer

from prepare_asr_manifest import asr_cut


def text_key(text):
    """Conservative equality: ignore case, punctuation, spacing and е/ё."""
    return " ".join(re.findall(r"\w+", text.casefold().replace("ё", "е")))


def quarantine_audio_hashes(input_manifest, output_manifest, hash_reasons):
    """Remove every admitted variant of audited conflicting/held-out audio."""
    removed = []
    with CutSet.open_writer(output_manifest, overwrite=False) as writer:
        for cut in CutSet.from_file(input_manifest):
            provenance = cut.custom["source"]
            reason = hash_reasons.get(provenance["audio_sha256"])
            if reason:
                removed.append({"id": cut.id.removesuffix("-asr"), "reason": reason,
                                "duration": cut.duration, "source": provenance})
            else:
                writer.write(cut)
    return removed


def prepare(source_root, output_dir, exclusions, tokenizer, system):
    source_root, output_dir = Path(source_root), Path(output_dir)
    source_manifest = source_root / "manifest.json"
    manifest = json.loads(source_manifest.read_text())
    if manifest["source"]["split"] != "train":
        raise ValueError("Only the staged train split is accepted")
    if not system.strip() or "<|" in system:
        raise ValueError("Expected a nonempty plain system prompt")
    shards = sorted((f for f in manifest["files"] if f["path"].endswith(".parquet")),
                    key=lambda f: f["path"])
    if not shards or any(not Path(f["path"]).name.startswith("train-") for f in shards):
        raise ValueError("Only train parquet shards are accepted")
    blocked_text = {text_key(t) for t in exclusions["texts"] if text_key(t)}
    blocked_ids = set(exclusions["ids"])
    blocked_hashes = set(exclusions["sha256"])
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / "train.jsonl.gz"
    if final.exists():
        raise FileExistsError(final)
    partial = output_dir / "train.incomplete.jsonl.gz"
    counts, seconds = Counter(), Counter()
    seen_hashes, seen_ids = set(), set()
    observed_texts, conflicting_hashes, heldout_text_hashes = {}, set(), set()
    shard_reports = []
    with CutSet.open_writer(partial, overwrite=False) as writer, \
            (output_dir / "excluded.jsonl").open("x") as rejected:
        for shard in shards:
            path = source_root / shard["path"]
            if not shard.get("checksum_verified") or path.stat().st_size != shard["size_bytes"]:
                raise ValueError(f"Missing verified staging checksum or changed size: {path}")
            shard_id = int(path.name.split("-")[1])
            n = 0
            before_count, before_seconds = counts["kept"], seconds["kept"]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["audio", "transcription"]):
                for row in batch.to_pylist():
                    row_id = n
                    n += 1
                    audio, text = row["audio"], row["transcription"]
                    data = audio["bytes"]
                    digest = hashlib.sha256(data).hexdigest()
                    normalized = text_key(text) if isinstance(text, str) else None
                    if digest in observed_texts and observed_texts[digest] != normalized:
                        conflicting_hashes.add(digest)
                    observed_texts[digest] = normalized
                    if normalized in blocked_text:
                        heldout_text_hashes.add(digest)
                    uid = f"sova-train-{shard_id:05d}-{row_id:06d}"
                    with wave.open(io.BytesIO(data)) as wav:
                        rate, frames = wav.getframerate(), wav.getnframes()
                        duration = frames / rate
                        if (rate, wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                            raise ValueError(f"{uid}: expected source mono16k PCM16 WAV")
                    counts["source"] += 1
                    seconds["source"] += duration
                    reason = None
                    source_path = audio.get("path")
                    aliases = {uid, uid + "-asr", digest}
                    if source_path:
                        aliases.update((source_path, Path(source_path).name, Path(source_path).stem))
                    if aliases & blocked_ids or digest in blocked_hashes:
                        reason = "heldout_id_or_sha256"
                    elif isinstance(text, str) and text_key(text) in blocked_text:
                        reason = "heldout_normalized_exact_text"
                    elif not isinstance(text, str) or not text.strip() or "<|" in text:
                        reason = "invalid_text"
                    elif not 0.5 <= duration <= 30:
                        reason = "duration_outside_existing_asr_range"
                    elif digest in seen_hashes:
                        reason = "duplicate_audio_bytes"
                    if uid in seen_ids:
                        raise ValueError(f"Duplicate source ID: {uid}")
                    seen_ids.add(uid)
                    provenance = {"dataset": manifest["source"]["repo_id"],
                                  "revision": manifest["source"]["revision"], "split": "train",
                                  "parquet": shard["path"], "parquet_sha256": shard["sha256"],
                                  "row": row_id, "source_audio_path": source_path,
                                  "audio_sha256": digest,
                                  "text_origin": "manual annotation according to upstream dataset card; not independently re-annotated"}
                    if reason:
                        counts[reason] += 1
                        seconds[reason] += duration
                        rejected.write(json.dumps({"id": uid, "reason": reason,
                                                   "duration": duration, "source": provenance}, ensure_ascii=False) + "\n")
                        continue
                    seen_hashes.add(digest)
                    audio_path = output_dir / "audio" / f"{shard_id:05d}" / f"{uid}.wav"
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    with audio_path.open("xb") as stream:
                        stream.write(data)
                    recording = Recording.from_file(audio_path, recording_id=uid)
                    if recording.num_samples != frames:
                        raise ValueError(f"{uid}: Lhotse/header sample count mismatch")
                    cut = recording.to_cut()
                    cut.supervisions = [SupervisionSegment(id=uid, recording_id=uid,
                        start=0, duration=recording.duration, channel=0, language="ru", text=text)]
                    cut.custom = {"source": provenance}
                    writer.write(asr_cut(cut, tokenizer, system))
                    counts["kept"] += 1
                    seconds["kept"] += duration
            if n != shard["rows"]:
                raise ValueError(f"{path}: staging row count changed")
            report = {"path": shard["path"], "source_rows": n,
                      "kept": counts["kept"] - before_count,
                      "kept_seconds": seconds["kept"] - before_seconds}
            shard_reports.append(report)
            print(json.dumps(report), flush=True)
    if counts["source"] != manifest["train_rows"] or abs(seconds["source"] - manifest["duration_seconds"]) > 0.001:
        raise ValueError("Staged source totals changed")
    hash_reasons = {h: "conflicting_audio_transcripts" for h in conflicting_hashes}
    hash_reasons.update({h: "heldout_audio_hash_closure" for h in heldout_text_hashes})
    removed = quarantine_audio_hashes(partial, final, hash_reasons)
    with (output_dir / "excluded.jsonl").open("a") as rejected:
        for row in removed:
            counts["kept"] -= 1
            seconds["kept"] -= row["duration"]
            counts[row["reason"]] += 1
            seconds[row["reason"]] += row["duration"]
            shard_report = next(s for s in shard_reports if s["path"] == row["source"]["parquet"])
            shard_report["kept"] -= 1
            shard_report["kept_seconds"] -= row["duration"]
            rejected.write(json.dumps(row, ensure_ascii=False) + "\n")
    partial.unlink()
    summary = {"counts": dict(counts), "hours": {k: v / 3600 for k, v in seconds.items()},
               "shards": shard_reports, "source": manifest["source"],
               "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
               "source_checksum_policy": "Existing staging verified SHA256 and current file size; parquet bytes not rehashed",
               "train_manifest": str(final),
               "train_manifest_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
               "heldout_sources": exclusions.get("sources", []),
               "exclusion_counts": {"texts": len(blocked_text), "ids": len(blocked_ids), "sha256": len(blocked_hashes)},
               "history": False, "speaker_disjoint": "unknown; no speaker IDs",
               "upstream_validation_test_comparison": "Unavailable locally; only train was staged",
               "audio": "Original embedded WAV bytes; mono 16000 Hz PCM16; no resampling or segmentation"}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "train.yaml").write_text(
        f"- manifest: {final}\n  name: sova-rudevices-human-train\n  hours: {seconds['kept']/3600}\n  weights: 1\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclusions-json", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    args = parser.parse_args()
    exclusions = json.loads(args.exclusions_json.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = prepare(args.source_root, args.output_dir, exclusions, tokenizer, args.system_file.read_text().strip())
    print(json.dumps({"counts": result["counts"], "hours": result["hours"]}))


if __name__ == "__main__":
    main()
