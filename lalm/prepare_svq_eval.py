"""Export pinned Russian SVQ recordings for evaluation only, without rewriting audio.

Uses the existing ASR conversation adapter and the stock GigaAM TSV format.
The four recorded environments are strata, not paired clean/noisy waveforms.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import wave
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import pyarrow.parquet as pq
from lhotse import CutSet, Recording, SupervisionSegment
from transformers import AutoTokenizer

from prepare_asr_manifest import asr_cut


DATASET = "google/svq"
REVISION = "74f17ec92f0654860c160029d3542199626f63ef"
ENVIRONMENTS = ("clean", "background_speech", "media_noise", "traffic_noise")
COLUMNS = ["utt_id", "speaker_id", "speaker_age", "speaker_gender", "locale",
           "environment", "text", "transcript_truth", "transcriptions/speech",
           "page_id_in_lang", "page_id_cross_lang", "passage_id_in_lang",
           "passage_id_cross_lang", "span_context_id_in_lang", "span_context_id_cross_lang"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_pilot(rows, per_environment, seed):
    """Freeze a hash-ranked subset independently of parquet row ordering."""
    if per_environment <= 0:
        raise ValueError("Pilot count must be positive")
    selected = []
    for environment in ENVIRONMENTS:
        candidates = [r for r in rows if r["environment"] == environment]
        if len(candidates) < per_environment:
            raise ValueError(f"Not enough rows for {environment}")
        candidates.sort(key=lambda r: (hashlib.sha256(
            f"{seed}\0{r['utt_id']}".encode()).hexdigest(), r["utt_id"]))
        selected.extend(r["utt_id"] for r in candidates[:per_environment])
    return selected


def prepare(source_root, output_dir, tokenizer, system, per_environment=100,
            seed="svq-ru-pilot-v1"):
    source_root, output_dir = Path(source_root).resolve(), Path(output_dir).resolve()
    if not system.strip() or "<|" in system:
        raise ValueError("Expected a plain nonempty system prompt")
    manifest = json.loads((source_root / "source-manifest.json").read_text())
    if (manifest.get("repository"), manifest.get("revision")) != (DATASET, REVISION):
        raise ValueError("Expected the pinned SVQ source")
    if not manifest.get("all_lfs_hashes_verified") or manifest.get("training_eligible") is not False:
        raise ValueError("Expected verified evaluation-only staging")
    expected = {f"2.0.0/utts_ru_ru_{e}.parquet" for e in ENVIRONMENTS}
    shards = sorted(manifest["files"], key=lambda f: f["path"])
    if len(shards) != 4 or {s["path"] for s in shards} != expected:
        raise ValueError("Expected exactly four Russian SVQ shards")
    rows, seen = [], set()
    for shard in shards:
        path = source_root / shard["path"]
        if path.stat().st_size != shard["bytes"] or sha256_file(path) != shard["lfs_sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        metadata = pq.read_table(path, columns=COLUMNS).to_pylist()
        for index, row in enumerate(metadata):
            uid = row["utt_id"]
            if not re.fullmatch(r"utt_[0-9]+", uid) or uid in seen:
                raise ValueError(f"Invalid or duplicate utterance ID: {uid}")
            seen.add(uid)
            if row["locale"] != "ru_ru" or shard["path"] != f"2.0.0/utts_ru_ru_{row['environment']}.parquet":
                raise ValueError(f"Unexpected locale/environment: {uid}")
            text = row["transcript_truth"]
            if row["transcriptions/speech"] is not True or not isinstance(text, str) or not text.strip() or "<|" in text:
                raise ValueError(f"Missing plain ASR reference: {uid}")
            row.update(source_parquet=shard["path"], source_row=index,
                       source_parquet_sha256=shard["lfs_sha256"])
            rows.append(row)
    pilot = select_pilot(rows, per_environment, seed)
    pilot_set = set(pilot)
    output_dir.mkdir(parents=True, exist_ok=False)
    pilot_path = output_dir / "pilot_ids.json"
    pilot_path.write_text(json.dumps({"repository": DATASET, "revision": REVISION,
        "seed": seed, "per_environment": per_environment, "utt_ids": pilot,
        "selection": "Lowest SHA256(seed + NUL + utt_id) per environment, before inference",
        "evaluation_only": True, "training_eligible": False}, indent=2) + "\n")
    counts, seconds, differences, durations = Counter(), Counter(), Counter(), []
    partials = []
    with ExitStack() as stack:
        writers, tsvs = {}, {}
        for split in ("full", "pilot"):
            for environment in ("all", *ENVIRONMENTS):
                name = f"{split}-{environment}"
                final = output_dir / f"{name}.jsonl.gz"
                partial = output_dir / f"{name}.incomplete.jsonl.gz"
                writers[split, environment] = stack.enter_context(CutSet.open_writer(partial, overwrite=False))
                partials.append((partial, final))
                final = output_dir / f"{name}.tsv"
                partial = output_dir / f"{name}.incomplete.tsv"
                stream = stack.enter_context(partial.open("x", newline=""))
                tsvs[split, environment] = csv.writer(stream, delimiter="\t")
                tsvs[split, environment].writerow(["path", "duration", "transcription"])
                partials.append((partial, final))
        index_stream = stack.enter_context((output_dir / "source-index.jsonl").open("x"))
        offset = 0
        for shard in shards:
            for batch in pq.ParquetFile(source_root / shard["path"]).iter_batches(batch_size=32, columns=["waveform"]):
                for audio in batch.to_pylist():
                    row = rows[offset]
                    offset += 1
                    uid, environment = row["utt_id"], row["environment"]
                    data = audio["waveform"]["bytes"]
                    with wave.open(io.BytesIO(data)) as wav:
                        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                            raise ValueError(f"{uid}: expected mono16k PCM16 WAV")
                        frames = wav.getnframes()
                    if frames <= 0:
                        raise ValueError(f"{uid}: empty audio")
                    duration = frames / 16000
                    audio_path = output_dir / "audio" / environment / f"{uid}.wav"
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    with audio_path.open("xb") as stream:
                        stream.write(data)
                    recording = Recording.from_file(audio_path, recording_id=uid)
                    if recording.num_samples != frames:
                        raise ValueError(f"{uid}: audio/header sample count mismatch")
                    cut = recording.to_cut()
                    cut.supervisions = [SupervisionSegment(id=uid, recording_id=uid,
                        start=0, duration=duration, channel=0, text=row["transcript_truth"],
                        language="ru", speaker=row["speaker_id"])]
                    metadata = {**row, "source_dataset": DATASET, "source_revision": REVISION,
                        "source_audio_sha256": hashlib.sha256(data).hexdigest(),
                        "derived_query_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                        "label_source": "SVQ transcript_truth; not the displayed text prompt",
                        "evaluation_only": True, "training_eligible": False,
                        "pilot_member": uid in pilot_set}
                    cut.custom = metadata
                    # Keep every evaluation duration; do not apply a training filter.
                    cut = asr_cut(cut, tokenizer, system)
                    index_stream.write(json.dumps({**metadata, "cut_id": cut.id,
                        "audio_filepath": str(audio_path), "duration": duration}, ensure_ascii=False) + "\n")
                    for split in (("full", "pilot") if uid in pilot_set else ("full",)):
                        for group in ("all", environment):
                            writers[split, group].write(cut)
                            tsvs[split, group].writerow([str(audio_path), duration, row["transcript_truth"]])
                        counts[f"{split}/{environment}"] += 1
                        seconds[f"{split}/{environment}"] += duration
                    differences[environment] += row["text"] != row["transcript_truth"]
                    durations.append(duration)
            print(json.dumps({"completed_shard": shard["path"], "cuts": offset}), flush=True)
        if offset != len(rows):
            raise ValueError("Source row count changed")
    for partial, final in partials:
        partial.rename(final)
    for split in ("full", "pilot"):
        (output_dir / f"{split}.yaml").write_text("".join(
            f"- name: svq-ru-{split}-{e}\n  manifest: {json.dumps(str(output_dir / f'{split}-{e}.jsonl.gz'))}\n"
            for e in ENVIRONMENTS))
    summary = {"source_dataset": DATASET, "source_revision": REVISION,
        "counts": dict(counts), "hours": {k: v / 3600 for k, v in seconds.items()},
        "reference_differs_from_prompt": dict(differences),
        "speakers": {e: len({r['speaker_id'] for r in rows if r['environment'] == e}) for e in ENVIRONMENTS},
        "duration_min": min(durations), "duration_max": max(durations),
        "under_0_5_seconds": sum(d < .5 for d in durations), "over_30_seconds": sum(d > 30 for d in durations),
        "pilot_ids_sha256": sha256_file(pilot_path), "adapter_sha256": sha256_file(__file__),
        "source_manifest_sha256": sha256_file(source_root / "source-manifest.json"),
        "audio": "Original embedded WAV bytes, no resampling, segmentation or filtering",
        "query_ids": "No upstream query_id; text retained and derived_query_sha256 is explicitly derived",
        "pairing": "Four independently recorded environment strata, not exact paired clean/noisy waveforms",
        "split_policy": "Full corpus and nested pilot remain evaluation-only; no speaker/text-disjoint claim",
        "evaluation_only": True, "training_eligible": False, "gpu_inference": "not run"}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    parser.add_argument("--per-environment", type=int, default=100)
    parser.add_argument("--seed", default="svq-ru-pilot-v1")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(json.dumps(prepare(args.source_root, args.output_dir, tokenizer,
        args.system_file.read_text().strip(), args.per_environment, args.seed), indent=2))


if __name__ == "__main__":
    main()
