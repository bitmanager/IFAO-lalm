"""Adapt official OpenSTT phone training CSV/audio/text to the existing ASR task."""
import argparse
import csv
import hashlib
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


def text_key(text):
    return " ".join(re.findall(r"\w+", text.casefold().replace("ё", "е")))


def duration_bin(duration):
    return ("<0.5" if duration < 0.5 else "0.5-1" if duration < 1 else
            "1-2" if duration < 2 else "2-5" if duration <= 5 else ">5")


def convert(manifest, audio_root, output, tokenizer, system, excluded_ids=(),
            exclusions=None, manifest_audio_root=None):
    """Keep common short text matches; quarantine source/hash or long-text matches.

    Prefix mapping changes file references only, after native ASR preparation
    has validated the real local files. Audio bytes and targets are unchanged.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_name(output.name + ".incomplete.jsonl.gz")
    exclusions = exclusions or {}
    excluded_ids = set(excluded_ids) | set(exclusions.get("ids", []))
    blocked_hashes = set(exclusions.get("sha256", []))
    blocked_text = {text_key(t) for t in exclusions.get("texts", []) if text_key(t)}
    counts, seen, seconds = Counter(), {}, Counter()
    observed_text, admitted_hashes, bad_hashes = {}, set(), {}
    bins = {}

    def account(reason, duration):
        counts[reason] += 1
        seconds[reason] += duration
        bucket = bins.setdefault(reason, {})
        stat = bucket.setdefault(duration_bin(duration), {"count": 0, "hours": 0.0})
        stat["count"] += 1
        stat["hours"] += duration / 3600

    excluded_path = output.with_name(output.name + ".excluded.jsonl")
    with open(manifest, newline="") as stream, \
            CutSet.open_writer(partial, overwrite=False) as writer, \
            excluded_path.open("x") as rejected:
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
            account("source_unique", float(duration))
            if not 0.5 <= float(duration) <= 30:
                account("duration", float(duration))
                continue
            audio = source_path(audio_root, audio_rel)
            text = source_path(audio_root, text_rel).read_text().strip()
            digest = hashlib.sha256(audio.read_bytes()).hexdigest()
            normalized = text_key(text)
            if digest in observed_text and observed_text[digest] != normalized:
                bad_hashes[digest] = "conflicting_audio_transcripts"
            observed_text[digest] = normalized
            aliases = {key, key + "-asr", audio_rel, str(audio),
                       Path(audio_rel).name, Path(audio_rel).stem}
            reason = None
            if aliases & excluded_ids or digest in blocked_hashes:
                reason = "heldout_audio"
                bad_hashes[digest] = reason
            elif normalized in blocked_text:
                if len(normalized.split()) >= 4:
                    reason = "heldout_long_exact_text"
                    bad_hashes[digest] = reason
                else:
                    account("common_short_text_match", float(duration))
            if reason is None and digest in admitted_hashes:
                reason = "duplicate_audio_bytes"
            if reason:
                account(reason, float(duration))
                rejected.write(json.dumps({"id": key, "reason": reason,
                    "duration": float(duration), "audio_sha256": digest}) + "\n")
                continue
            if not text or not re.search("[А-Яа-яЁё]", text) or "<|" in text:
                account("invalid_text", float(duration))
                continue
            recording = Recording.from_file(audio, recording_id=key)
            if recording.num_channels != 1:
                raise ValueError(f"Expected mono audio: {audio}")
            if abs(recording.duration - float(duration)) > 0.1:
                raise ValueError(f"Duration disagrees with official CSV: {audio}")
            if not 0.5 <= recording.duration <= 30:
                account("duration", recording.duration)
                continue
            cut = recording.to_cut()
            cut.supervisions = [SupervisionSegment(
                id=key, recording_id=key, start=0, duration=recording.duration,
                channel=0, text=text, language="ru",
            )]
            cut.custom = {"source_dataset": subset, "label_source": "OpenSTT automatic ASR",
                          "source": {"audio_sha256": digest, "audio_relative": audio_rel,
                                     "text_relative": text_rel, "csv_duration": float(duration)},
                          "heldout_common_short_text_match": normalized in blocked_text}
            cut = asr_cut(cut, tokenizer, system)
            if manifest_audio_root is not None:
                mapped = str(Path(manifest_audio_root) / audio.relative_to(audio_root.resolve()))
                cut.recording.sources[0].source = mapped
                for message in cut.conversation:
                    if isinstance(message["content"], list):
                        for item in message["content"]:
                            if item.get("type") == "audio":
                                item["audio"] = mapped
            writer.write(cut)
            admitted_hashes.add(digest)
            account("accepted", cut.duration)
            if counts["accepted"] % 10000 == 0:
                print(json.dumps({"counts": counts, "hours": seconds["accepted"] / 3600}), flush=True)
    # A later source row may reveal an earlier admitted byte-identical recording
    # as held-out or inconsistently labelled. Remove every such variant.
    with CutSet.open_writer(output, overwrite=False) as writer, excluded_path.open("a") as rejected:
        for cut in CutSet.from_file(partial) or []:
            digest = cut.custom["source"]["audio_sha256"]
            if digest in bad_hashes:
                reason = bad_hashes[digest] + "_closure"
                account(reason, cut.duration)
                counts["accepted"] -= 1
                seconds["accepted"] -= cut.duration
                stat = bins["accepted"][duration_bin(cut.duration)]
                stat["count"] -= 1
                stat["hours"] -= cut.duration / 3600
                rejected.write(json.dumps({"id": cut.id, "reason": reason,
                    "duration": cut.duration, "audio_sha256": digest}) + "\n")
                continue
            writer.write(cut)
    partial.unlink()
    if not counts["accepted"]:
        raise ValueError("No admitted phone training audio")
    return {"counts": counts, "hours": seconds["accepted"] / 3600, "history": False,
            "hours_by_reason": {k: v / 3600 for k, v in seconds.items()}, "duration_bins": bins,
            "labels": "automatic ASR, not human gold", "manifest": str(manifest),
            "output_manifest": str(output), "audio_root": str(audio_root),
            "manifest_audio_root": str(manifest_audio_root) if manifest_audio_root else None,
            "exclusions_sources": exclusions.get("sources", []),
            "text_policy": "exact normalized >=4 words excluded; 1-3 words counted and retained unless source/hash excluded",
            "hash_scope": "encoded file bytes, not cross-codec acoustic identity",
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[])
    parser.add_argument("--exclusions-json", type=Path)
    parser.add_argument("--manifest-audio-root", type=Path)
    args = parser.parse_args()
    excluded = set()
    for manifest in args.exclude_csv:
        with manifest.open() as stream:
            excluded.update(Path(row[0]).stem for row in csv.reader(stream))
    summary = convert(args.manifest, args.audio_root, args.output,
                      AutoTokenizer.from_pretrained(args.tokenizer),
                      args.system_file.read_text().strip(), excluded,
                      json.loads(args.exclusions_json.read_text()) if args.exclusions_json else None,
                      args.manifest_audio_root)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
