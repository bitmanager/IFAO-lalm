"""Format cached native VAD and call metadata for the existing Qwen runner."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import wave


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def finite_time(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def group_speech_segments(segments):
    """Join nearby VAD islands, never crossing a known agent turn or 30s."""
    grouped = []
    for source in segments:
        segment = copy.deepcopy(source)
        if not 0 <= segment["start"] < segment["end"] or segment["end"] - segment["start"] > 30:
            raise ValueError("Invalid speech interval")
        if grouped and segment["start"] < grouped[-1]["end"]:
            raise ValueError("Overlapping speech intervals")
        if (grouped and segment["start"] - grouped[-1]["end"] <= 2.0
                and segment["end"] - grouped[-1]["start"] <= 30
                and segment["history"] == grouped[-1]["history"]):
            grouped[-1]["end"] = segment["end"]
            grouped[-1]["native_vad_end"] = segment["native_vad_end"]
            grouped[-1]["source_segment_ids"].append(segment["id"])
        else:
            segment["source_segment_ids"] = [segment["id"]]
            grouped.append(segment)
    for index, segment in enumerate(grouped):
        segment["id"] = f"segment-{index:04d}"
    return grouped


def speech_segments(timestamps, duration):
    """Pad rounded native VAD bounds, clipping adjacent padding at gap midpoints."""
    if not timestamps or not finite_time(duration) or duration <= 0:
        raise ValueError("Positive native speech timestamps and duration required")
    previous = 0
    for span in timestamps:
        start, end = span["start"], span["end"]
        if not all(map(finite_time, (start, end))) or not 0 <= start < end <= duration or start < previous:
            raise ValueError("Invalid or overlapping native VAD timestamps")
        previous = end
    segments = []
    for index, span in enumerate(timestamps):
        start, end = max(0, span["start"] - .1), min(duration, span["end"] + .1)
        if index:
            start = max(start, (timestamps[index - 1]["end"] + span["start"]) / 2)
        if index + 1 < len(timestamps):
            end = min(end, (span["end"] + timestamps[index + 1]["start"]) / 2)
        start, end = round(start * 16000) / 16000, round(end * 16000) / 16000
        if end - start > 30:
            raise ValueError("Native VAD span exceeds 30s; needs native segmentation before this adapter")
        segments.append(dict(id=f"segment-{index:04d}", start=start, end=end, history=[],
                             native_vad_start=span["start"], native_vad_end=span["end"]))
    return segments


def prepare_call(environment, asset, call, vad, audio_path, source_hashes):
    """Client text is never an input; uncertain clocks/aliases hold agent text too."""
    if vad["audio_sha256"] != asset["user_sha256"] or not vad["speech_detected"]:
        raise ValueError("Native speech gate/source identity mismatch")
    if call["call_id"] not in asset["call_ids"]:
        raise ValueError("Selected source call is not an asset alias")
    with wave.open(str(audio_path), "rb") as audio:
        if audio.getframerate() != 16000 or audio.getnchannels() != 1:
            raise ValueError("Expected native mono 16kHz WAV")
        duration = audio.getnframes() / 16000
    if digest(audio_path) != asset["user_sha256"] or abs(duration - vad["duration_seconds"]) > .001:
        raise ValueError("WAV hash/duration differs from cached native VAD")
    segments = speech_segments(vad["timestamps"], duration)
    turns = call["online_turns"]
    aliases = len(asset["call_ids"]) > 1
    bad_clock = any(not finite_time(t.get("at")) or not 0 <= t["at"] <= duration + .5 for t in turns)
    flags = ["audio_clock_alignment_unverified"]
    if aliases:
        flags += ["agent_history_unverified", "original_agent_history_held_for_multi_call_alias"]
    if bad_clock:
        flags += ["original_agent_history_held_for_invalid_clock"]
    if not aliases and not bad_clock:
        for segment in segments:
            segment["history"] = [dict(role="assistant", content=t["text"], source_at_seconds=t["at"],
                                       source_available_at_seconds=t["at"])
                                  for t in turns if t["role"] == "agent" and t["at"] < segment["start"]
                                  and "<|" not in t["text"]]
    segments = group_speech_segments(segments)
    return dict(environment=environment, media_id=asset["media_id"], call_id=call["call_id"],
                source_call_ids=asset["call_ids"], user_audio_path=str(audio_path), user_sha256=asset["user_sha256"],
                source_duration_seconds=duration, channel_role_verified=asset.get("channel_role_verified", False),
                channel_policy=asset.get("channel_policy"), segments=segments,
                history_policy=dict(selection="past assistant events plus completed Qwen self-transcripts",
                                    original_client_text_included=False, current_and_future_turns_included=False,
                                    self_transcript_availability="prior completed segment end", gemini_or_giga_text_used=False,
                                    original_agent_history_held=aliases or bad_clock, flags=flags),
                segmentation_provenance=dict(backend=vad["backend"], version=vad["version"], config=vad["config"],
                                             padding_seconds=.1, gap_policy="midpoint-clipped nonoverlap",
                                             grouping="gap <=2s; <=30s; identical past agent context",
                                             timestamps_precision="native rounded seconds; not word alignment",
                                             source_audio_sha256=vad["audio_sha256"], vad_source_sha256=source_hashes["vad"]),
                source_selection=dict(canonical_key="environment/media_id/user_sha256", source_hashes=source_hashes,
                                      call_id_use="source CSV first alias; provenance only when ambiguous"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--vad", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--source", nargs=3, action="append", required=True, metavar=("ENV", "ROOT", "REVIEW_CSV"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    plan, vad = json.loads(args.plan.read_text()), json.loads(args.vad.read_text())
    if not vad["complete"] or args.output.exists():
        raise ValueError("Complete native VAD required and output must be new")
    vad_by_key = {(r["environment"], r["media_id"]): r for r in vad["results"]}
    sources = {}
    for env, root, review in args.source:
        if env in sources:
            raise ValueError("Duplicate source environment")
        root, review = Path(root), Path(review)
        assets = {r["media_id"]: r for r in map(json.loads, (root / "assets.jsonl").read_text().splitlines())}
        calls = {r["call_id"]: r for r in map(json.loads, (root / "calls.jsonl").read_text().splitlines())}
        selected = {}
        with review.open() as file:
            for row in csv.DictReader(file):
                selected.setdefault(row["media_id"], row["call_id"])
        sources[env] = (assets, calls, selected, dict(assets=digest(root / "assets.jsonl"),
                                                   calls=digest(root / "calls.jsonl"), review=digest(review), vad=digest(args.vad)))
    probe = json.loads(args.probe.read_text())
    spec = {key: probe[key] for key in ("checkpoint", "checkpoint_sha256", "system_prompt", "user_suffix")}
    spec.update(history_mode="reference_history_v2", calls=[])
    seen = set()
    for row in plan["source_keys"]:
        env, media = row["source_environment"], row["media_id"]
        if (env, media) in seen:
            raise ValueError("Duplicate canonical key")
        seen.add((env, media))
        assets, calls, selected, hashes = sources[env]
        asset = assets[media]
        if asset["user_sha256"] != row["audio_sha256"]:
            raise ValueError("Canonical audio SHA differs from source asset")
        spec["calls"].append(prepare_call(env, asset, calls[selected[media]], vad_by_key[env, media],
                                          Path(asset["user_audio_path"]), hashes))
    if len(spec["calls"]) != plan["eligible_task_count"]:
        raise ValueError("Canonical plan count mismatch")
    spec["dataset_provenance"] = dict(plan_sha256=digest(args.plan), vad_sha256=digest(args.vad),
                                      probe_sha256=digest(args.probe), adapter_sha256=digest(Path(__file__)),
                                      canonical_keys=len(seen), unique_audio_sha=len({c["user_sha256"] for c in spec["calls"]}))
    with args.output.open("x") as file:
        json.dump(spec, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(dict(path=str(args.output), sha256=digest(args.output), calls=len(spec["calls"]),
                         segments=sum(len(c["segments"]) for c in spec["calls"]))))


if __name__ == "__main__":
    main()
