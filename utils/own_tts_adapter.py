"""Run MultiTalk's unchanged TTS worker/mixer against our Russian TTS API."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import re
import sys
import time
import types
from urllib.request import Request, urlopen

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


class OwnTTS:
    def __init__(self, **unused):
        pass

    def infer(self, *, spk_audio_prompt, text, **unused):
        payload = dict(model="optimized_short", voice=spk_audio_prompt,
                       input=text, language="ru", stream=True, response_format="pcm")
        req = Request(os.environ["OWN_TTS_URL"] + "/v1/audio/speech",
                      data=json.dumps(payload, ensure_ascii=False).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=90) as response:
            pcm = response.read(24000 * 2 * 90)
            if response.read(1):
                raise ValueError("TTS exceeded 90 seconds for one short turn")
        if len(pcm) < 4800 or len(pcm) % 2:
            raise ValueError("TTS returned empty or invalid PCM")
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        if np.max(np.abs(audio)) < 100:
            raise ValueError("TTS returned silence")
        # The upstream worker assumes 22.05 kHz and divides samples by 32768.
        return 22050, resample_poly(audio, 147, 160)


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_voice_map(mapping, records, eligible):
    if not isinstance(mapping, dict) or set(mapping) != {str(r["request_id"]) for r in records}:
        raise ValueError("Voice map must contain exactly the input request IDs")
    registry = {v["voice_id"]: v for v in eligible}
    for record in records:
        names = {p["name"] for p in record["participants"]}
        assigned = mapping[str(record["request_id"])]
        if (len(names) != len(record["participants"]) or "silence" in names
                or not isinstance(assigned, dict) or set(assigned) != names):
            raise ValueError("Voice map must contain exactly the participant names")
        if any(not isinstance(v, str) or v not in registry for v in assigned.values()):
            raise ValueError("Voice map contains an unavailable or unapproved training voice")
        if len(set(assigned.values())) != len(assigned):
            raise ValueError("Participants must use distinct voices")
        for participant in record["participants"]:
            if registry[assigned[participant["name"]]]["gender"] != participant["gender"]:
                raise ValueError("Voice gender conflicts with participant metadata")


def render_one(job):
    record, output_dir, upstream_dir, endpoint, voices, split, voice_map = job
    os.environ["OWN_TTS_URL"] = endpoint
    stub = types.ModuleType("indextts.infer_v2")
    stub.IndexTTS2 = OwnTTS
    sys.modules["indextts"] = types.ModuleType("indextts")
    sys.modules["indextts.infer_v2"] = stub
    sys.path.insert(0, upstream_dir)
    native = importlib.import_module("tts")
    native.load_audio_prompts = lambda language: (
        list(voices["male"]), list(voices["female"]))
    if not hasattr(native, "_original_assign_for_own_tts"):
        native._original_assign_for_own_tts = native.assign_prompts
    native.assign_prompts = native._original_assign_for_own_tts
    if voice_map is not None:
        native.assign_prompts = lambda example, male, female: {"silence": None, **voice_map}
    captured = {}
    if not hasattr(native, "_original_mix_for_own_tts"):
        native._original_mix_for_own_tts = native.mix_to_multichannel_wav
    original_mix = native._original_mix_for_own_tts

    def capture_mix(*args, **kwargs):
        result = original_mix(*args, **kwargs)
        captured["mapping"], captured["timings"] = result[1:]
        return result

    native.mix_to_multichannel_wav = capture_mix
    rid = str(record["request_id"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", rid):
        raise ValueError("Unsafe request_id")
    destination = Path(output_dir) / rid[:2] / (rid + ".wav")
    metadata_path = destination.with_suffix(".json")
    marker = destination.with_suffix(".complete.json")
    if marker.exists():
        if voice_map is not None:
            previous = json.loads(metadata_path.read_text())
            if previous["source_script"] != record or previous["voice"] != {"silence": None, **voice_map}:
                raise ValueError(f"Cached source/voice map conflicts: {rid}")
        return json.loads(marker.read_text())
    if destination.exists() or metadata_path.exists():
        raise ValueError(f"Incomplete prior output must be reviewed: {rid}")
    tasks, results = queue.Queue(), queue.Queue()
    tasks.put(dict(example=record, output_path=str(destination), request_id=rid))
    tasks.put(None)
    native.tts_worker(0, tasks, results, enable_eval=False)
    result = results.get_nowait()
    if result["status"] != "success":
        raise RuntimeError(result)
    meta = json.loads(metadata_path.read_text())
    if voice_map is not None and meta["voice"] != {"silence": None, **voice_map}:
        raise ValueError(f"Rendered voice map conflicts: {rid}")
    segments = []
    for start, end, index in captured["timings"]:
        turn = record["dialogue"][index]
        channel = captured["mapping"][turn["speaker"]]
        segments.append(dict(speaker="A" if channel == 0 else "B", channel=channel,
                             participant=turn["speaker"], start=start, end=end,
                             text=turn["text"], words=[]))
    meta.update(audio=str(destination), split=split, segments=segments,
                status="alignment_pending", training_ready=False,
                source_script=record, sample_rate=22050,
                tts_provenance=dict(endpoint=endpoint, model="bitmanagerai/Qwen3TTS-v3.7-mix50",
                                    revision="384ffff264f0407498f3ca7138871b9cf03f69f6",
                                    controls_applied=["voice", "language"],
                                    controls_not_applied=["emotion", "speed", "volume", "dialect"]),
                assembly="upstream MultiTalk tts_worker + mix_to_multichannel_wav")
    meta["metadata:all_transcript"] = segments
    metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    info = sf.info(destination)
    assert info.channels == 2 and info.samplerate == 22050 and info.duration > 0
    for segment in segments:
        assert 0 <= segment["start"] < segment["end"] <= info.duration + 1 / 22050
    result.update(path=str(destination), metadata=str(metadata_path), split=split,
                  duration=info.duration, status="alignment_pending", training_ready=False)
    marker.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--tts-url", default="http://127.0.0.1:19099")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--voice-map", help="JSON mapping exact request IDs to participant names and approved voice IDs")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with urlopen(args.tts_url + "/health/ready", timeout=5) as f:
        health = json.load(f)
    assert health["resolved_revision"] == "384ffff264f0407498f3ca7138871b9cf03f69f6"
    with urlopen(args.tts_url + "/v1/audio/voices", timeout=5) as f:
        registry = json.load(f)
    eligible = [v for v in registry["voices"] if v.get("language") == "ru"
                and not any("validation" in t.lower() for t in v.get("tags", []))
                and v["voice_id"] in {"anastasia-1", "ekaterina-1", "college-01", "college-02",
                                      "college-03", "college-04", "college-05"}]
    voices = {gender: [v["voice_id"] for v in eligible if v["gender"] == gender]
              for gender in ("male", "female")}
    assert all(voices.values()), "Missing eligible training voice pool"
    records = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    assert records and len({x["request_id"] for x in records}) == len(records)
    for record in records:
        assert record["config"]["language"] == "ru"
        assert sorted(p["role"] for p in record["participants"]) == ["assistant", "user"]
    voice_map = None
    if args.voice_map:
        voice_map = json.loads(Path(args.voice_map).read_text(), object_pairs_hook=unique_json_object)
        validate_voice_map(voice_map, records, eligible)
    provenance = dict(input=args.input, input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                      upstream_sha256=hashlib.sha256((Path(args.upstream) / "tts.py").read_bytes()).hexdigest(),
                      health=health, voice_revision=registry["remote_revision"], voices=voices,
                      records=len(records), workers=args.workers, started_at=time.time())
    if voice_map is not None:
        provenance.update(voice_map=voice_map, voice_map_sha256=hashlib.sha256(Path(args.voice_map).read_bytes()).hexdigest())
    provenance_path = out / "provenance.json"
    if voice_map is not None and provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous.get("input_sha256") != provenance["input_sha256"] or previous.get("voice_map") != voice_map:
            raise ValueError("Existing output provenance conflicts with input/voice map")
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    jobs = [(record, str(out / "multichannel_wavs"), args.upstream, args.tts_url, voices,
             record.get("split", "train"), voice_map[str(record["request_id"])] if voice_map is not None else None) for record in records]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for result in pool.map(render_one, jobs):
            results.append(result)
            (out / "rendered.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results))
            print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = dict(state="render_complete_alignment_pending", count=len(results),
                   hours=sum(x["duration"] for x in results) / 3600, training_ready=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
