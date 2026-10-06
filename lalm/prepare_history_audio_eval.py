"""Format fixed text-history cases as audio evaluation using existing Teachers/Lhotse.

Only the final user utterance is synthesized. Identical utterances share a WAV;
systems, histories and authored expected-fact rubrics are preserved verbatim.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from lhotse import CutSet, Recording, SupervisionSegment
from transformers import AutoTokenizer

from prepare_conversation import prepare_asr_cut, prepare_cut


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_views(case, path, tokenizer):
    messages = case["messages"]
    if (messages[0]["role"], messages[-1]["role"]) != ("system", "user"):
        raise ValueError("Expected system and final user message")
    if any(m["role"] not in ("system", "user", "assistant") or not isinstance(m["content"], str)
           or "<|" in m["content"] for m in messages):
        raise ValueError("Expected plain text messages")
    recording = Recording.from_file(path)
    cut = recording.to_cut()
    cut.id = "historyfacts-" + case["idx"]
    cut.supervisions = [SupervisionSegment(id=cut.id, recording_id=recording.id,
        start=0, duration=cut.duration, channel=0, language="ru", text=messages[-1]["content"],
        custom={"answer": case["expected"]})]
    cut.custom = {"system": messages[0]["content"], "history": copy.deepcopy(messages[1:-1]),
        "task": "answer", "case_id": case["idx"],
        "case_type": "with_history" if messages[1:-1] else "no_history",
        "source_group_id": case["idx"].removesuffix("_no_history").rsplit("_", 1)[0],
        "split": "test", "evaluation_only": True, "training_eligible": False,
        "main_training_connected": False, "expected_facts": case["expected"],
        "reference_kind": "Fixed case expected-fact rubric; not a teacher-generated response"}
    answer = copy.deepcopy(cut)
    answer.id += "-answer"
    return prepare_cut(answer, tokenizer), prepare_asr_cut(cut, tokenizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--teacher-client", type=Path, required=True)
    parser.add_argument("--tts-url", required=True)
    parser.add_argument("--voice", default="anastasia-1")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(x) for x in args.cases.read_text().splitlines() if x.strip()]
    if len({r["idx"] for r in cases}) != len(cases) or any(Path(r["idx"]).name != r["idx"] for r in cases):
        raise ValueError("Duplicate or unsafe case ID")
    if any(not r["expected"].strip() or not r["messages"][-1]["content"].strip() for r in cases):
        raise ValueError("Empty current utterance or expected-fact rubric")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    spec = importlib.util.spec_from_file_location("existing_teacher_client", args.teacher_client)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.output.mkdir(parents=True, exist_ok=False)
    client = module.Teachers(args.tts_url, "", args.output / "raw-source")
    response = client.client.get(args.tts_url.rstrip("/") + "/health/ready")
    response.raise_for_status()
    health = response.json()
    if health.get("upstream_model") != "bitmanagerai/Qwen3TTS-v3.7-mix50":
        raise ValueError("Unexpected existing TTS model")
    if (health.get("sample_rate"), health.get("channels"), health.get("audio_format")) != (24000, 1, "int16"):
        raise ValueError("Existing Teachers client requires 24 kHz mono PCM16")
    catalog = client.voices()
    voice = next(v for v in catalog["voices"] if v["voice_id"] == args.voice)
    if voice.get("language") != "ru" or "validation" in voice.get("tags", []):
        raise ValueError("Expected an existing ordinary Russian voice")
    (args.output / "cases.jsonl").write_bytes(args.cases.read_bytes())
    (args.output / "audio").mkdir()
    waveforms, sources, rows = {}, [], []
    for case in cases:
        text = case["messages"][-1]["content"]
        if text not in waveforms:
            raw_path, metadata = client.synthesize(text, args.voice)
            source = Recording.from_file(raw_path).to_cut().resample(16000)
            audio = source.load_audio()
            if not .3 <= source.duration <= 15 or not np.isfinite(audio).all() or float(np.max(np.abs(audio))) < .001:
                raise ValueError("Unexpected source audio")
            key = hashlib.sha256(text.encode()).hexdigest()
            path = args.output / "audio" / (key + ".wav")
            sf.write(path, audio[0], 16000, subtype="FLOAT")
            waveforms[text] = path
            sources.append({"text": text, "voice_id": args.voice, "audio_path": str(path),
                "raw_audio_path": str(raw_path), "raw_wav_sha256": digest(raw_path),
                "wav_sha256": digest(path), "duration": source.duration,
                "sample_rate": 16000, "tts_metadata": metadata,
                "peak": float(np.max(np.abs(audio))), "acoustic_asr_qc": "pending"})
            print(json.dumps({"synthesized_unique_utterance": len(sources), "duration": source.duration}), flush=True)
        rows.extend(build_views(case, waveforms[text], tokenizer))
    for task in ("answer", "asr"):
        task_rows = [c for c in rows if c.task == task]
        CutSet.from_cuts(task_rows).to_file(args.output / f"history-{task}16.jsonl.gz")
        config = []
        for condition in ("with_history", "no_history"):
            subset = [c for c in task_rows if c.case_type == condition]
            name = f"history-facts-{condition}-{task}"
            path = args.output / (name + ".jsonl.gz")
            CutSet.from_cuts(subset).to_file(path)
            config.append({"name": name, "manifest": str(path)})
        (args.output / f"eval-{task}.yaml").write_text(yaml.safe_dump(config))
    (args.output / "sources.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sources))
    provenance = {"cases_sha256": digest(args.cases), "adapter_sha256": digest(__file__),
        "teacher_client": str(args.teacher_client), "teacher_client_sha256": digest(args.teacher_client),
        "tts_url": args.tts_url, "tts_health": health, "voice_bank_revision": catalog["remote_revision"],
        "voice": voice, "evaluation_only": True, "training_eligible": False,
        "main_training_connected": False, "history_audio": False,
        "expected_facts_source": "Unchanged fixed cases.jsonl", "teacher_answer_generation": False,
        "cases": len(cases), "unique_current_utterance_wavs": len(sources),
        "unique_audio_seconds": sum(r["duration"] for r in sources),
        "per_task_exposure_seconds": sum(c.duration for c in rows if c.task == "answer"),
        "acoustic_asr_qc": "pending", "human_listening": "not performed"}
    (args.output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    client.client.close()
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
