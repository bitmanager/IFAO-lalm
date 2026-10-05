"""Adapt an explicit phrase recipe through an existing TTS client and Lhotse.mix.

This exports an isolated pilot manifest, not native training cuts. Speech synthesis
and waveform mixing are delegated to existing upstream implementations.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import lhotse
import numpy as np
import soundfile as sf
from lhotse import Recording


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_recipe(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids, texts, voices = set(), {}, {}
    for row in rows:
        if row["id"] in ids or Path(row["id"]).name != row["id"]:
            raise ValueError("Duplicate or unsafe pair ID")
        ids.add(row["id"])
        if row["split"] not in ("train", "validation"):
            raise ValueError("Unsupported split")
        if not 6 <= row["snr_db"] <= 24 or not 0 <= row["background_offset_seconds"] <= 2:
            raise ValueError("Expected a quieter background and bounded offset")
        if row["foreground_voice"] == row["background_voice"]:
            raise ValueError("The two sources must use distinct voice IDs")
        for role in ("foreground", "background"):
            text, voice = row[role + "_text"], row[role + "_voice"]
            if not text.strip() or len(text) > 220:
                raise ValueError("Expected a short, nonempty explicit phrase")
            for groups, key in ((texts, text.casefold()), (voices, voice)):
                if groups.setdefault(key, row["split"]) != row["split"]:
                    raise ValueError("Source phrase or voice ID crosses splits")
    if not rows:
        raise ValueError("Empty recipe")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--teacher-client", type=Path, required=True)
    parser.add_argument("--tts-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_recipe(args.recipe)
    # Import the reviewed existing client without copying or changing its code.
    spec = importlib.util.spec_from_file_location("existing_teacher_client", args.teacher_client)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.output.mkdir(parents=True, exist_ok=False)
    client = module.Teachers(args.tts_url, "", args.output / "raw-source")
    response = client.client.get(args.tts_url.rstrip("/") + "/health/ready")
    response.raise_for_status()
    health = response.json()
    if health.get("upstream_model") != "bitmanagerai/Qwen3TTS-v3.7-mix50":
        raise ValueError("Unexpected TTS model")
    if (health.get("sample_rate"), health.get("channels"), health.get("audio_format")) != (24000, 1, "int16"):
        raise ValueError("Existing Teachers client requires 24 kHz mono PCM16")
    catalog = client.voices()
    selected = {row[role + "_voice"] for row in rows for role in ("foreground", "background")}
    references = {v["voice_id"]: v for v in catalog["voices"] if v["voice_id"] in selected}
    if set(references) != selected or any("validation" in v.get("tags", []) for v in references.values()):
        raise ValueError("Unknown or reserved benchmark voice")
    provenance = {
        "recipe_sha256": digest(args.recipe), "adapter_sha256": digest(__file__),
        "teacher_client": str(args.teacher_client), "teacher_client_sha256": digest(args.teacher_client),
        "tts_url": args.tts_url, "tts_health": health,
        "voice_bank_revision": catalog["remote_revision"], "voice_refs": references,
        "mixer": "lhotse.Cut.mix / MixedCut.load_audio", "lhotse_version": lhotse.__version__,
        "target_policy": "Designated foreground, louder than competing speech; empty when absent",
        "enrollment_conditioning": False, "main_training_connected": False,
        "label_quality": "Authored TTS input; acoustic transcript verification pending",
    }
    write_json(args.output / "provenance.json", provenance)
    (args.output / "recipe.jsonl").write_bytes(args.recipe.read_bytes())
    examples, checks, sources = [], [], []
    rate = 16000
    for row in rows:
        pair_dir = args.output / "audio" / row["id"]
        pair_dir.mkdir(parents=True)
        cuts = []
        for role in ("foreground", "background"):
            path, info = client.synthesize(row[role + "_text"], row[role + "_voice"])
            if not 0.3 <= info["duration"] <= 15:
                raise ValueError(f"Unexpected TTS duration for {row['id']} {role}: {info['duration']}")
            source = {"pair_id": row["id"], "split": row["split"], "role": role,
                      "path": str(path), "wav_sha256": digest(path), **info}
            sources.append(source)
            cuts.append(Recording.from_file(path).to_cut().resample(rate))
            print(json.dumps({"synthesized": row["id"], "role": role, "seconds": info["duration"]}), flush=True)
        raw_target = cuts[0].load_audio()
        target_gain = 0.06 / float(np.sqrt(np.mean(raw_target ** 2)))
        target = cuts[0].perturb_volume(target_gain)
        interference = cuts[1]
        kwargs = {"offset_other_by": row["background_offset_seconds"],
                  "snr": row["snr_db"], "allow_padding": True}
        mixed = target.mix(interference, **kwargs)
        preliminary = mixed.load_audio()
        common_gain = min(1.0, 0.9 / max(float(np.abs(preliminary).max()), 1e-8))
        if common_gain < 1:
            mixed = target.perturb_volume(common_gain).mix(interference.perturb_volume(common_gain), **kwargs)
        audio, stems = mixed.load_audio(), mixed.load_audio(mixed=False)
        if isinstance(stems, list):
            stems = np.concatenate(stems, axis=0)
        if stems.shape != (2, audio.shape[1]):
            raise ValueError("Unexpected Lhotse track layout")
        error = float(np.max(np.abs(audio[0] - stems.sum(axis=0))))
        if error > 1e-6 or not np.isfinite(audio).all() or float(np.abs(audio).max()) >= 0.95:
            raise ValueError("Invalid additive mixture or insufficient headroom")
        silence = np.zeros_like(stems[0])
        files = {name: pair_dir / (name + ".wav") for name in ("mix", "target", "interference", "silence")}
        for name, samples in (("mix", audio[0]), ("target", stems[0]), ("interference", stems[1]), ("silence", silence)):
            # FLOAT WAV preserves upstream additive references without PCM16 clipping/rounding.
            sf.write(files[name], samples, rate, subtype="FLOAT")
        write_json(pair_dir / "lhotse-mix.json", mixed.to_dict())
        duration = audio.shape[1] / rate
        for condition, inp, tgt, bg, text in (
            ("clean", "target", "target", "silence", row["foreground_text"]),
            ("mixture", "mix", "target", "interference", row["foreground_text"]),
            ("background_only", "interference", "silence", "interference", ""),
        ):
            example_id = row["id"] + "-" + condition
            text_path = pair_dir / (condition + ".txt")
            text_path.write_text(text + ("\n" if text else ""))
            examples.append({"id": example_id, "source_group_id": row["id"], "split": row["split"],
                "condition": condition, "audio_path": str(files[inp]), "target_audio_path": str(files[tgt]),
                "interference_audio_path": str(files[bg]), "target_text": text, "transcript_path": str(text_path),
                "sample_rate": rate, "num_samples": audio.shape[1], "duration": duration,
                "foreground_voice": row["foreground_voice"], "background_voice": row["background_voice"],
                "voice_bank_revision": catalog["remote_revision"], "snr_db_requested": row["snr_db"],
                "background_offset_seconds": row["background_offset_seconds"],
                "foreground_present": condition != "background_only", "category": row["category"],
                "asr_qc": "pending", "training_eligible": False})
        checks.append({"pair_id": row["id"], "duration": duration, "max_additivity_error": error,
            "peak": float(np.abs(audio).max()), "target_gain": target_gain, "common_gain": common_gain,
            "aligned_stem_snr_db": float(10 * np.log10(np.mean(stems[0] ** 2) / np.mean(stems[1] ** 2))),
            "file_sha256": {k: digest(v) for k, v in files.items()}})
    for name, records in (("pilot.jsonl", examples), ("sources.jsonl", sources)):
        (args.output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    for split in ("train", "validation"):
        subset = [r for r in examples if r["split"] == split]
        (args.output / (split + ".jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset))
    write_json(args.output / "audio-qc.json", checks)
    summary = {"examples": len(examples), "pairs": len(rows), "tts_sources": len(sources),
        "split_examples": {s: sum(r["split"] == s for r in examples) for s in ("train", "validation")},
        "manifest_hours_including_controls": sum(r["duration"] for r in examples) / 3600,
        "unique_mixture_hours": sum(r["duration"] for r in checks) / 3600,
        "raw_source_hours": sum(r["duration"] for r in sources) / 3600,
        "empty_targets": sum(not r["target_text"] for r in examples),
        "numeric_qc": "passed", "asr_qc": "pending", "human_listening_qc": "pending",
        "split_policy": "Disjoint authored phrases and voice IDs; each pair and controls stay together",
        "limitations": ["Synthetic foreground salience curriculum, not arbitrary speaker selection",
                       "No enrollment, real telephone codec/noise, or human gold transcripts",
                       "Small validation set; voice-ID disjointness does not independently verify speaker identity",
                       "Background-only target silence follows the declared task, not speech absence",
                       "Not connected to main training; empty labels are intentionally outside asr_cut"]}
    write_json(args.output / "summary.json", summary)
    client.client.close()
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
