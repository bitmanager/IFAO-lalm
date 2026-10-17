"""Export an overlap/SNR comparison using saved sources and upstream Lhotse mix.

No synthesis, model inference, enrollment, or training is performed here.
The first active voice is the declared target, independent of source level.
"""

import argparse
import hashlib
import json
from pathlib import Path

import lhotse
import numpy as np
import soundfile as sf
from lhotse import Recording

RATE, FRAME = 16000, 320


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def activity(audio):
    """QC only: 20 ms RMS frames above -30 dB of that source's peak frame RMS."""
    y = np.pad(audio.reshape(-1), (0, (-audio.size) % FRAME))
    rms = np.sqrt(np.mean(y.reshape(-1, FRAME) ** 2, axis=1))
    if rms.max() <= 1e-6:
        raise ValueError("Silent source")
    return rms > rms.max() * 10 ** (-30 / 20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[3, 0, -3])
    parser.add_argument("--exclude-pair", nargs="*", default=[], help="Explicit source pairs excluded by QC or split audit")
    args = parser.parse_args()
    if len(set(args.snr_db)) != len(args.snr_db) or any(not -6 <= x <= 6 for x in args.snr_db):
        raise ValueError("Expected unique SNR levels between -6 and +6 dB")
    sources = read_jsonl(args.source_pilot / "sources.jsonl")
    recipe = read_jsonl(args.source_pilot / "recipe.jsonl")
    excluded = set(args.exclude_pair)
    if excluded - {row["id"] for row in recipe}:
        raise ValueError("Unknown excluded pair")
    recipe = [row for row in recipe if row["id"] not in excluded]
    sources = [row for row in sources if row["pair_id"] not in excluded]
    if not recipe:
        raise ValueError("No source pairs retained")
    lookup = {(r["pair_id"], r["role"]): r for r in sources}
    if len(lookup) != len(sources) or len(sources) != len(recipe) * 2:
        raise ValueError("Expected exactly two existing sources per pair")
    for source in sources:
        if digest(source["path"]) != source["wav_sha256"]:
            raise ValueError("Source checksum changed")
    split_keys = {}
    for row in recipe:
        if Path(row["id"]).name != row["id"] or row["split"] not in ("train", "validation"):
            raise ValueError("Unsafe pair ID or unsupported split")
        for role in ("foreground", "background"):
            source = lookup[(row["id"], role)]
            if (source["split"], source["text"], source["voice"]) != (row["split"], row[role + "_text"], row[role + "_voice"]):
                raise ValueError("Source differs from recipe")
            for key in (("text", source["text"].casefold()), ("voice", source["voice"])):
                if split_keys.setdefault(key, row["split"]) != row["split"]:
                    raise ValueError("Source phrase or voice ID crosses splits")
    args.output.mkdir(parents=True, exist_ok=False)
    source_provenance = json.loads((args.source_pilot / "provenance.json").read_text())
    provenance = {"source_pilot": str(args.source_pilot), "source_provenance": source_provenance,
        "source_manifest_sha256": digest(args.source_pilot / "sources.jsonl"),
        "excluded_pair_ids": sorted(excluded),
        "adapter_sha256": digest(__file__), "mixer": "lhotse.Cut.mix / MixedCut.load_audio",
        "lhotse_version": lhotse.__version__, "snr_db_levels": args.snr_db,
        "target_policy": "First active voice; competing voice enters at least 0.26 seconds later; not selected by loudness",
        "activity_qc": "20 ms RMS frames > -30 dB relative to each isolated source's peak frame RMS; energy proxy, not speech annotation",
        "enrollment_conditioning": False, "main_training_connected": False,
        "label_quality": "Saved TTS scripts; consult this source pilot's ASR QC sidecars; human listening pending"}
    save_json(args.output / "provenance.json", provenance)
    examples, checks = [], []
    for row in recipe:
        pair = row["id"]
        cuts = [Recording.from_file(lookup[(pair, role)]["path"]).to_cut().resample(RATE)
                for role in ("foreground", "background")]
        raw = [c.load_audio() for c in cuts]
        masks = [activity(y) for y in raw]
        starts = [int(np.flatnonzero(mask)[0]) for mask in masks]
        offset_frames = max(0, starts[0] + 13 - starts[1])
        offset = offset_frames * FRAME / RATE
        size = max(len(masks[0]), offset_frames + len(masks[1]))
        fgmask = np.pad(masks[0], (0, size - len(masks[0])))
        bgmask = np.pad(masks[1], (offset_frames, size - offset_frames - len(masks[1])))
        overlap_mask = fgmask & bgmask
        overlap_seconds = float(overlap_mask.sum() * FRAME / RATE)
        overlap_fraction = float(overlap_mask.sum() / fgmask.sum())
        if overlap_fraction < 0.5:
            raise ValueError(f"Insufficient simultaneous activity for {pair}: {overlap_fraction}")
        gain = 0.06 / float(np.sqrt(np.mean(raw[0] ** 2)))
        target = cuts[0].perturb_volume(gain)
        pair_dir = args.output / "audio" / pair
        pair_dir.mkdir(parents=True)
        clean = pair_dir / "clean.wav"
        sf.write(clean, target.load_audio()[0], RATE, subtype="FLOAT")
        transcript = pair_dir / "target.txt"
        transcript.write_text(row["foreground_text"] + "\n")
        base = {"source_group_id": pair, "split": row["split"], "target_text": row["foreground_text"],
            "transcript_path": str(transcript), "sample_rate": RATE,
            "foreground_voice": row["foreground_voice"], "background_voice": row["background_voice"],
            "background_text": row["background_text"], "category": row["category"],
            "voice_bank_revision": source_provenance["voice_bank_revision"],
            "target_policy": "first_active_voice", "enrollment_conditioning": False,
            "asr_qc": "pending", "training_eligible": False}
        examples.append({**base, "id": pair + "-clean", "condition": "clean",
            "audio_path": str(clean), "target_audio_path": str(clean),
            "duration": target.duration, "num_samples": target.num_samples})
        for snr in args.snr_db:
            tag = f"snr_{snr:+g}db"
            dest = pair_dir / tag
            dest.mkdir()
            kwargs = dict(offset_other_by=offset, snr=snr, allow_padding=True)
            mixed = target.mix(cuts[1], **kwargs)
            peak = float(np.abs(mixed.load_audio()).max())
            common_gain = min(1.0, 0.9 / max(peak, 1e-8))
            if common_gain < 1:
                mixed = target.perturb_volume(common_gain).mix(cuts[1].perturb_volume(common_gain), **kwargs)
            audio, stems = mixed.load_audio(), mixed.load_audio(mixed=False)
            if isinstance(stems, list):
                stems = np.concatenate(stems, axis=0)
            if stems.shape != (2, audio.shape[1]):
                raise ValueError("Unexpected upstream track layout")
            error = float(np.max(np.abs(audio[0] - stems.sum(axis=0))))
            if error > 1e-6 or not np.isfinite(audio).all() or np.abs(audio).max() >= 0.95:
                raise ValueError("Invalid mixture")
            files = {name: dest / (name + ".wav") for name in ("mix", "target", "interference")}
            for name, y in zip(files, (audio[0], stems[0], stems[1])):
                sf.write(files[name], y, RATE, subtype="FLOAT")
            save_json(dest / "lhotse-mix.json", mixed.to_dict())
            sample_mask = np.repeat(overlap_mask, FRAME)[:audio.shape[1]]
            sample_mask = np.pad(sample_mask, (0, audio.shape[1] - sample_mask.size))
            overlap_snr = float(10 * np.log10(np.mean(stems[0, sample_mask] ** 2) / np.mean(stems[1, sample_mask] ** 2)))
            timing = {"foreground_offset_seconds": 0.0, "background_offset_seconds": offset,
                "foreground_first_active_seconds": starts[0] * FRAME / RATE,
                "background_first_active_seconds": (offset_frames + starts[1]) * FRAME / RATE,
                "simultaneous_active_seconds": overlap_seconds,
                "foreground_active_overlap_fraction": overlap_fraction}
            examples.append({**base, **timing, "id": pair + "-" + tag, "condition": "mixture",
                "audio_path": str(files["mix"]), "target_audio_path": str(files["target"]),
                "interference_audio_path": str(files["interference"]),
                "snr_db_requested": snr, "overlap_active_snr_db_measured": overlap_snr,
                "duration": audio.shape[1] / RATE, "num_samples": audio.shape[1]})
            checks.append({"pair_id": pair, "level": tag, **timing,
                "snr_db_requested": snr, "overlap_active_snr_db_measured": overlap_snr,
                "max_additivity_error": error, "peak": float(np.abs(audio).max()),
                "target_gain_total": gain * common_gain, "common_gain": common_gain,
                "background_gain_total": float(np.linalg.norm(stems[1]) / np.linalg.norm(raw[1])),
                "file_sha256": {name: digest(path) for name, path in files.items()}})
    exported_recipe = [{**{k: v for k, v in row.items() if k not in ("snr_db", "background_offset_seconds")},
                        "snr_db_levels": args.snr_db, "background_onset_policy": "At least 0.26 s after target activity"}
                       for row in recipe]
    for name, records in [("pilot.jsonl", examples), ("sources.jsonl", sources), ("recipe.jsonl", exported_recipe)]:
        (args.output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    for split in ("train", "validation"):
        (args.output / (split + ".jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in examples if r["split"] == split))
    save_json(args.output / "audio-qc.json", checks)
    summary = {"examples": len(examples), "pairs": len(recipe), "reused_tts_sources": len(sources),
        "mixtures": len(checks), "snr_db_levels": args.snr_db,
        "split_examples": {s: sum(r["split"] == s for r in examples) for s in ("train", "validation")},
        "manifest_hours_including_controls": sum(r["duration"] for r in examples) / 3600,
        "numeric_qc": "passed", "human_listening_qc": "pending", "training_eligible": False,
        "minimum_foreground_active_overlap_fraction": min(r["foreground_active_overlap_fraction"] for r in checks),
        "maximum_peak": max(r["peak"] for r in checks),
        "limitations": ["First-active-voice task needs a consistent instruction or conversation context",
            "No arbitrary target speaker enrollment; source activity timing is an energy proxy",
            "Background-only negatives remain in v1; first-entrant identity alone cannot label an isolated competing voice as non-target",
            "Synthetic sources, no telephone noise or codec, small held-out voice-ID set"]}
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
