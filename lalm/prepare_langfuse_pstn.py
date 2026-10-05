"""Adapt a reviewed, exported Langfuse PSTN selection without estimating boundaries.

The recorder's CH0 is caller and CH1 is agent. The transport inserts non-acoustic
user messages into its context: remove only its exact initial greeting trigger,
and quarantine calls containing the known missing-transcript marker. Preserve
the complete source events separately. Text remains operational ASR pseudo.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from lhotse import CutSet, MonoCut, Recording, SupervisionSegment

SEED = {"role": "client", "text": "Начни разговор"}
MISSING = "[реплика клиента не распознана — коротко и вежливо переспроси]"


def client_text(events, row):
    matches = []
    for event in events:
        body = event["body"]
        output = body.get("output")
        if isinstance(output, str):
            output = json.loads(output)
        if not isinstance(output, dict) or row["media_id"] not in str(output.get("recording", "")):
            continue
        if body["id"] != row["trace_id"] or body["sessionId"] != row["session_id"]:
            raise ValueError("Media and trace/session identities disagree")
        matches.append(output["transcript"])
    if len(matches) != 1 or not matches[0] or matches[0][0] != SEED:
        raise ValueError("Expected one matching event with the verified initial transport seed")
    clients = [turn["text"].strip() for turn in matches[0][1:] if turn["role"] == "client"]
    if any(text == MISSING for text in clients):
        return None, "missing_asr_marker"
    if any(text == SEED["text"] or "<|" in text for text in clients):
        return None, "unexpected_control_text"
    text = " ".join(clients)
    if not re.search("[А-Яа-яЁё]", text):
        return None, "no_cyrillic_client_text_after_seed"
    return text, None


def native_cuts(args):
    """Reuse the frozen IVR converter's RNNT boundaries; keep only caller ASR."""
    from prepare_calls_context import call_cuts, validate_manifest

    if args.system_file is None or args.accepted_calls is None:
        raise ValueError("Native calls require system-file and the reviewed accepted-calls manifest")
    accepted = {row["metadata"]["source_call_id"]: row["metadata"] for row in
                map(json.loads, args.accepted_calls.read_text().splitlines())}
    rows = list(map(json.loads, args.native_calls.read_text().splitlines()))
    validate_manifest(rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    count, seconds = 0, 0.0
    with CutSet.open_writer(args.output_dir / "long-source.jsonl.gz") as writer:
        for row in rows:
            metadata = accepted[row["call_id"]]
            if (row["split"] != "train" or row["provenance"]["speaker_channel_permutation"] != [1, 2]
                    or row["provenance"]["audio_sha256"] != metadata["source_sha256"]):
                raise ValueError("Native call differs from the accepted source or channel map")
            for cut in call_cuts(row, args.export_root / "audio", args.system_file.read_text().strip()):
                if cut.channel != 0:
                    continue
                cut.custom.pop("history")
                cut.custom.update(metadata, text_provenance="native_gigaam_sip_rnnt_pseudo",
                                  asr_model_sha256=row["provenance"]["asr_model_sha256"],
                                  asr_preparer_sha256=row["provenance"]["asr_preparer_sha256"])
                writer.write(cut)
                count += 1
                seconds += cut.duration
    summary = {"source_calls": len(rows), "caller_cuts": count, "caller_cut_hours": seconds / 3600,
               "boundaries": "Existing prepare_calls_context.call_cuts, RNNT emission approximate",
               "text_quality": "native GigaAM pseudo, human accuracy unmeasured"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-calls", type=Path, help="Optional output of unchanged prepare_sip.py")
    parser.add_argument("--accepted-calls", type=Path, help="calls.jsonl from the source-event adapter")
    parser.add_argument("--system-file", type=Path)
    args = parser.parse_args()
    if args.native_calls:
        return native_cuts(args)
    rows = json.loads((args.export_root / "selection.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seen, accepted, quarantined = set(), [], []
    counts, seconds = Counter(), Counter()
    with CutSet.open_writer(args.output_dir / "short-source.jsonl.gz") as writer:
        for row in rows:
            if row["session_id"] in seen:
                raise ValueError("Repeated source session in reviewed selection")
            seen.add(row["session_id"])
            audio = args.export_root / "audio" / (row["media_id"] + ".wav")
            event = args.export_root / "source-events" / (row["media_id"] + ".json")
            if hashlib.sha256(audio.read_bytes()).hexdigest() != row["logical_audio_sha256"]:
                raise ValueError("Audio SHA256 differs from reviewed source")
            if hashlib.sha256(event.read_bytes()).hexdigest() != row["event_sha256"]:
                raise ValueError("Event SHA256 differs from exported source")
            text, reason = client_text(json.loads(event.read_text()), row)
            if reason:
                quarantined.append({"media_id": row["media_id"], "reason": reason})
                continue
            call_id = "langfuse-" + row["media_id"]
            recording = Recording.from_file(audio.resolve(), recording_id=call_id)
            if recording.sampling_rate != 8000 or recording.channel_ids != [0, 1]:
                raise ValueError("Expected original stereo 8 kHz PSTN recording")
            if recording.num_samples != row["frames"]:
                raise ValueError("Recording duration differs from audited frame count")
            metadata = {
                "source_group_id": row["session_id"], "source_call_id": call_id,
                "source_sha256": row["logical_audio_sha256"], "event_sha256": row["event_sha256"],
                "trace_id": row["trace_id"], "media_id": row["media_id"],
                "split": "train", "text_provenance": "operational_asr_pseudo",
                "source_channel": 0, "channel_role": "client", "removed_initial_transport_seed": True,
                "target_speaker_identity_verified": False, "background_speech_verified": False,
            }
            accepted.append({"audio_filepath": str(audio.resolve()), "channel": 0,
                             "duration": recording.duration, "text": text, "metadata": metadata})
            category = "short" if 0.5 <= recording.duration <= 30 else "requires_alignment"
            counts[category] += 1
            seconds[category] += recording.duration
            if category == "short":
                writer.write(MonoCut(
                    id=call_id, start=0, duration=recording.duration, channel=0, recording=recording,
                    supervisions=[SupervisionSegment(
                        id=call_id, recording_id=call_id, start=0, duration=recording.duration,
                        channel=0, text=text, language="Russian")], custom=metadata,
                ).resample(16000))
    (args.output_dir / "calls.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted))
    (args.output_dir / "quarantine.json").write_text(json.dumps(quarantined, ensure_ascii=False, indent=2) + "\n")
    summary = {"calls": counts, "hours": {k: v / 3600 for k, v in seconds.items()},
               "quarantine": dict(Counter(row["reason"] for row in quarantined)),
               "text_quality": "pseudo, accuracy unmeasured; no acoustic alignment inferred"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
