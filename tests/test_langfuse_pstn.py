import importlib.util
import json
import hashlib
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_langfuse_pstn", Path(__file__).parents[1] / "lalm/prepare_langfuse_pstn.py"
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def event(turns):
    return [{"body": {"id": "trace", "sessionId": "session", "output": json.dumps({
        "recording": "@@@langfuseMedia:id=media@@@", "transcript": turns,
    })}}]


ROW = {"media_id": "media", "trace_id": "trace", "session_id": "session"}


def test_seed_is_not_audio_and_agent_text_is_not_client_text():
    source = event([adapter.SEED, {"role": "agent", "text": "Здравствуйте"},
                    {"role": "client", "text": "Я вас слышу"}])
    assert adapter.client_text(source, ROW) == ("Я вас слышу", None)
    assert json.loads(source[0]["body"]["output"])["transcript"][0] == adapter.SEED


def test_seed_only_is_not_russian_audio_supervision():
    assert adapter.client_text(event([adapter.SEED]), ROW) == (
        None, "no_cyrillic_client_text_after_seed"
    )


def test_missing_asr_quarantines_whole_call():
    assert adapter.client_text(event([adapter.SEED, {"role": "client", "text": "Алло"},
                                      {"role": "client", "text": adapter.MISSING}]), ROW) == (
        None, "missing_asr_marker"
    )


def test_wrong_trace_cannot_supply_transcript():
    with pytest.raises(ValueError, match="identities"):
        adapter.client_text(event([adapter.SEED]), {**ROW, "trace_id": "other"})


def test_native_adapter_preserves_existing_bounds_and_selects_caller(tmp_path, monkeypatch):
    from lhotse import CutSet

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "lalm"))
    (tmp_path / "audio").mkdir()
    audio = tmp_path / "audio" / "call.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0" * 5 * 8000 * 4)
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    raw = {"call_id": "call", "language": "ru", "split": "train", "audio_path": str(audio),
           "duration_seconds": 5, "provenance": {
               "source_sha256": digest, "audio_sha256": digest, "speaker_channel_permutation": [1, 2],
               "asr_model_sha256": "a" * 64, "asr_preparer_sha256": "b" * 64},
           "turns": [{"id": "t0", "start": 0.5, "end": 2.5, "speaker": "speaker_1", "text": "Клиент"},
                     {"id": "t1", "start": 2.5, "end": 4.5, "speaker": "speaker_2", "text": "Агент"}]}
    native = tmp_path / "native.jsonl"
    native.write_text(json.dumps(raw) + "\n")
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text(json.dumps({"metadata": {"source_call_id": "call", "source_sha256": digest}}) + "\n")
    system = tmp_path / "system.txt"
    system.write_text("Расшифруй речь")
    output = tmp_path / "output"
    adapter.native_cuts(SimpleNamespace(system_file=system, accepted_calls=accepted,
                                        native_calls=native, output_dir=output, export_root=tmp_path))
    cuts = list(CutSet.from_file(output / "long-source.jsonl.gz"))
    assert len(cuts) == 1
    cut = cuts[0]
    assert (cut.start, cut.duration, cut.channel, cut.supervisions[0].text) == (0.5, 2, 0, "Клиент")
    assert "history" not in cut.custom
    assert cut.custom["boundary_source"] == "rnnt_emission_approximate"
