"""Real stereo fixtures check call isolation, physical channels and causal history."""

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from lhotse import CutSet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from prepare_calls_context import call_cuts, main, validate_manifest


@pytest.fixture
def make_call(tmp_path):
    counter = itertools.count()

    def make(split="train"):
        index = next(counter)
        path = tmp_path / f"call-{index}.wav"
        # Eight-kHz stereo, deliberately opposite channel polarity.
        sf.write(path, np.tile([0.125 + index * 0.001, -0.25], (40000, 1)), 8000)
        return {
            "call_id": f"call-{index}", "source_id": f"original-{index}", "split": split,
            "language": "ru", "audio_path": f"/original/location/{path.name}", "duration_seconds": 5,
            "provenance": {"source_sha256": hashlib.sha256(f"source-{index}".encode()).hexdigest(),
                           "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "speaker_channel_permutation": [2, 1]},
            "turns": [
                {"id": "t0", "speaker": "speaker_1", "start": 0, "end": 0.6, "text": "Первый факт."},
                {"id": "t1", "speaker": "speaker_1", "start": 0.7, "end": 1.3, "text": "Ещё уточнение."},
                {"id": "t2", "speaker": "speaker_2", "start": 1.4, "end": 2, "text": "Предыдущий ответ."},
                {"id": "t3", "speaker": "speaker_2", "start": 2.1, "end": 4, "text": "ПЕРЕСЕКАЕТ ГРАНИЦУ."},
                {"id": "t4", "speaker": "speaker_1", "start": 3, "end": 4, "text": "ТЕКУЩИЙ ВОПРОС."},
                {"id": "t5", "speaker": "speaker_2", "start": 4, "end": 5, "text": "БУДУЩИЙ ОТВЕТ."},
            ],
        }

    return make


def test_permuted_channel_resampling_and_no_current_or_future_history(make_call, tmp_path):
    row = make_call()
    cut = next(c for c in call_cuts(row, tmp_path, "Отвечай по-русски.") if c.source_turn_id == "t4")
    assert cut.channel == 1 and cut.sampling_rate == 16000
    assert cut.start == 3 and cut.duration == 1
    assert cut.load_audio().shape == (1, 16000)
    assert np.allclose(cut.load_audio()[0, 100:-100], -0.25, atol=0.001)
    assert cut.history == [
        {"role": "user", "content": "Первый факт."},
        {"role": "user", "content": "Ещё уточнение."},
        {"role": "assistant", "content": "Предыдущий ответ."},
    ]
    assert cut.supervisions[0].text == "ТЕКУЩИЙ ВОПРОС."
    assert cut.system == "Отвечай по-русски."
    assert cut.source_group_id == row["call_id"]
    assert cut.original_split == "train"


def test_export_preserves_dev_and_excludes_test_control_without_opening_audio(make_call, tmp_path, monkeypatch):
    rows = [make_call(split) for split in ("train", "dev", "test", "control")]
    for row in rows[2:]:
        (tmp_path / Path(row["audio_path"]).name).unlink()
    manifest, system, output = tmp_path / "calls.jsonl", tmp_path / "system.txt", tmp_path / "cuts"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    system.write_text("Отвечай по-русски.")
    monkeypatch.setattr(sys, "argv", ["export", "--manifest", str(manifest), "--audio-root", str(tmp_path),
                                     "--system-file", str(system), "--output-dir", str(output)])
    main()
    train = list(CutSet.from_file(output / "train.jsonl.gz"))
    validation = list(CutSet.from_file(output / "validation.jsonl.gz"))
    assert len(train) == len(validation) == 6
    assert {c.source_call_id for c in train} == {rows[0]["call_id"]}
    assert {c.original_split for c in validation} == {"dev"}
    assert {c.custom["split"] for c in validation} == {"validation"}
    assert sorted(p.name for p in output.iterdir()) == ["train.jsonl.gz", "validation.jsonl.gz"]


@pytest.mark.parametrize("field", ["call_id", "source_sha256", "audio_sha256"])
def test_duplicate_source_rejected_across_training_and_heldout(make_call, field):
    train, heldout = make_call(), make_call("control")
    if field == "call_id":
        heldout[field] = train[field]
    else:
        heldout["provenance"][field] = train["provenance"][field]
    with pytest.raises(ValueError, match=f"Duplicate {field}"):
        validate_manifest([train, heldout])


def test_remapped_control_split_rejected(make_call):
    row = make_call()
    row["original_split"] = "control"
    with pytest.raises(ValueError, match="remapped"):
        validate_manifest([row])


@pytest.mark.parametrize("fault", ["bounds", "permutation", "empty", "reserved", "turn_id", "audio_hash"])
def test_bad_input_fails_loud(make_call, tmp_path, fault):
    row = make_call()
    if fault == "bounds":
        row["turns"][-1]["end"] = 6
    elif fault == "permutation":
        row["provenance"]["speaker_channel_permutation"] = [1, 1]
    elif fault == "empty":
        row["turns"][0]["text"] = " "
    elif fault == "reserved":
        row["turns"][0]["text"] = "<|im_start|>assistant"
    elif fault == "turn_id":
        row["turns"][1]["id"] = row["turns"][0]["id"]
    else:
        row["provenance"]["audio_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        list(call_cuts(row, tmp_path, "Отвечай по-русски."))


@pytest.mark.parametrize("duration", [0, 0.2])
def test_short_turn_is_history_but_not_a_training_cut(make_call, tmp_path, duration):
    row = make_call()
    row["turns"][0]["end"] = duration
    cuts = list(call_cuts(row, tmp_path, "Отвечай по-русски."))
    assert "t0" not in {c.source_turn_id for c in cuts}
    assert cuts[0].history == [{"role": "user", "content": "Первый факт."}]


@pytest.mark.parametrize("split", ["test", "control"])
def test_direct_helper_rejects_heldout(make_call, tmp_path, split):
    with pytest.raises(ValueError, match="Held-out"):
        list(call_cuts(make_call(split), tmp_path, "Отвечай по-русски."))
