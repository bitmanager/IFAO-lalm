import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from lhotse import CutSet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from prepare_openstt import convert
from test_asr_manifest import Tokenizer


def fixture_files(tmp_path, subset="asr_public_phone_calls_1"):
    folder = tmp_path / subset / "a" / "bc"
    folder.mkdir(parents=True)
    sf.write(folder / "sample.wav", np.ones(8000, dtype=np.float32) * 0.1, 8000)
    (folder / "sample.txt").write_text("Алло, плохо вас слышно.")
    manifest = tmp_path / "source.csv"
    with manifest.open("w") as stream:
        csv.writer(stream).writerow([f"{subset}/a/bc/sample.wav", f"{subset}/a/bc/sample.txt", 1])
    return manifest


def test_phone_text_is_target_only_and_waveform_is_unchanged(tmp_path):
    manifest = fixture_files(tmp_path)
    output = tmp_path / "cuts.jsonl.gz"
    stats = convert(manifest, tmp_path, output, Tokenizer(), "Русский.")
    cut = next(iter(CutSet.from_file(output)))
    assert stats["counts"]["accepted"] == 1 and cut.task == "asr"
    assert cut.conversation[-1]["content"] == "Алло, плохо вас слышно."
    assert "Алло" not in str(cut.conversation[:-1])
    assert "history" not in cut.custom
    original, _ = sf.read(tmp_path / "asr_public_phone_calls_1/a/bc/sample.wav")
    np.testing.assert_array_equal(cut.load_audio()[0], original)


def test_official_validation_is_rejected(tmp_path):
    manifest = fixture_files(tmp_path, "asr_calls_2_val")
    with pytest.raises(ValueError, match="training subset"):
        convert(manifest, tmp_path, tmp_path / "out.jsonl.gz", Tokenizer(), "Русский.")


def test_heldout_audio_is_excluded(tmp_path):
    manifest = fixture_files(tmp_path)
    with pytest.raises(ValueError, match="No admitted"):
        convert(manifest, tmp_path, tmp_path / "out.jsonl.gz", Tokenizer(), "Русский.", {"sample"})


def test_exact_duplicate_is_counted_once(tmp_path):
    manifest = fixture_files(tmp_path)
    manifest.write_text(manifest.read_text() * 2)
    stats = convert(manifest, tmp_path, tmp_path / "out.jsonl.gz", Tokenizer(), "Русский.")
    assert stats["counts"] == {"accepted": 1, "duplicate": 1}
