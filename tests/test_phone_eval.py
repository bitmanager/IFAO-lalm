import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from prepare_phone_eval import phone_eval_cut
from test_asr_manifest import Tokenizer


def test_short_eval_audio_and_reference_are_preserved(tmp_path):
    audio = np.linspace(-0.1, 0.1, 3200, dtype=np.float32)
    sf.write(tmp_path / "short.wav", audio, 16000, subtype="FLOAT")
    row = {"path": "short.wav", "duration": "0.200", "transcription": "Ещё 2?"}
    cut = phone_eval_cut(row, 7, tmp_path, Tokenizer(), "Русский.")
    np.testing.assert_array_equal(cut.load_audio()[0], audio)
    assert cut.id == "phone-test-000007-asr"
    assert cut.supervisions[0].text == "Ещё 2?"
    assert cut.conversation[-1]["content"] == "Ещё 2?"
    assert "Ещё 2?" not in str(cut.conversation[:-1])
    assert cut.evaluation_only and not cut.training_eligible
    assert cut.source_split == "test" and cut.task == "asr"


def test_eval_rejects_bad_duration_and_external_path(tmp_path):
    sf.write(tmp_path / "one.wav", np.zeros(16000), 16000)
    row = {"path": "one.wav", "duration": "2.000", "transcription": "Да."}
    with pytest.raises(ValueError, match="duration differs"):
        phone_eval_cut(row, 0, tmp_path, Tokenizer(), "Русский.")
    row["path"] = str(tmp_path / "one.wav")
    with pytest.raises(ValueError, match="relative audio"):
        phone_eval_cut(row, 0, tmp_path, Tokenizer(), "Русский.")
