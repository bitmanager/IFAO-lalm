import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from lhotse import CutSet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from prepare_foreground_asr import convert
from test_asr_manifest import Tokenizer


def source(tmp_path, split="validation"):
    sf.write(tmp_path / "mix.wav", np.ones(16000, dtype=np.float32) * 0.1, 16000)
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"id": "pair-mixture", "source_group_id": "pair", "split": split,
        "target_policy": "first_active_voice", "audio_path": "/mnt/local/drive1/mix.wav",
        "target_text": "Перенеси встречу на завтра.", "background_text": "Суп уже готов.",
        "duration": 1.0, "condition": "mixture", "snr_db_requested": -3}))
    return path


def test_foreground_reference_and_background_do_not_leak_into_prompt(tmp_path):
    convert(source(tmp_path), tmp_path / "out", tmp_path, Tokenizer(), "Русский.")
    paths = list((tmp_path / "out").glob("*.jsonl.gz"))
    assert len(paths) == 2
    for path in paths:
        cut = next(iter(CutSet.from_file(path)))
        assert cut.conversation[-1]["content"] == "Перенеси встречу на завтра."
        assert "Перенеси" not in str(cut.conversation[:-1])
        assert "Суп" not in str(cut.conversation)
        assert cut.source_group_id == "pair" and cut.custom["split"] == "validation"
        assert cut.load_audio().shape == (1, 16000)


def test_train_cannot_be_mislabeled_as_heldout(tmp_path):
    with pytest.raises(ValueError, match="held-out"):
        convert(source(tmp_path, "train"), tmp_path / "out", tmp_path, Tokenizer(), "Русский.")
