import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from lhotse import Recording, SupervisionSegment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from prepare_asr_manifest import asr_cut


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return str(messages)

    def __call__(self, text, **kwargs):
        return type("Tokens", (), {"input_ids": list(range(len(text)))})()


@pytest.fixture
def cut(tmp_path):
    path = tmp_path / "speech.wav"
    sf.write(path, np.ones(16000, dtype=np.float32) * 0.1, 16000)
    recording = Recording.from_file(path)
    cut = recording.to_cut()
    cut.supervisions = [SupervisionSegment(id="one", recording_id=recording.id,
        start=0, duration=1, channel=0, text="Встреча в пятницу.", language="ru")]
    return cut


def test_padding_is_removed_without_changing_audio_or_target(cut):
    padded = cut.pad(duration=2, direction="both")
    result = asr_cut(padded, Tokenizer(), "Отвечай по-русски.")
    np.testing.assert_array_equal(result.load_audio(), cut.load_audio())
    assert result.duration == 1 and result.task == "asr"
    assert result.conversation[-1]["content"] == "Встреча в пятницу."
    assert "Встреча в пятницу." not in str(result.conversation[:-1])
    assert "history" not in result.custom
    assert result.recording.id == cut.recording.id


def test_mixed_speakers_are_not_silently_unwrapped(cut):
    with pytest.raises(ValueError, match="one unmodified"):
        asr_cut(cut.mix(cut, offset_other_by=0.2), Tokenizer(), "Русский.")


def test_partial_transcript_is_rejected(cut):
    cut.supervisions[0].duration = 0.5
    with pytest.raises(ValueError, match="entire unpadded"):
        asr_cut(cut, Tokenizer(), "Русский.")
