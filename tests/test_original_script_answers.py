"""Source-bound script answer export must not leak or silently repair labels."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from export_original_script_answers import original_answer
from prepare_conversation import teacher_messages
from prepare_multitalk_context import dialogue_cuts


@pytest.fixture
def source(tmp_path):
    sf.write(tmp_path / "dialog.wav", np.zeros((64000, 2)), 16000)
    segments = [
        {"participant": "U", "channel": 1, "start": 0., "end": 1., "text": "Нужен билет."},
        {"participant": "A", "channel": 0, "start": .9, "end": 2., "text": "На какой день?"},
        {"participant": "U", "channel": 1, "start": 1.9, "end": 3., "text": "На пятницу."},
        {"participant": "A", "channel": 0, "start": 2.9, "end": 4., "text": "Оригинальный ответ про пятницу."},
    ]
    metadata = {"id": "dialog", "split": "train", "speaker_to_channel": {"U": 1, "A": 0},
                "segments": segments, "source_script": {"source_group_id": "group",
                "participants": [{"name": "U", "role": "user"}, {"name": "A", "role": "assistant"}],
                "dialogue": [{"speaker": t["participant"], "text": t["text"]} for t in segments]}}
    path = tmp_path / "dialog.json"
    path.write_text(json.dumps(metadata))
    native = {c.id: c for c in dialogue_cuts(path, "Отвечай кратко.")}
    cut = copy.deepcopy(native["dialog-turn-0002"])
    cut.supervisions[0].custom = {"answer": "Старый sampled teacher ответ."}
    return cut, path, metadata, native


def test_exact_script_target_with_native_prompt_and_overlap(source):
    cut, path, metadata, native = source
    response, provenance = original_answer(cut, path, metadata, native)
    assert set(response) == {"idx", "messages", "response"}
    assert response["messages"] == teacher_messages(cut)
    assert response["response"] == "Оригинальный ответ про пятницу."
    assert response["response"] not in json.dumps(response["messages"], ensure_ascii=False)
    assert cut.supervisions[0].custom["answer"] == "Старый sampled teacher ответ."
    assert provenance["answer_onset_overlap_seconds"] == pytest.approx(.1)


@pytest.mark.parametrize("failure", ["missing_next", "script_text", "channel", "history"])
def test_reject_inconsistent_sources(source, failure):
    cut, path, metadata, native = source
    if failure == "missing_next":
        metadata["segments"].pop()
        metadata["source_script"]["dialogue"].pop()
    elif failure == "script_text":
        metadata["source_script"]["dialogue"][-1]["text"] = "Другой ответ."
    elif failure == "channel":
        metadata["segments"][-1]["channel"] = 1
    else:
        cut.custom["history"].append({"role": "assistant", "content": "Будущий ответ."})
    with pytest.raises(ValueError):
        original_answer(cut, path, metadata, native)
