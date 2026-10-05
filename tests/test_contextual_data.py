"""Check context/target isolation with the real Qwen tokenizer and Lhotse cuts."""

import json
import gzip
import os
import sys
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from prepare_multitalk_context import dialogue_cuts, main as export_dialogues
from lhotse import CutSet
from prepare_conversation import prepare_cut, prepare_asr_cut, teacher_messages
from lalm_core.model.processing_lalm import LALMProcessor


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(os.environ["QWEN_MODEL_PATH"])
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|audio|>"]})
    return tokenizer


@pytest.fixture
def cuts(tmp_path):
    # Channel 0 and 1 differ so slicing the wrong channel cannot pass unnoticed.
    audio = np.stack([np.full(64000, 0.1), np.full(64000, 0.2)], axis=1)
    sf.write(tmp_path / "dialog.wav", audio, 16000, subtype="FLOAT")
    data = {
        "id": "dialog", "split": "train",
        "source_script": {"source_group_id": "conversation-1", "participants": [
            {"name": "U", "role": "user"}, {"name": "A", "role": "assistant"}]},
        "segments": [
            {"participant": "U", "channel": 1, "start": 0, "end": 1, "text": "Номер на пятницу."},
            {"participant": "A", "channel": 0, "start": 1, "end": 2, "text": "На сколько ночей?"},
            {"participant": "U", "channel": 1, "start": 2, "end": 2.6, "text": "На две."},
            {"participant": "U", "channel": 1, "start": 2.7, "end": 3.4, "text": "С завтраком."},
            {"participant": "A", "channel": 0, "start": 3.4, "end": 4, "text": "БУДУЩИЙ ОТВЕТ"},
        ],
    }
    path = tmp_path / "dialog.json"
    path.write_text(json.dumps(data))
    return list(dialogue_cuts(path, "Отвечай по-русски."))


def test_history_audio_bounds_and_teacher_parity(cuts, tokenizer):
    first, second = cuts
    assert first.history == []
    assert second.start == 2 and second.duration == pytest.approx(1.4)
    assert np.allclose(second.load_audio(), 0.2)
    teacher = teacher_messages(second)
    assert teacher[-1]["content"] == "На две. С завтраком."
    assert "БУДУЩИЙ ОТВЕТ" not in json.dumps(teacher, ensure_ascii=False)
    assert len(second.history) == 2
    second.supervisions[0].custom = {"answer": "На две ночи, с завтраком. Назовите имя."}
    prepared = prepare_cut(second, tokenizer)
    assert prepared.conversation[:-2] == teacher[:-1]
    assert prepared.conversation[-2]["content"][0]["type"] == "audio"
    assert prepared.rendered_conversation.count("<|audio|>") == 1
    assert "На две. С завтраком." not in prepared.rendered_conversation


def test_only_current_answer_has_loss(cuts, tokenizer):
    cut = cuts[1]
    answer = "На две ночи, с завтраком. Назовите имя."
    cut.supervisions[0].custom = {"answer": answer}
    prepared = prepare_cut(cut, tokenizer)
    batch = tokenizer([prepared.rendered_conversation, "коротко"], padding=True, return_tensors="pt")
    # Exercise the upstream mask method directly, without constructing an audio tower.
    processor = object.__new__(LALMProcessor)
    processor.tokenizer = tokenizer
    labels = processor._prepare_labels(batch.input_ids, batch.attention_mask)
    assert tokenizer.decode(labels[0][labels[0] != -100]) == answer + "<|im_end|>"
    assert torch.all(labels[1] == -100)
    assert torch.all(labels[batch.attention_mask == 0] == -100)


def test_no_silent_asr_target_or_prompt_override(cuts, tokenizer):
    with pytest.raises(ValueError, match="teacher answer"):
        prepare_cut(cuts[0], tokenizer)
    with pytest.raises(ValueError, match="teacher answer"):
        prepare_cut(cuts[1], tokenizer)
    with pytest.raises(ValueError, match="system prompt"):
        teacher_messages(cuts[1], "Другой промпт")


def test_asr_target_is_separate_from_agent_context(cuts, tokenizer):
    cut = cuts[1]
    cut.supervisions[0].custom = {"answer": "Назовите имя."}
    agent = prepare_cut(cut, tokenizer)
    asr = prepare_asr_cut(agent, tokenizer)
    assert asr.task == "asr" and asr.id != agent.id
    assert asr.conversation[-1]["content"] == cut.supervisions[0].text
    assert agent.conversation[-1]["content"] == "Назовите имя."
    assert asr.conversation[:-2] == agent.conversation[:-2]
    assert cut.supervisions[0].text not in json.dumps(asr.conversation[:-1], ensure_ascii=False)


def test_export_uses_recording_split_and_ignores_completion_sidecar(cuts, tmp_path, monkeypatch):
    source = Path(cuts[0].recording.sources[0].source)
    folder = tmp_path / "rendered/multichannel_wavs/di"
    folder.mkdir(parents=True)
    shutil.copy(source, folder / "dialog.wav")
    shutil.copy(source.with_suffix(".json"), folder / "dialog.json")
    (folder / "dialog.complete.json").write_text('{"status": "alignment_pending"}')
    system = tmp_path / "system.txt"
    system.write_text("Отвечай по-русски.")
    output = tmp_path / "cuts"
    monkeypatch.setattr(sys, "argv", ["export", "--metadata-root", str(tmp_path / "rendered"),
                                     "--system-file", str(system), "--output-dir", str(output)])
    export_dialogues()
    exported = list(CutSet.from_file(output / "train.jsonl.gz"))
    assert len(exported) == 2
    assert {c.custom["split"] for c in exported} == {"train"}
    with gzip.open(output / "validation.jsonl.gz", "rt") as stream:
        assert stream.read() == ""
    assert np.allclose(exported[1].load_audio(), 0.2)
