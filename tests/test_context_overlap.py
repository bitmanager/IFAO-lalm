import sys
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from lhotse import CutSet, Recording, SupervisionSegment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from prepare_context_overlap import FIRST_VOICE_POLICY, task_views, validate_pairs
from prepare_conversation import prepare_asr_cut
from test_asr_manifest import Tokenizer
import prepare_context_overlap


def source(tmp_path, name, split, group):
    path = tmp_path / (name + ".wav")
    # Distinct source waveforms exercise (rather than bypass) split deduplication.
    frequency = 100 + sum((i + 1) * ord(c) for i, c in enumerate(name)) % 600
    sf.write(path, .05 * np.sin(2 * np.pi * frequency * np.arange(16000) / 16000), 16000)
    cut = Recording.from_file(path, recording_id=name).to_cut()
    cut.supervisions = [SupervisionSegment(id=name, recording_id=name,
        start=0, duration=1, channel=0, text="Перенеси на пятницу.", language="ru",
        custom={"answer": "Встреча перенесена на пятницу."})]
    cut.custom = {"split": split, "source_group_id": group, "system": "Русский.",
        "history": [{"role": "user", "content": "Когда встреча?"},
                    {"role": "assistant", "content": "В четверг."}]}
    return cut


def test_clean_and_mixed_tasks_preserve_history_and_separate_targets(tmp_path):
    fg = source(tmp_path, "foreground", "train", "group-a")
    original = prepare_asr_cut(fg, Tokenizer()).conversation[-2]["content"][1]["text"]
    for condition in ("clean", "mixture"):
        example = {"id": "pair-" + condition, "source_group_id": "pair", "duration": 1,
            "audio_path": fg.recording.sources[0].source, "condition": condition,
            "background_text": "Чужой секретный ответ."}
        answer, asr = task_views(fg, example, Tokenizer())
        for c in (answer, asr):
            assert c.history == fg.history and c.system == fg.system
            assert "Чужой секретный ответ" not in c.rendered_conversation
            assert c.training_eligible is False
        assert answer.task == "answer" and asr.task == "asr"
        assert answer.conversation[-1]["content"] == fg.supervisions[0].custom["answer"]
        assert asr.conversation[-1]["content"] == fg.supervisions[0].text
        assert original in asr.conversation[-2]["content"][1]["text"]
        assert (FIRST_VOICE_POLICY in answer.rendered_conversation) == (condition == "mixture")
        assert (FIRST_VOICE_POLICY in asr.rendered_conversation) == (condition == "mixture")
        assert fg.supervisions[0].custom["answer"] not in str(answer.conversation[:-1])


def test_source_group_crossing_and_invented_split_are_rejected(tmp_path):
    cuts = {name: source(tmp_path, name, split, group) for name, split, group in
        [("a", "train", "shared"), ("b", "train", "b"),
         ("c", "validation", "shared"), ("d", "validation", "d")]}
    def pair(name, split, fg, bg):
        return {"id": name, "split": split, "foreground_cut_id": fg, "background_cut_id": bg,
            "foreground_voice": "one", "background_voice": "two",
            "foreground_topic": "calendar", "background_topic": "cooking"}
    with pytest.raises(ValueError, match="crosses splits"):
        validate_pairs([pair("one", "train", "a", "b"),
                        pair("two", "validation", "c", "d")], cuts)
    with pytest.raises(ValueError, match="split/group"):
        validate_pairs([pair("one", "train", "c", "b")], cuts)


def test_native_export_writes_string_splits_and_both_tasks(tmp_path, monkeypatch):
    cuts, recipe = [], []
    for split in ("train", "validation"):
        for role in ("foreground", "background"):
            name = split + "-" + role
            c = source(tmp_path, name, split, name)
            c.supervisions[0].text += " " + name
            cuts.append(c)
        recipe.append({"id": split, "split": split,
            "foreground_cut_id": split + "-foreground", "background_cut_id": split + "-background",
            "foreground_voice": "one", "background_voice": "two",
            "foreground_topic": "calendar", "background_topic": "cooking"})
    manifest = tmp_path / "cuts.jsonl.gz"
    CutSet.from_cuts(cuts).to_file(manifest)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(r) + "\n" for r in recipe))
    output = tmp_path / "export"
    monkeypatch.setattr(prepare_context_overlap.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(sys, "argv", ["adapter", "--source-manifests", str(manifest),
        "--recipe", str(pairs), "--output", str(output), "--tokenizer", "unused"])
    prepare_context_overlap.main()
    for split in ("train", "validation"):
        exported = list(CutSet.from_file(output / (split + ".jsonl.gz")))
        assert len(exported) == 8
        assert {c.custom["split"] for c in exported} == {split}
        assert {c.task for c in exported} == {"answer", "asr"}
