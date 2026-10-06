import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from lhotse import CutSet, Recording, SupervisionSegment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from prepare_context_overlap import CONTEXT_TARGET_POLICY, FIRST_VOICE_POLICY, task_views, validate_pairs
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


@pytest.mark.parametrize("target_policy", ["first_active_voice", "context_target"])
def test_clean_and_mixed_tasks_preserve_history_and_separate_targets(tmp_path, target_policy):
    fg = source(tmp_path, "foreground", "train", "group-a")
    original = prepare_asr_cut(fg, Tokenizer()).conversation[-2]["content"][1]["text"]
    for condition in ("clean", "mixture"):
        example = {"id": "pair-" + condition, "source_group_id": "pair", "duration": 1,
            "audio_path": fg.recording.sources[0].source, "condition": condition,
            "target_policy": target_policy,
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
        policy = FIRST_VOICE_POLICY if target_policy == "first_active_voice" else CONTEXT_TARGET_POLICY
        assert (policy in answer.rendered_conversation) == (condition == "mixture")
        assert (policy in asr.rendered_conversation) == (condition == "mixture")
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


def test_unknown_phone_identity_requires_source_bound_opt_in_review(tmp_path):
    fg, bg = source(tmp_path, "phone", "train", "call"), source(tmp_path, "tts", "train", "script")
    fg.custom.update(source_call_id="original-call", source_sha256="original-audio-hash")
    metadata = tmp_path / "tts.json"
    metadata.write_text(json.dumps({"speaker_to_channel": {"person": 0}, "voice": {"person": "two"}}))
    row = {"id": "unknown", "split": "train", "foreground_cut_id": fg.id,
        "background_cut_id": bg.id, "foreground_voice": None, "background_voice": "two",
        "foreground_speaker_identity_verified": False, "foreground_source_channel": 0,
        "background_topic": "plant care", "background_voice_metadata_path": str(metadata),
        "background_voice_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "semantic_review": {"method": "agent_full_grouped_transcript_review", "unrelated": True,
            "source_call_id": "original-call", "source_audio_sha256": "original-audio-hash",
            "background_cut_id": bg.id, "note": "Debt discussion versus plant care; agent review, not gold."}}
    cuts = {fg.id: fg, bg.id: bg}
    with pytest.raises(ValueError, match="Unknown foreground"):
        validate_pairs([row], cuts)
    validate_pairs([row], cuts, allow_unknown_foreground_speaker=True)
    for key, value in [("source_call_id", "another-call"), ("source_audio_sha256", "changed"),
                       ("background_cut_id", "another-background"), ("unrelated", False)]:
        changed = {**row, "semantic_review": {**row["semantic_review"], key: value}}
        with pytest.raises(ValueError, match="Unknown foreground"):
            validate_pairs([changed], cuts, allow_unknown_foreground_speaker=True)
    example = {"id": "unknown-clean", "source_group_id": "unknown", "duration": 1,
               "audio_path": fg.recording.sources[0].source, "condition": "clean"}
    for cut in task_views(fg, example, Tokenizer(), row):
        assert cut.speaker_identity_verified is False
        assert cut.distinct_voice_identity_verified is False
        assert cut.semantic_review == row["semantic_review"]
        assert row["semantic_review"]["note"] not in cut.rendered_conversation


@pytest.mark.parametrize("duplicate_waveforms", [False, True])
@pytest.mark.parametrize("context_target", [False, True])
def test_native_export_writes_string_splits_and_both_tasks(tmp_path, monkeypatch, duplicate_waveforms, context_target):
    cuts, recipe = [], []
    for split in ("train", "validation"):
        for role in ("foreground", "background"):
            name = split + "-" + role
            c = source(tmp_path, name, split, name)
            if duplicate_waveforms:
                sf.write(c.recording.sources[0].source, np.ones(16000) * .05, 16000)
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
    argv = ["adapter", "--source-manifests", str(manifest),
        "--recipe", str(pairs), "--output", str(output), "--tokenizer", "unused"]
    if context_target:
        mask = []
        for row in recipe:
            qc = {"pair_id": row["id"], "split": row["split"], "clean_qc_candidate": True}
            for role in ("foreground", "background"):
                cid = row[role + "_cut_id"]
                data = sf.read(tmp_path / (cid + ".wav"), dtype="float32")[0]
                qc[role + "_source_cut_id"] = cid
                qc[role + "_pcm_float32_le_sha256"] = hashlib.sha256(data.astype("<f4").tobytes()).hexdigest()
            mask.append(qc)
        mask_path = tmp_path / "mask.jsonl"
        mask_path.write_text("".join(json.dumps(r) + "\n" for r in mask))
        argv += ["--context-target", "--clean-qc-mask", str(mask_path)]
    monkeypatch.setattr(sys, "argv", argv)
    if duplicate_waveforms:
        # Different container hashes must not hide identical decoded samples.
        monkeypatch.setattr(prepare_context_overlap, "digest", lambda path: str(path))
        with pytest.raises(ValueError, match="Identical source waveform crosses splits"):
            prepare_context_overlap.main()
        return
    prepare_context_overlap.main()
    for split in ("train", "validation"):
        exported = list(CutSet.from_file(output / (split + ".jsonl.gz")))
        assert len(exported) == (14 if context_target else 8)
        assert {c.custom["split"] for c in exported} == {split}
        assert {c.task for c in exported} == {"answer", "asr"}
    examples = [json.loads(x) for x in (output / "mixed/pilot.jsonl").read_text().splitlines()]
    for row in examples:
        if row["condition"] == "clean":
            continue
        first = row["foreground_first_active_seconds"]
        second = row["background_first_active_seconds"]
        assert (first < second) == (row["onset_order"] == "foreground-first")
        assert abs(first - second) >= .25999
        energies = []
        for role, path_key in (("foreground", "target_audio_path"), ("background", "interference_audio_path")):
            original = sf.read(tmp_path / (row["split"] + "-" + role + ".wav"), dtype="float32")[0]
            stem = sf.read(row[path_key], dtype="float32")[0]
            offset = round(row[role + "_offset_seconds"] * 16000)
            aligned = stem[offset:offset + original.size]
            scale = np.dot(aligned, original) / np.dot(original, original)
            assert np.allclose(aligned, original * scale, atol=1e-6)
            energies.append(np.mean(aligned ** 2))
        assert 10 * np.log10(energies[0] / energies[1]) == pytest.approx(row["snr_db_requested"], abs=1e-4)
