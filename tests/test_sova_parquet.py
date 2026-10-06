import hashlib
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lhotse import CutSet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from prepare_sova_parquet import prepare, text_key


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return str(messages)

    def __call__(self, text, **kwargs):
        return type("Tokens", (), {"input_ids": list(range(len(text)))})()


def wav_bytes(value):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(np.full(16000, value, dtype="<i2").tobytes())
    return stream.getvalue()


def stage(tmp_path, rows):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "train-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    manifest = {"source": {"repo_id": "bond005/sova_rudevices", "revision": "pinned", "split": "train"},
                "files": [{"path": path.name, "rows": len(rows), "size_bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "checksum_verified": True}],
                "train_rows": len(rows), "duration_seconds": len(rows)}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_heldout_text_and_bytes_are_excluded_without_audio_rewriting(tmp_path):
    source = [wav_bytes(n) for n in (100, 200, 300)]
    rows = [{"audio": {"bytes": audio, "path": None}, "transcription": text}
            for audio, text in zip(source, ["Здесь ёлка!", "Это закрытый пример", "Нужная запись"])]
    root = stage(tmp_path, rows)
    result = prepare(root, tmp_path / "out", {"texts": ["здесь елка"], "ids": [],
                     "sha256": [hashlib.sha256(source[1]).hexdigest()]}, Tokenizer(), "Русский.")
    assert result["counts"] == {"source": 3, "kept": 1,
        "heldout_normalized_exact_text": 1, "heldout_id_or_sha256": 1}
    cut = next(iter(CutSet.from_file(result["train_manifest"])))
    assert Path(cut.recording.sources[0].source).read_bytes() == source[2]
    assert cut.task == "asr" and "history" not in cut.custom
    assert cut.supervisions[0].text == cut.conversation[-1]["content"] == "Нужная запись"
    assert cut.source["row"] == 2 and cut.source["source_audio_path"] is None
    assert cut.load_audio().shape == (1, 16000)


def test_duplicate_audio_and_source_ids_are_excluded(tmp_path):
    data = wav_bytes(100)
    rows = [{"audio": {"bytes": data, "path": None}, "transcription": "Привет"}] * 2
    rows += [{"audio": {"bytes": wav_bytes(200), "path": "/old/heldout.wav"}, "transcription": "Другой текст"}]
    result = prepare(stage(tmp_path, rows), tmp_path / "out",
                     {"texts": [], "ids": ["heldout"], "sha256": []}, Tokenizer(), "Русский.")
    assert result["counts"]["kept"] == 1
    assert result["counts"]["duplicate_audio_bytes"] == 1
    assert result["counts"]["heldout_id_or_sha256"] == 1
    assert text_key(" Ещё,  РАЗ! ") == "еще раз"


def test_conflicting_audio_and_late_heldout_variant_remove_first_copy(tmp_path):
    rows = [{"audio": {"bytes": wav_bytes(value), "path": None}, "transcription": text}
            for value, text in [(100, "Первый текст"), (200, "Ранний текст"),
                                (300, "Нужная запись"), (100, "Иной текст"),
                                (200, "Закрытая фраза")]]
    result = prepare(stage(tmp_path, rows), tmp_path / "out",
        {"texts": ["Закрытая фраза"], "ids": [], "sha256": []}, Tokenizer(), "Русский.")
    cuts = list(CutSet.from_file(result["train_manifest"]))
    assert [c.supervisions[0].text for c in cuts] == ["Нужная запись"]
    assert result["counts"]["conflicting_audio_transcripts"] == 1
    assert result["counts"]["heldout_audio_hash_closure"] == 1
