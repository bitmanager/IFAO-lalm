import csv
import hashlib
import io
import json
import sys
import wave
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lhotse import CutSet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from prepare_svq_eval import DATASET, REVISION, ENVIRONMENTS, prepare, select_pilot


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return str(messages)

    def __call__(self, text, **kwargs):
        return type("Tokens", (), {"input_ids": list(range(len(text)))})()


def source(tmp_path):
    root = tmp_path / "source"
    (root / "2.0.0").mkdir(parents=True)
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 4000)
    data, files, rows = stream.getvalue(), [], []
    for group, environment in enumerate(ENVIRONMENTS):
        records = []
        for number in range(2):
            row = dict(utt_id=f"utt_{group * 2 + number}", speaker_id=f"speaker_{number}",
                speaker_age=30, speaker_gender="female", locale="ru_ru",
                environment=environment, text="Исходный вопрос?", transcript_truth="Произнесённый текст!",
                **{"transcriptions/speech": True})
            for name in ("page_id", "passage_id", "span_context_id"):
                for language in ("in_lang", "cross_lang"):
                    row[f"{name}_{language}"] = f"{name}-original"
            records.append({**row, "waveform": {"bytes": data}})
            rows.append(row)
        path = root / "2.0.0" / f"utts_ru_ru_{environment}.parquet"
        pq.write_table(pa.Table.from_pylist(records), path)
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                      "lfs_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (root / "source-manifest.json").write_text(json.dumps({"repository": DATASET,
        "revision": REVISION, "files": files, "all_lfs_hashes_verified": True,
        "evaluation_only": True, "training_eligible": False}))
    return root, data, rows


def test_truth_audio_and_holdout_metadata_survive_without_filtering(tmp_path):
    root, data, rows = source(tmp_path)
    output = tmp_path / "prepared"
    result = prepare(root, output, Tokenizer(), "Русский ассистент.", per_environment=1)
    cuts = list(CutSet.from_file(output / "full-all.jsonl.gz"))
    assert len(cuts) == 8 and result["under_0_5_seconds"] == 8
    assert all(c.duration == .25 and c.task == "asr" for c in cuts)
    assert all(c.supervisions[0].text == "Произнесённый текст!" for c in cuts)
    assert all(c.conversation[-1]["content"] == "Произнесённый текст!" for c in cuts)
    assert all(c.text == "Исходный вопрос?" and c.evaluation_only and not c.training_eligible for c in cuts)
    assert all(Path(c.recording.sources[0].source).read_bytes() == data for c in cuts)
    assert all(c.supervisions[0].speaker == c.speaker_id for c in cuts)
    assert all(c.page_id_in_lang == "page_id-original" for c in cuts)
    assert len(list(CutSet.from_file(output / "pilot-all.jsonl.gz"))) == 4
    with (output / "full-all.tsv").open() as stream:
        baseline = list(csv.DictReader(stream, delimiter="\t"))
    assert len(baseline) == 8 and all(r["transcription"] == "Произнесённый текст!" for r in baseline)
    pilot = json.loads((output / "pilot_ids.json").read_text())["utt_ids"]
    assert pilot == select_pilot(rows, 1, "svq-ru-pilot-v1")
    assert pilot == select_pilot(list(reversed(rows)), 1, "svq-ru-pilot-v1")
    assert not list(output.glob("*.incomplete.*")) and not list(output.glob("train*"))


def test_modified_source_fails_before_output_is_created(tmp_path):
    root, _, _ = source(tmp_path)
    path = next((root / "2.0.0").glob("*.parquet"))
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare(root, output, Tokenizer(), "Русский.", per_environment=1)
    assert not output.exists()


def test_pilot_cannot_silently_shrink_a_stratum(tmp_path):
    _, _, rows = source(tmp_path)
    with pytest.raises(ValueError, match="Not enough rows"):
        select_pilot(rows, 3, "fixed")
