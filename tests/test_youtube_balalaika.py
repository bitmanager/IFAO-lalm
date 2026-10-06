import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
from lhotse import CutSet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
import prepare_youtube_balalaika as adapter
from test_asr_manifest import Tokenizer


def test_excludes_entire_heldout_source_and_preserves_asr_target(tmp_path, monkeypatch):
    shards = tmp_path / "shards"
    shards.mkdir()
    wav = io.BytesIO()
    sf.write(wav, np.ones(16000, dtype=np.float32) * 0.1, 16000, format="FLAC")
    with tarfile.open(shards / "one.tar", "w") as archive:
        for key, podcast in [("one", "train-source"), ("two", "eval-source")]:
            for suffix, data in [("json", json.dumps({"podcast_id": podcast, "punct": "Плохо слышно."}).encode()),
                                 ("flac", wav.getvalue())]:
                info = tarfile.TarInfo(f"{key}.{suffix}")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    (tmp_path / "heldout.json").write_text('[{"voice":"eval-source:0"}]')
    (tmp_path / "system.txt").write_text("Русский.")
    monkeypatch.setattr(adapter.AutoTokenizer, "from_pretrained", lambda _: Tokenizer())
    monkeypatch.setattr(sys, "argv", ["prepare", "--shards", str(shards), "--heldout", str(tmp_path / "heldout.json"),
        "--output-dir", str(tmp_path / "out"), "--tokenizer", "stub", "--system-file", str(tmp_path / "system.txt")])
    adapter.main()
    cuts = list(CutSet.from_file(tmp_path / "out/train-asr.jsonl.gz"))
    assert len(cuts) == 1 and cuts[0].source_recording == "train-source"
    assert cuts[0].conversation[-1]["content"] == "Плохо слышно."
    assert "history" not in cuts[0].custom
    assert not (tmp_path / "out/audio/one/two.flac").exists()
