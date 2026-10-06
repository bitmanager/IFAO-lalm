import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from prepare_history_audio_eval import build_views
from test_asr_manifest import Tokenizer


def test_fixed_pairs_reuse_audio_preserve_history_and_isolate_labels(tmp_path):
    cases = [json.loads(x) for x in (Path(__file__).resolve().parents[1] /
             "assets/context_history_eval_cases.jsonl").read_text().splitlines()]
    paths, outputs = {}, {}
    for case in cases:
        text = case["messages"][-1]["content"]
        if text not in paths:
            paths[text] = tmp_path / (str(len(paths)) + ".wav")
            sf.write(paths[text], np.ones(16000) * .01, 16000)
        answer, asr = build_views(case, paths[text], Tokenizer())
        for c in (answer, asr):
            assert c.system == case["messages"][0]["content"]
            assert c.history == case["messages"][1:-1]
            assert c.conversation[:-2] == case["messages"][:-1]
            assert c.evaluation_only and not c.training_eligible
            assert c.custom["split"] == "test"
        assert answer.conversation[-1]["content"] == case["expected"]
        assert asr.conversation[-1]["content"] == text
        outputs[case["idx"]] = answer
    assert len(cases) == 16 and len(paths) == 6
    for stem in ("hotel", "correction", "negation", "reference", "name", "carry"):
        a, b = outputs[stem + "_a"], outputs[stem + "_b"]
        assert a.recording.sources == b.recording.sources
        assert a.history != b.history and a.expected_facts != b.expected_facts
        if stem + "_a_no_history" in outputs:
            control = outputs[stem + "_a_no_history"]
            assert control.recording.sources == a.recording.sources
            assert control.history == []
