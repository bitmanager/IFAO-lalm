"""Join native call outputs to a frozen cohort and score with stock jiwer."""
import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import jiwer


def normalize(text):
    return " ".join(re.sub(r"[\W_]+", " ", text.lower().replace("ё", "е")).split())


def metrics(rows, transform):
    refs = [transform(r["gemini_text"]) for r in rows]
    result = {}
    for name, field in (("online_asr", "online_text"), ("offline_gigaam", "giga_text"),
                        ("qwen_history_v3", "qwen_history_text")):
        hypotheses = [transform(r[field]) for r in rows]
        words = jiwer.process_words(refs, hypotheses)
        chars = jiwer.process_characters(refs, hypotheses)
        result[name] = dict(calls=len(rows), wer=words.wer, cer=chars.cer,
                            reference_words=words.hits + words.substitutions + words.deletions,
                            reference_chars=chars.hits + chars.substitutions + chars.deletions,
                            substitutions=words.substitutions, deletions=words.deletions,
                            insertions=words.insertions, empty_outputs=sum(not x.strip() for x in hypotheses))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cohort", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    assert not a.output.exists(), "Use a new output directory"
    assert importlib.metadata.version("jiwer") == "4.0.0"
    spec = json.loads(a.manifest.read_text())
    rows = list(csv.DictReader(a.cohort.open()))
    expected = {(str(c["environment"]), str(c["media_id"])): c for c in spec["calls"]}
    assert len(expected) == len(spec["calls"]) == len(rows)
    outputs = {}
    for path in sorted(a.results.glob("*.json")):
        out = json.loads(path.read_text())
        key = (str(out["environment"]), str(out["media_id"]))
        assert key not in outputs and key in expected
        assert out["status"] in ("complete", "truncated"), "Unfinished inference"
        assert out["recipe"]["checkpoint_sha256"] == spec["checkpoint_sha256"]
        for field in ("system_prompt", "user_suffix", "history_mode"):
            assert out["recipe"][field] == spec[field], "Inference recipe changed"
        assert out["audio_sha256"] == expected[key]["user_sha256"]
        assert [s["source"] for s in out["segments"]] == expected[key]["segments"][:len(out["segments"])]
        assert out["transcript"] == "\n".join(s["output"]["text"] for s in out["segments"])
        outputs[key] = out
    assert set(outputs) == set(expected), "Missing calls"
    assert len({r["qwen_history_audio_sha256"] for r in rows}) == len(rows)
    for row in rows:
        out = outputs[(row["source_environment"], row["media_id"])]
        assert out["audio_sha256"] == row["qwen_history_audio_sha256"]
        row.update(qwen_history_text=out["transcript"], qwen_history_status=out["status"])
        score = jiwer.process_words(normalize(row["gemini_text"]), normalize(out["transcript"]))
        row["qwen_history_v3_wer"] = score.wer
        row["qwen_history_v3_reference_words"] = score.hits + score.substitutions + score.deletions
    report = dict(checkpoint_sha256=spec["checkpoint_sha256"], human_gold=False,
                  cohort_sha256=hashlib.sha256(a.cohort.read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                  qwen_statuses=dict(collections.Counter(r["qwen_history_status"] for r in rows)),
                  normalization="lowercase, ё→е, punctuation/underscore→spaces, whitespace collapse; no number conversion",
                  metrics={})
    for name, subset in (("all", rows), ("qwen_complete_same_subset", [r for r in rows if r["qwen_history_status"] == "complete"])):
        if subset:
            report["metrics"][name] = dict(raw=metrics(subset, lambda s: s), normalized=metrics(subset, normalize))
    a.output.mkdir()
    with (a.output / "cohort.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
