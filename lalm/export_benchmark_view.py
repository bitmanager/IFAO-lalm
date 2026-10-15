"""Reshape completed promptfoo exports for its stock side-by-side viewer.

No inference or judging: original outputs and grades are retained. Additional
named metrics separate adjudicable scores from UNCERTAIN_REFERENCE coverage.
"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path


LABELS = {
    "online_text": "Онлайн Giga", "giga_text": "Giga офлайн",
    "qwen_history_text": "Qwen · аудио + история",
    "reference": "Gemini текст → Qwen", "online": "Онлайн Giga → Qwen",
    "offline_giga": "Giga офлайн → Qwen", "audio": "Аудио → projector → Qwen",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--description", required=True)
    p.add_argument("--expected-cases", required=True, type=int)
    p.add_argument("--review-queue", type=Path, help="Optional existing human-review queue; filter by task ID")
    p.add_argument("--bucket", action="append", help="Optional existing eligibility bucket; repeat to combine")
    a = p.parse_args()
    assert not a.output.exists(), "Use a new output; never overwrite a previous run"
    documents = [json.loads(path.read_text()) for path in a.inputs]
    review_queue = {str(t["task_id"]): t for t in json.loads(a.review_queue.read_text())["tasks"]} if a.review_queue else None
    cells, case_vars, branches = {}, {}, []
    sources = []
    for path, doc in zip(a.inputs, documents):
        assert doc["results"]["stats"]["errors"] == 0, "Resolve technical errors before import"
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "eval_id": doc.get("evalId")})
        for original in doc["results"]["results"]:
            if a.bucket and original.get("metadata", {}).get("eligibility_bucket") not in a.bucket:
                continue
            row = copy.deepcopy(original)
            case, branch = row["testCase"]["description"].rsplit("/", 1)
            task_id = case.removeprefix("fullwav-")
            if review_queue is not None and task_id not in review_queue:
                continue
            assert branch in LABELS and (case, branch) not in cells
            if branch not in branches:
                branches.append(branch)
            variables = copy.deepcopy(row["vars"])
            assert case_vars.setdefault(case, variables) == variables, "Branch inputs must match"
            meta = row.setdefault("metadata", {})
            meta.update(source_eval_id=doc.get("evalId"), source_result_id=row.pop("id", None))
            if review_queue is not None:
                meta["review_selection"] = review_queue[task_id]["reasons"]
                meta["label_studio_url"] = review_queue[task_id]["label_studio_url"]
            cells[case, branch] = row
    cases = sorted(case_vars)
    assert len(cases) == a.expected_cases
    assert set(cells) == {(case, branch) for case in cases for branch in branches}, "Incomplete matrix"
    out = copy.deepcopy(documents[0])
    out.pop("evalId", None)
    out["shareableUrl"] = None
    out["config"] = {"description": "SAVED REPORT · " + a.description, "providers": ["echo"],
                     "prompts": [], "tests": [], "tags": {"protocol": "view-v2-abstain"}}
    out["metadata"] = {"sources": sources, "grades_unchanged": True,
                       "pass_rate_is_conservative": True, "human_gold": False,
                       "eligibility_buckets": a.bucket,
                       "review_queue_sha256": hashlib.sha256(a.review_queue.read_bytes()).hexdigest() if a.review_queue else None}
    out["vars"] = ["case_id", "reference"]
    if review_queue is not None:
        out["vars"].append("review_reason")
    prompts, results = [], []
    summary = {"description": a.description, "cases": len(cases), "sources": sources,
               "branches": {}, "slices": {}, "paired": {}}
    judgments = {}
    for bi, branch in enumerate(branches):
        prompt_id = hashlib.sha256((a.description + "/" + branch).encode()).hexdigest()
        prompt = {"id": prompt_id, "raw": "{{input}}", "label": LABELS[branch], "provider": "echo"}
        m = dict(score=0, testPassCount=0, testFailCount=0, testErrorCount=0,
                 assertPassCount=0, assertFailCount=0, totalLatencyMs=0, cost=0,
                 namedScores={}, namedScoresCount={}, namedScoreWeights={})
        totals = collections.defaultdict(collections.Counter)
        for ci, case in enumerate(cases):
            row = cells[case, branch]
            metadata = row["metadata"]
            bucket = metadata.get("eligibility_bucket", "all")
            condition = json.loads(case_vars[case]["input"])
            variables = {"case_id": case, "reference": condition.get("REFERENCE", condition.get("current_user_reference"))}
            assert isinstance(variables["reference"], str)
            metadata["source_vars"] = case_vars[case]
            if review_queue is not None:
                variables["review_reason"] = json.dumps(metadata["review_selection"], ensure_ascii=False)
            if condition.get("history"):
                variables["history"] = json.dumps(condition["history"], ensure_ascii=False)
                if "history" not in out["vars"]:
                    out["vars"].append("history")
            shared_test = {"description": case, "vars": variables, "assert": [{
                "type": "javascript", "value": "(() => { throw new Error('Saved report only: use the benchmark recipe to evaluate a checkpoint.'); })()"}]}
            if bi == 0:
                out["config"]["tests"].append(shared_test)
            row.update(testIdx=ci, promptIdx=bi, promptId=prompt_id, prompt=prompt,
                       provider={"id": "echo", "label": LABELS[branch]},
                       vars=variables, testCase=shared_test)
            review = []
            for component in row["gradingResult"]["componentResults"]:
                metric = component["assertion"]["metric"]
                uncertain = "UNCERTAIN_REFERENCE" in component.get("reason", "")
                verdict = "review" if uncertain else "pass" if component["pass"] else "fail"
                judgments[case, branch, metric] = verdict
                totals[metric][verdict] += 1
                sl = summary["slices"].setdefault(bucket, {}).setdefault(branch, {}).setdefault(metric, {})
                sl[verdict] = sl.get(verdict, 0) + 1
                row["namedScores"][metric + "_coverage"] = int(not uncertain)
                if uncertain:
                    review.append(metric)
                else:
                    row["namedScores"][metric + "_judged"] = int(component["pass"])
            metadata["needs_review"] = review
            m["score"] += row["score"]
            m["testPassCount"] += bool(row["success"])
            m["testFailCount"] += not row["success"]
            m["totalLatencyMs"] += row.get("latencyMs", 0)
            m["cost"] += row.get("cost", 0) or 0
            for component in row["gradingResult"]["componentResults"]:
                m["assertPassCount"] += bool(component["pass"])
                m["assertFailCount"] += not component["pass"]
            for metric, score in row["namedScores"].items():
                m["namedScores"][metric] = m["namedScores"].get(metric, 0) + score
                m["namedScoresCount"][metric] = m["namedScoresCount"].get(metric, 0) + 1
                m["namedScoreWeights"][metric] = m["namedScoreWeights"].get(metric, 0) + 1
            results.append(row)
        for metric, counts in totals.items():
            known = counts["pass"] + counts["fail"]
            counts.update(total=len(cases), adjudicable=known)
            counts["pass_rate_adjudicable"] = counts["pass"] / known if known else None
            counts["coverage"] = known / len(cases)
        summary["branches"][branch] = dict(totals)
        prompts.append({**prompt, "metrics": m})
        out["config"]["prompts"].append({"raw": "{{input}}", "label": LABELS[branch]})
    baseline, candidate = ("giga_text", "qwen_history_text") if "giga_text" in branches else ("offline_giga", "audio")
    for metric in summary["branches"][candidate]:
        paired = collections.Counter()
        for case in cases:
            left, right = judgments[case, baseline, metric], judgments[case, candidate, metric]
            status = "review" if "review" in (left, right) else "tie" if left == right else "win" if right == "pass" else "loss"
            paired[status] += 1
        summary["paired"][metric] = dict(paired)
    out["results"].update(prompts=prompts, results=results)
    out["results"]["stats"] = {"successes": sum(bool(r["success"]) for r in results),
                               "failures": sum(not r["success"] for r in results), "errors": 0}
    with a.output.open("x") as f:
        json.dump(out, f, ensure_ascii=False)
    with a.output.with_suffix(".summary.json").open("x") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"cases": len(cases), "columns": branches, "cells": len(results)}))


if __name__ == "__main__":
    main()
