"""Execute explicit OpenRouter evaluation requests; preserve native outputs."""
import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path
from benchmark_gemini_api import ENDPOINT, billed_cost
from benchmark_langfuse_asr import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--key-file", type=Path, required=True)
    args = p.parse_args()
    key = args.key_file.read_text().strip()
    jobs = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    def run(job):
        body = job["body"]
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        target = args.output / (job["id"] + ".json")
        if target.exists():
            old = json.loads(target.read_text())
            assert old["request_sha256"] == digest
            return old
        req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            native = json.load(response)
        save(target.with_suffix(".native.json"), native)
        assert native["model"] in job.get("allowed_models", [body["model"]])
        choice, = native["choices"]
        assert choice["finish_reason"] == "stop", "Incomplete output"
        result = dict(id=job["id"], request_sha256=digest, model=native["model"],
                      text=choice["message"]["content"], cost=billed_cost(native))
        save(target, result)
        print(json.dumps(dict(id=job["id"], status="complete")), flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, jobs))
    save(args.output / "summary.json", dict(complete=len(results), cost=sum(r["cost"] for r in results)))


if __name__ == "__main__":
    main()
