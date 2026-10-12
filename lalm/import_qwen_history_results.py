"""Incrementally attach completed KL1532 history-ASR outputs to Label Studio tasks.

This is a format/API adapter only. It does not run models, alter annotations, or
select pseudo-labels. The canonical call manifest and import plan must match the
result's environment, media ID, call ID, checkpoint, and audio SHA-256.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


CHECKPOINT_SHA256 = "860552e6f0183bd43f27cb79691c5b8b6f2995ad009d220dc2d2b1b253ce3310"
COMPLETE = "complete"
PARTIAL_STATUSES = {"truncated", "failed", "error"}


def key(environment, media_id):
    if not environment or not media_id:
        raise ValueError("Missing environment or media ID")
    return environment, media_id


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_sources(manifest_path, plan_path):
    manifest, plan = load_json(manifest_path), load_json(plan_path)
    calls, planned = {}, {}
    for call in manifest["calls"]:
        k = key(call["environment"], call["media_id"])
        if k in calls:
            raise ValueError(f"Duplicate manifest key: {k[0]}:{k[1]}")
        calls[k] = call
    for row in plan["source_keys"]:
        k = key(row["source_environment"], row["media_id"])
        if k in planned:
            raise ValueError(f"Duplicate import-plan key: {k[0]}:{k[1]}")
        planned[k] = row
    for k, call in calls.items():
        source = planned.get(k)
        if not source or source["audio_sha256"] != call["user_sha256"]:
            raise ValueError(f"Manifest/import-plan mismatch: {k[0]}:{k[1]}")
        if call.get("call_id") not in call.get("source_call_ids", [call.get("call_id")]):
            raise ValueError(f"Selected call is not among source aliases: {k[0]}:{k[1]}")
    return manifest, calls, planned


def transcript_from_result(result):
    text = result.get("transcript")
    if isinstance(text, str):
        return text.strip()
    return "\n".join(
        segment.get("output", {}).get("text", "").strip()
        for segment in result.get("segments", [])
        if isinstance(segment.get("output", {}).get("text", ""), str)
    ).strip()


def effective_history_mode(recipe_mode, history_policy):
    selection = (history_policy or {}).get("selection", "").lower()
    if selection.startswith("self-only prior completed qwen transcripts"):
        return "self_transcripts_only"
    if selection.startswith("only original agent turns") or selection.startswith(
        "original past assistant messages"
    ):
        return "self_transcripts_and_agent"
    return recipe_mode


def validate_result(
    path, result, manifest, calls, planned, expected_checkpoint=CHECKPOINT_SHA256, raw_bytes=None
):
    status = result.get("status")
    if status == "running":
        return None
    if status not in {COMPLETE, *PARTIAL_STATUSES}:
        raise ValueError(f"Unsupported result status: {status!r}")
    recipe, provenance = result.get("recipe") or {}, result.get("provenance") or {}
    if recipe.get("checkpoint_sha256") != expected_checkpoint:
        raise ValueError("Checkpoint hash mismatch")
    k = key(result.get("environment"), result.get("media_id"))
    call = calls.get(k)
    source = planned.get(k)
    if not call or not source:
        raise ValueError(f"No canonical call for {k[0]}:{k[1]}")
    mode = recipe.get("history_mode")
    history_policy = provenance.get("history_policy", call.get("history_policy"))
    effective_mode = effective_history_mode(mode, history_policy)
    if not mode or (manifest.get("history_mode") and mode not in {manifest["history_mode"], effective_mode}):
        raise ValueError("History mode mismatch")
    audio_sha = result.get("audio_sha256")
    if (result.get("call_id") != call.get("call_id")
            or audio_sha != call.get("user_sha256")
            or audio_sha != source.get("audio_sha256")
            or provenance.get("user_sha256") != audio_sha):
        raise ValueError(f"Call/audio identity mismatch: {k[0]}:{k[1]}")
    segments = result.get("segments") or []
    if status == COMPLETE:
        if not segments or any(not s.get("output", {}).get("complete") for s in segments):
            raise ValueError("Complete result contains incomplete segments")
        if not transcript_from_result(result):
            raise ValueError("Complete result has no transcript")
    transcript = transcript_from_result(result)
    digest = sha256_bytes(raw_bytes if raw_bytes is not None else Path(path).read_bytes())
    if result.get("key") and result["key"] != Path(path).stem:
        raise ValueError("Result filename/key mismatch")
    return {
        "status": status,
        "environment": k[0],
        "media_id": k[1],
        "call": call,
        "source": source,
        "recipe": recipe,
        "provenance": provenance,
        "effective_history_mode": effective_mode,
        "history_policy": history_policy,
        "transcript": transcript,
        "result_sha256": digest,
        "result_key": result.get("key", Path(path).stem),
        "segment_count": len(segments),
        "audio_sha256": audio_sha,
    }


def mode_display(mode, policy):
    if mode == "past_online_history":
        return "Предыдущие онлайн-распознанные реплики"
    if mode == "self_transcripts_and_agent":
        return "Предыдущие собственные расшифровки Qwen и история агента; онлайн-текст не подавался"
    if mode == "self_transcripts_only":
        return "Только предыдущие расшифровки Qwen; исходная история агента не подавалась"
    if mode == "self_transcripts_only":
        return "Только предыдущие собственные расшифровки Qwen; история агента не подавалась"
    return f"{mode}; политика: {json.dumps(policy, ensure_ascii=False, sort_keys=True)}"


def task_fields(item):
    status, text = item["status"], item["transcript"]
    if status == COMPLETE:
        display = text
    elif status == "truncated":
        display = "ЧАСТИЧНЫЙ ВЫВОД: генерация оборвалась; не использовать как готовый текст.\n" + text
    else:
        display = f"Генерация завершилась статусом {status}; готового текста нет."
    call, recipe, provenance = item["call"], item["recipe"], item["provenance"]
    history_policy = item["history_policy"]
    effective_mode = item["effective_history_mode"]
    return {
        "qwen_history_status": status,
        "qwen_history_display": display,
        "qwen_history_text": text if status in {COMPLETE, "truncated"} else "",
        "qwen_history_mode": effective_mode,
        "qwen_history_mode_display": mode_display(effective_mode, history_policy),
        "qwen_history_variant": effective_mode,
        "qwen_history_model": "KL1532 Qwen",
        "qwen_history_environment": item["environment"],
        "qwen_history_media_id": item["media_id"],
        "qwen_history_call_id": call["call_id"],
        "qwen_history_source_call_ids": call.get("source_call_ids", [call["call_id"]]),
        "qwen_history_audio_sha256": item["audio_sha256"],
        "qwen_history_checkpoint_sha256": recipe["checkpoint_sha256"],
        "qwen_history_result_key": item["result_key"],
        "qwen_history_result_sha256": item["result_sha256"],
        "qwen_history_segment_count": item["segment_count"],
        "qwen_history_provenance": {
            "history_mode": recipe["history_mode"],
            "effective_history_mode": effective_mode,
            "history_policy": history_policy,
            "segmentation_provenance": provenance.get(
                "segmentation_provenance", call.get("segmentation_provenance")),
            "channel_policy": call.get("channel_policy"),
            "channel_role_verified": call.get("channel_role_verified"),
            "source_selection": call.get("source_selection"),
            "recipe": recipe,
        },
    }


def human_work(task):
    return {
        "annotations": [
            {k: a.get(k) for k in ("id", "result", "was_cancelled", "ground_truth")}
            for a in task.get("annotations", [])
        ],
        "drafts": [
            {k: d.get(k) for k in ("id", "result", "lead_time", "created_at", "updated_at")}
            for d in task.get("drafts", [])
        ],
    }


def request_json(api_url, token, path, method="GET", payload=None, cookie_header=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    if cookie_header:
        headers["Cookie"] = cookie_header
    req = urllib.request.Request(
        api_url.rstrip("/") + path, data=data, method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read()) if response.status != 204 else None
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Label Studio API {method} {path} returned HTTP {exc.code}") from None


def index_tasks(api_url, token, project_id, cookie_header=None):
    query = urllib.parse.urlencode({"project": project_id, "page_size": 1000})
    body = request_json(api_url, token, "/api/tasks/?" + query, cookie_header=cookie_header)
    tasks = body.get("tasks", [])
    index = {}
    for task in tasks:
        data = task.get("data", {})
        k = key(data.get("source_environment"), data.get("media_id"))
        if k in index:
            raise ValueError(f"Duplicate Label Studio task key: {k[0]}:{k[1]}")
        index[k] = task
    return index


def previous_candidate(data):
    digest = data.get("qwen_history_result_sha256")
    if not digest:
        return None
    return {
        "status": data.get("qwen_history_status"),
        "text": data.get("qwen_history_text", ""),
        "mode": data.get("qwen_history_mode"),
        "mode_display": data.get("qwen_history_mode_display"),
        "environment": data.get("qwen_history_environment"),
        "media_id": data.get("qwen_history_media_id"),
        "call_id": data.get("qwen_history_call_id"),
        "audio_sha256": data.get("qwen_history_audio_sha256"),
        "checkpoint_sha256": data.get("qwen_history_checkpoint_sha256"),
        "result_key": data.get("qwen_history_result_key"),
        "result_sha256": digest,
        "provenance": data.get("qwen_history_provenance"),
        "archived_utc": datetime.now(timezone.utc).isoformat(),
    }


def import_item(api_url, token, task_summary, item, cookie_header=None, supersede_complete=False):
    task_id = task_summary["id"]
    task = request_json(api_url, token, f"/api/tasks/{task_id}", cookie_header=cookie_header)
    data = task["data"]
    if key(data.get("source_environment"), data.get("media_id")) != (item["environment"], item["media_id"]):
        raise ValueError(f"Task identity changed: {task_id}")
    if data.get("audio_sha256") and data["audio_sha256"] != item["audio_sha256"]:
        raise ValueError(f"Task audio hash mismatch: {task_id}")
    fields = task_fields(item)
    current_status = data.get("qwen_history_status", "not_run")
    old_digest = data.get("qwen_history_result_sha256")
    if current_status == item["status"] and old_digest == item["result_sha256"]:
        return "already_imported"
    if current_status == COMPLETE:
        if (not supersede_complete or item["status"] != COMPLETE or not old_digest
                or data.get("qwen_history_audio_sha256") != item["audio_sha256"]
                or data.get("qwen_history_call_id") != item["call"]["call_id"]
                or data.get("qwen_history_checkpoint_sha256") != item["recipe"]["checkpoint_sha256"]
                or data.get("qwen_history_mode") != item["effective_history_mode"]):
            return "conflict"
        archived = list(data.get("qwen_history_previous_candidates", []))
        if not any(entry.get("result_sha256") == old_digest for entry in archived):
            prior = previous_candidate(data)
            if prior:
                archived.append(prior)
        fields["qwen_history_previous_candidates"] = archived
    if current_status in {"truncated", "failed", "error"} and item["status"] != COMPLETE:
        if old_digest != item["result_sha256"]:
            return "conflict"
        return "already_imported"
    before = human_work(task)
    if current_status not in {"not_run", "truncated", "failed", "error"} and not (
        current_status == COMPLETE and supersede_complete and item["status"] == COMPLETE
    ):
        return "conflict"
    old_keys = set(data)
    data.update(fields)
    request_json(api_url, token, f"/api/tasks/{task_id}/", method="PATCH", payload={"data": data}, cookie_header=cookie_header)
    after = request_json(api_url, token, f"/api/tasks/{task_id}", cookie_header=cookie_header)
    if any(after["data"].get(k) != v for k, v in fields.items()):
        raise RuntimeError(f"Task fields did not verify after update: {task_id}")
    if any(after["data"].get(k) != data[k] for k in old_keys):
        raise RuntimeError(f"Existing task data changed: {task_id}")
    if human_work(after) != before:
        raise RuntimeError(f"Human annotations or drafts changed: {task_id}")
    return "imported"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def scan(args, token, manifest, calls, planned, cookie_header=None):
    tasks = index_tasks(args.api_url, token, args.project_id, cookie_header)
    counts = {k: 0 for k in ("complete", "truncated", "failed", "error", "running", "invalid", "unmatched", "imported", "already_imported", "conflict")}
    errors, imported = [], []
    for result_dir in args.results_dir:
        for path in sorted(Path(result_dir).glob("*.json")):
            try:
                raw_bytes = Path(path).read_bytes()
                result = json.loads(raw_bytes)
                status = result.get("status", "invalid")
                if status in counts:
                    counts[status] += 1
                if status == "running":
                    continue
                item = validate_result(
                    path, result, manifest, calls, planned, args.checkpoint_sha256, raw_bytes
                )
                if item is None:
                    continue
                k = key(item["environment"], item["media_id"])
                summary = tasks.get(k)
                if not summary:
                    counts["unmatched"] += 1
                    errors.append(f"{path.name}: no Label Studio task for {k[0]}:{k[1]}")
                    continue
                # The project task index already contains these fields. Avoid an
                # extra API round-trip for every unchanged result on every poll.
                existing_data = summary.get("data", {})
                existing_status = existing_data.get("qwen_history_status", "not_run")
                existing_digest = existing_data.get("qwen_history_result_sha256")
                if existing_status == item["status"] and existing_digest == item["result_sha256"]:
                    counts["already_imported"] += 1
                    continue
                if existing_status in PARTIAL_STATUSES and item["status"] in PARTIAL_STATUSES:
                    counts["conflict"] += 1
                    errors.append(f"{path.name}: existing Qwen result conflicts with this result")
                    continue
                outcome = import_item(
                    args.api_url, token, summary, item, cookie_header, args.supersede_complete
                )
                counts[outcome] += 1
                if outcome == "imported":
                    summary.setdefault("data", {}).update(task_fields(item))
                    imported.append({"task_id": summary["id"], "environment": k[0], "media_id": k[1], "status": item["status"]})
                elif outcome == "conflict":
                    errors.append(f"{path.name}: existing Qwen result conflicts with this result")
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                counts["invalid"] += 1
                errors.append(f"{path.name}: {exc}")
    state = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "project_id": args.project_id,
        "checkpoint_sha256": args.checkpoint_sha256,
        "history_mode": manifest.get("history_mode"),
        "manifest_sha256": sha256_bytes(Path(args.manifest).read_bytes()),
        "plan_sha256": sha256_bytes(Path(args.plan).read_bytes()),
        "manifest_call_count": len(calls),
        "plan_source_count": len(planned),
        "result_directories": [str(Path(p)) for p in args.results_dir],
        "counts_this_scan": counts,
        "task_count": len(tasks),
        "queue_qwen_status_counts": dict(Counter(
            t.get("data", {}).get("qwen_history_status", "missing") for t in tasks.values())),
        "imports_this_scan": imported,
        "errors_last_scan": errors[-50:],
    }
    if args.status_file:
        atomic_json(args.status_file, state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    return counts


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", action="append", required=True, help="Repeat for multiple result directories")
    parser.add_argument("--manifest", required=True, type=Path, help="Canonical call manifest used by inference")
    parser.add_argument("--plan", required=True, type=Path, help="Canonical Label Studio import plan")
    parser.add_argument("--project-id", type=int, default=108)
    parser.add_argument("--api-url", default=os.environ.get("LABEL_STUDIO_URL", "https://label.bitmanager.ai"))
    parser.add_argument("--token-file", type=Path, help="Private file containing a Label Studio API bearer token")
    parser.add_argument("--cookie-file", type=Path, help="Optional private JSON cookie export for deployments behind session auth")
    parser.add_argument("--checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument(
        "--supersede-complete", action="store_true",
        help="Replace a complete result only when call/audio/checkpoint/history identities match; archive the prior candidate",
    )
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--watch-seconds", type=int, default=0, help="Poll repeatedly; 0 runs once")
    args = parser.parse_args()
    token = args.token_file.read_text().strip() if args.token_file else os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
    if not token:
        parser.error("set LABEL_STUDIO_ACCESS_TOKEN or pass --token-file")
    cookie_header = None
    if args.cookie_file:
        cookie_rows = load_json(args.cookie_file)
        if not isinstance(cookie_rows, list) or any(not c.get("name") or "value" not in c for c in cookie_rows):
            parser.error("cookie file must be a list of browser cookie objects")
        cookie_header = "; ".join(f"{c['name']}={c['value']}" for c in cookie_rows)
    manifest, calls, planned = load_sources(args.manifest, args.plan)
    if args.watch_seconds < 0:
        parser.error("--watch-seconds cannot be negative")
    while True:
        scan(args, token, manifest, calls, planned, cookie_header)
        if not args.watch_seconds:
            return
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
