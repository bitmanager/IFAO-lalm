#!/usr/bin/env python3
"""Import numeric training logs and poll hardware/serving metrics without restarting jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lalm" / "lalm_core"))
from mlflow_monitoring import _gpu_metrics, _parse_nvidia_gpu_metrics


def numeric(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_key(value):
    return re.sub(r"[^A-Za-z0-9_. /-]", "_", str(value))[:240]


class Client:
    def __init__(self, uri, experiment):
        self.uri = uri.rstrip("/")
        found = self.request(
            "experiments/search",
            {
                "filter": "name = '" + experiment.replace("'", "\\'") + "'",
                "max_results": 1,
            },
        )
        rows = found.get("experiments", [])
        self.experiment_id = (
            rows[0]["experiment_id"]
            if rows
            else self.request("experiments/create", {"name": experiment})[
                "experiment_id"
            ]
        )

    def request(self, endpoint, data):
        req = urllib.request.Request(
            self.uri + "/api/2.0/mlflow/" + endpoint,
            data=json.dumps(data, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    def create(self, name, kind, source):
        tags = {
            "mlflow.runName": name,
            "capture": "external-log-collector",
            "run_type": kind,
            "source": source,
            "host": os.uname().nodename,
        }
        return self.request(
            "runs/create",
            {
                "experiment_id": self.experiment_id,
                "start_time": int(time.time() * 1000),
                "tags": [{"key": k, "value": v} for k, v in tags.items()],
            },
        )["run"]["info"]["run_id"]

    def log(self, run_id, metrics):
        for index in range(0, len(metrics), 500):
            self.request(
                "runs/log-batch",
                {
                    "run_id": run_id,
                    "metrics": metrics[index : index + 500],
                    "params": [],
                    "tags": [],
                },
            )


def tensorboard_metrics(source, cursor):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(source["path"], size_guidance={"scalars": 0})
    acc.Reload()
    metrics, next_cursor = [], dict(cursor)
    for tag in acc.Tags().get("scalars", []):
        last = tuple(cursor.get(tag, [-1, -1]))
        for event in sorted(acc.Scalars(tag), key=lambda e: (e.wall_time, e.step)):
            position = (event.wall_time, event.step)
            value = numeric(event.value)
            if position > last and value is not None:
                metrics.append(
                    {
                        "key": metric_key(tag),
                        "value": value,
                        "step": int(event.step),
                        "timestamp": int(event.wall_time * 1000),
                    }
                )
                next_cursor[tag] = list(position)
    return metrics, next_cursor


def jsonl_metrics(source, cursor):
    metrics, next_cursor = [], dict(cursor)
    last = int(cursor.get("line", 0))
    with open(source["path"]) as stream:
        for index, line in enumerate(stream, 1):
            if index <= last:
                continue
            if not line.endswith("\n"):
                break  # The writer has not completed this record yet.
            row = json.loads(line)
            step = int(row.get("step", index))
            timestamp = int(time.time() * 1000)
            for key, value in row.items():
                if key in ("step", "timestamp") or (
                    source.get("metrics") and key not in source["metrics"]
                ):
                    continue
                value = numeric(value)
                if value is not None:
                    metrics.append(
                        {
                            "key": "train/" + metric_key(key),
                            "value": value,
                            "step": step,
                            "timestamp": timestamp,
                        }
                    )
            next_cursor["line"] = index
    return metrics, next_cursor


def prometheus_metrics(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        lines = response.read().decode().splitlines()
    metrics = {}
    for line in lines:
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)", line)
        if not match:
            continue
        name, labels, raw = match.groups()
        value = numeric(raw)
        if value is None:
            continue
        suffix = (
            "_" + hashlib.sha256(labels.encode()).hexdigest()[:12] if labels else ""
        )
        metrics["serving/" + metric_key(name) + suffix] = value
    return metrics


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state))
    temp.replace(path)


def collect(client, config, state, state_path):
    for source in config.get("sources", []):
        saved = state.setdefault(source["name"], {})
        if not saved.get("run_id"):
            saved["run_id"] = client.create(
                source["name"], source.get("kind", "imported-training"), source["path"]
            )
            save_state(state_path, state)
        loader = {"tensorboard": tensorboard_metrics, "jsonl": jsonl_metrics}[
            source["format"]
        ]
        metrics, cursor = loader(source, saved.get("cursor", {}))
        if source.get("pid"):
            metrics.append(
                {
                    "key": "monitor/source_process_running",
                    "value": float(Path("/proc", str(source["pid"])).exists()),
                    "step": max((m["step"] for m in metrics), default=0),
                    "timestamp": int(time.time() * 1000),
                }
            )
        client.log(saved["run_id"], metrics)
        saved["cursor"] = cursor  # Advance only after successful delivery.
        save_state(state_path, state)
        logging.info("%s: delivered %d metrics", source["name"], len(metrics))
    if config.get("hardware") or config.get("prometheus") or config.get("gpu_command"):
        saved = state.setdefault("_host", {})
        if not saved.get("run_id"):
            saved["run_id"] = client.create(
                config.get("host_run_name", "host-" + os.uname().nodename),
                "hardware-and-serving",
                os.uname().nodename,
            )
            save_state(state_path, state)
        values = {"monitor/heartbeat_timestamp": time.time()}
        if config.get("gpu_command"):
            probe = subprocess.run(
                config["gpu_command"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            values.update(_parse_nvidia_gpu_metrics(probe.stdout)[0])
        if config.get("hardware"):
            values.update(_gpu_metrics()[0])
            for index, load in enumerate(os.getloadavg()):
                values["system/load_average_" + str(index)] = load
            with open("/proc/meminfo") as stream:
                memory = {
                    line.split(":")[0]: float(line.split()[1]) / 1024
                    for line in stream
                    if line.startswith(("MemTotal:", "MemAvailable:"))
                }
            values["system/ram_total_mib"] = memory["MemTotal"]
            values["system/ram_used_mib"] = memory["MemTotal"] - memory["MemAvailable"]
        for url in config.get("prometheus", []):
            values.update(prometheus_metrics(url))
        step = saved.get("step", 0)
        client.log(
            saved["run_id"],
            [
                {
                    "key": key,
                    "value": value,
                    "step": step,
                    "timestamp": int(time.time() * 1000),
                }
                for key, value in values.items()
            ],
        )
        saved["step"] = step + 1
        save_state(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = json.loads(args.config.read_text())
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    client = Client(config["tracking_uri"], config.get("experiment", "SpeechLLM R&D"))
    while True:
        try:
            collect(client, config, state, args.state)
        except Exception:
            logging.exception(
                "Collection failed; retrying without advancing unsent cursors"
            )
            if args.once:
                raise
        if args.once:
            for item in state.values():
                if item.get("run_id"):
                    client.request(
                        "runs/update",
                        {
                            "run_id": item["run_id"],
                            "status": "FINISHED",
                            "end_time": int(time.time() * 1000),
                        },
                    )
            return
        time.sleep(config.get("interval_seconds", 30))


if __name__ == "__main__":
    main()
