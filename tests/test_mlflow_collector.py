import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lalm/lalm_core"))
spec = importlib.util.spec_from_file_location(
    "collector", ROOT / "infra/mlflow/collect_existing_runs.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class Client:
    def __init__(self, fail=False):
        self.fail = fail
        self.metrics = []

    def create(self, *args):
        return "run"

    def log(self, run, metrics):
        if self.fail:
            raise OSError("server unavailable")
        self.metrics.extend(metrics)


class CollectorTests(unittest.TestCase):
    def test_partial_record_retried_and_nonnumeric_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "metrics.jsonl"
            p.write_text(
                '{"step":2,"loss":0.5,"text":"private","invalid":"NaN"}\n{"step":3'
            )
            metrics, cursor = collector.jsonl_metrics({"path": str(p)}, {})
            self.assertEqual(
                [(m["key"], m["step"], m["value"]) for m in metrics],
                [("train/loss", 2, 0.5)],
            )
            p.write_text(p.read_text() + ',"loss":0.25}\n')
            metrics, cursor = collector.jsonl_metrics({"path": str(p)}, cursor)
            self.assertEqual(metrics[0]["step"], 3)
            self.assertEqual(cursor["line"], 2)
            self.assertEqual(collector.jsonl_metrics({"path": str(p)}, cursor)[0], [])

    def test_failed_delivery_does_not_advance_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "metrics.jsonl"
            p.write_text('{"step":1,"loss":0.5}\n')
            config = {"sources": [{"name": "run", "path": str(p), "format": "jsonl"}]}
            state = {}
            state_path = Path(directory) / "state.json"
            with self.assertRaises(OSError):
                collector.collect(Client(fail=True), config, state, state_path)
            self.assertNotIn("cursor", state["run"])
            client = Client()
            collector.collect(client, config, state, state_path)
            self.assertEqual(len(client.metrics), 1)
            collector.collect(client, config, state, state_path)
            self.assertEqual(len(client.metrics), 1)
            self.assertEqual(
                json.loads(state_path.read_text())["run"]["cursor"]["line"], 1
            )

    def test_prometheus_labels_do_not_collide(self):
        from io import BytesIO

        body = b'# TYPE sglang:tokens counter\nsglang:tokens{model="a"} 12\nsglang:tokens{model="b"} 7\nsglang:bad NaN\n'
        with patch.object(
            collector.urllib.request, "urlopen", return_value=BytesIO(body)
        ):
            values = collector.prometheus_metrics("http://local/metrics")
        self.assertEqual(sorted(values.values()), [7, 12])
        self.assertEqual(len(values), 2)

    def test_remote_gpu_probe_does_not_report_collector_host_ram(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            state = {}
            result = SimpleNamespace(
                stdout="0, RTX, 99, 10, 100, 200, 70, 123.5, 250, 1200, 4000\n"
            )
            with patch.object(collector.subprocess, "run", return_value=result) as run:
                collector.collect(
                    client,
                    {"gpu_command": ["ssh", "gpu-probe"]},
                    state,
                    Path(directory) / "state.json",
                )
            run.assert_called_once_with(
                ["ssh", "gpu-probe"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            values = {m["key"]: m["value"] for m in client.metrics}
            self.assertEqual(values["system/gpu_0/power_draw_watts"], 123.5)
            self.assertNotIn("system/ram_used_mib", values)

    def test_finished_process_status_remains_current_without_new_events(self):
        with tempfile.TemporaryDirectory() as directory:
            cursor = {"train/loss": [1000, 1135]}
            state = {"run": {"run_id": "existing", "cursor": cursor}}
            state_path = Path(directory) / "state.json"
            config = {
                "sources": [
                    {
                        "name": "run",
                        "format": "tensorboard",
                        "path": "unused",
                        "pid": -1,
                    }
                ]
            }
            client = Client()
            with patch.object(
                collector,
                "tensorboard_metrics",
                side_effect=lambda *args: ([], dict(cursor)),
            ):
                collector.collect(client, config, state, state_path)
                restarted = json.loads(state_path.read_text())
                collector.collect(client, config, restarted, state_path)
            statuses = [
                m
                for m in client.metrics
                if m["key"] == "monitor/source_process_running"
            ]
            old = {"step": 1135, "timestamp": 1000000, "value": 1.0}
            latest = max([old] + statuses, key=lambda m: (m["step"], m["timestamp"]))
            self.assertEqual(latest["value"], 0.0)
            self.assertGreater(statuses[1]["step"], statuses[0]["step"])
            self.assertTrue(
                any(m["key"] == "monitor/heartbeat_timestamp" for m in client.metrics)
            )


if __name__ == "__main__":
    unittest.main()
