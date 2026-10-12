import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "monitor",
    Path(__file__).resolve().parents[1] / "lalm/lalm_core/mlflow_monitoring.py",
)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class MonitoringTests(unittest.TestCase):
    def test_disabled_and_nonzero_rank_require_no_mlflow(self):
        with patch.dict(os.environ, {"MLFLOW_ENABLED": "false"}):
            self.assertFalse(monitor.MLflowMonitor({}, "run", 0, None).enabled)
        with patch.dict(os.environ, {"MLFLOW_ENABLED": "true"}):
            self.assertFalse(monitor.MLflowMonitor({}, "run", 1, None).enabled)

    def test_env_boolean_is_parsed(self):
        for value, expected in [
            ("false", False),
            ("0", False),
            ("true", True),
            ("yes", True),
        ]:
            with patch.dict(os.environ, {"MLFLOW_ENABLED": value}):
                self.assertEqual(monitor.monitoring_enabled({}), expected)

    def test_nvidia_unsupported_values_are_omitted(self):
        result = type(
            "Result",
            (),
            {"stdout": "0, RTX, 99, 10, 100, 200, 70, 123.5, [N/A], 1200, 4000\n"},
        )()
        with patch.object(monitor.subprocess, "run", return_value=result):
            metrics, tags = monitor._nvidia_gpu_metrics("nvidia-smi")
        self.assertEqual(metrics["system/gpu_0/power_draw_watts"], 123.5)
        self.assertNotIn("system/gpu_0/power_limit_watts", metrics)
        self.assertEqual(tags["gpu_0_name"], "RTX")

    def test_delivery_failure_does_not_abort_training(self):
        from types import SimpleNamespace

        with patch.dict(os.environ, {"MLFLOW_ENABLED": "false"}):
            instance = monitor.MLflowMonitor({}, "run", 0, None)
        instance.mlflow = SimpleNamespace(
            log_metrics=lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("offline")
            )
        )
        tracker = SimpleNamespace(_values={"loss": 0.5})
        with self.assertLogs(level="WARNING"):
            instance.log_training(
                1,
                1,
                tracker,
                tracker,
                0.001,
                1,
                None,
                SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                "cpu",
            )
            instance.log_validation(1, "validation", tracker)


if __name__ == "__main__":
    unittest.main()
