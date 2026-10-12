"""Optional real-server smoke test: python tests/mlflow_native_smoke.py."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm/lalm_core"))
from mlflow import MlflowClient
from mlflow_monitoring import MLflowMonitor


class Model:
    def parameters(self):
        return []


cfg = {
    "monitoring": {
        "enabled": True,
        "experiment_name": "SpeechLLM monitoring validation",
        "run_name": "native-hooks-cpu-smoke",
        "system_metrics_sampling_interval": 0,
        "log_system_metrics": False,
    }
}
monitor = MLflowMonitor(cfg, "smoke", 0, Model())
monitor.add_batch(2, 10)
monitor.log_training(
    12,
    1,
    SimpleNamespace(_values={"loss": 0.7}),
    SimpleNamespace(_values={"loss": 0.5}),
    1e-4,
    1,
    0.25,
    SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    "cpu",
)
monitor.log_validation(12, "smoke", SimpleNamespace(_values={"loss": 0.4}))
monitor.close()
run = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"]).get_run(
    monitor.run_id
)
assert run.info.status == "FINISHED"
assert run.data.metrics["train/current/loss"] == 0.7
assert run.data.metrics["train/average/loss"] == 0.5
assert run.data.metrics["validation/smoke/loss"] == 0.4
print("Native MLflow hooks smoke passed:", monitor.run_id)
