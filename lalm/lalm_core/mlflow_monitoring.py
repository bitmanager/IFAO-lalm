"""MLflow run and hardware telemetry helpers for LALM training."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def _get(obj: Any, path: str, default: Any = None) -> Any:
    value = obj
    for part in path.split("."):
        if value is None:
            return default
        try:
            value = value.get(part, default)
        except AttributeError:
            value = getattr(value, part, default)
    return default if value is None else value


def monitoring_enabled(cfg: Any) -> bool:
    value = os.getenv("MLFLOW_ENABLED", _get(cfg, "monitoring.enabled", False))
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _file_sha256(value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_metrics() -> tuple[dict[str, float], dict[str, str]]:
    """Read per-device NVIDIA utilization, memory, temperature, power and clocks."""
    executable = shutil.which("nvidia-smi")
    if executable is not None:
        return _nvidia_gpu_metrics(executable)
    executable = shutil.which("amd-smi")
    if executable is not None:
        metrics, tags = _amd_smi_metrics(executable)
        if metrics:
            return metrics, tags
    executable = shutil.which("rocm-smi")
    if executable is not None:
        return _amd_gpu_metrics(executable)
    return {}, {}


def _amd_metric_number(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value")
    return _number(value)


def _amd_smi_metrics(executable: str) -> tuple[dict[str, float], dict[str, str]]:
    """Read the current AMD SMI JSON schema (including integrated GPUs)."""
    try:
        result = subprocess.run(
            [executable, "metric", "-m", "-u", "-p", "-c", "-t", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        cards = json.loads(result.stdout).get("gpu_data", [])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logging.warning("Could not query AMD SMI GPU metrics: %s", exc)
        return {}, {}

    metrics: dict[str, float] = {}
    tags: dict[str, str] = {}
    for card in cards:
        try:
            index = int(card["gpu"])
        except (KeyError, TypeError, ValueError):
            continue
        prefix = f"system/gpu_{index}"
        tags[f"gpu_{index}_name"] = f"AMD GPU {index}"

        memory = card.get("mem_usage", {})
        memory_values: dict[str, float] = {}
        for key, source in (
            ("memory_total_mib", "total_vram"),
            ("memory_used_mib", "used_vram"),
        ):
            reading = memory.get(source, {})
            value = _amd_metric_number(reading)
            if value is None:
                continue
            unit = (
                str(reading.get("unit", "MB")).upper()
                if isinstance(reading, dict)
                else "MB"
            )
            if unit == "GB":
                value *= 1024
            memory_values[key] = value
            metrics[f"{prefix}/{key}"] = value

        if memory_values.get("memory_total_mib", 0) > 0:
            metrics[f"{prefix}/utilization_memory_percent"] = (
                100.0
                * memory_values.get("memory_used_mib", 0)
                / memory_values["memory_total_mib"]
            )

        usage = card.get("usage", {})
        if isinstance(usage, dict):
            for source in (
                "gfx_activity",
                "gpu_activity",
                "gpu_usage",
                "gfx_utilization",
            ):
                value = _amd_metric_number(usage.get(source))
                if value is not None:
                    metrics[f"{prefix}/utilization_gpu_percent"] = value
                    break

        temperatures = card.get("temperature", {})
        for source in ("hotspot", "edge"):
            value = _amd_metric_number(temperatures.get(source))
            if value is not None:
                metrics[f"{prefix}/temperature_celsius"] = value
                break

        power = card.get("power", {})
        value = _amd_metric_number(power.get("socket_power"))
        if value is not None:
            metrics[f"{prefix}/power_draw_watts"] = value

        clocks = card.get("clock", {})
        for source, key in (
            ("gfx_0", "graphics_clock_mhz"),
            ("mem_0", "memory_clock_mhz"),
        ):
            reading = clocks.get(source, {}).get("clk")
            value = _amd_metric_number(reading)
            if value is not None:
                unit = (
                    str(reading.get("unit", "MHz")).upper()
                    if isinstance(reading, dict)
                    else "MHz"
                )
                if unit == "GHZ":
                    value *= 1000
                metrics[f"{prefix}/{key}"] = value

    return metrics, tags


def _nvidia_gpu_metrics(executable: str) -> tuple[dict[str, float], dict[str, str]]:
    fields = (
        "index,name,utilization.gpu,utilization.memory,memory.used,memory.total,"
        "temperature.gpu,power.draw,power.limit,clocks.current.graphics,"
        "clocks.current.memory"
    )
    try:
        result = subprocess.run(
            [executable, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logging.warning("Could not query NVIDIA GPU metrics: %s", exc)
        return {}, {}

    return _parse_nvidia_gpu_metrics(result.stdout)


def _parse_nvidia_gpu_metrics(output: str) -> tuple[dict[str, float], dict[str, str]]:
    """Parse the same fixed NVIDIA SMI CSV schema for local or remote probes."""
    metrics: dict[str, float] = {}
    tags: dict[str, str] = {}
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) < 11:
            continue
        index, name = row[0].strip(), row[1].strip()
        prefix = f"system/gpu_{index}"
        tags[f"gpu_{index}_name"] = name[:250]
        for key, value in (
            ("utilization_gpu_percent", row[2]),
            ("utilization_memory_percent", row[3]),
            ("memory_used_mib", row[4]),
            ("memory_total_mib", row[5]),
            ("temperature_celsius", row[6]),
            ("power_draw_watts", row[7]),
            ("power_limit_watts", row[8]),
            ("graphics_clock_mhz", row[9]),
            ("memory_clock_mhz", row[10]),
        ):
            numeric = _number(value.strip().replace(" MiB", "").replace(" W", ""))
            if numeric is not None:
                metrics[f"{prefix}/{key}"] = numeric
    return metrics, tags


def _amd_gpu_metrics(executable: str) -> tuple[dict[str, float], dict[str, str]]:
    try:
        result = subprocess.run(
            [
                executable,
                "--showuse",
                "--showmeminfo",
                "vram",
                "--showpower",
                "--showtemp",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        cards = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logging.warning("Could not query AMD GPU metrics: %s", exc)
        return {}, {}

    metrics: dict[str, float] = {}
    tags: dict[str, str] = {}
    for card, values in cards.items():
        try:
            index = int(card.removeprefix("card"))
        except ValueError:
            continue
        prefix = f"system/gpu_{index}"
        fields = (
            ("utilization_gpu_percent", "GPU use (%)", 1.0),
            ("temperature_celsius", "Temperature (Sensor edge) (C)", 1.0),
            (
                "power_draw_watts",
                "Current Socket Graphics Package Power (W)",
                1.0,
            ),
            ("memory_total_mib", "VRAM Total Memory (B)", 1.0 / (1024**2)),
            ("memory_used_mib", "VRAM Total Used Memory (B)", 1.0 / (1024**2)),
        )
        for key, source, scale in fields:
            number = _number(values.get(source))
            if number is not None:
                metrics[f"{prefix}/{key}"] = number * scale
        tags[f"gpu_{index}_name"] = "AMD GPU"
    return metrics, tags


class MLflowMonitor:
    """Own one MLflow run and expose low-volume training/system logging."""

    def __init__(self, cfg: Any, exp_dir: str, rank: int, model: Any):
        self.enabled = monitoring_enabled(cfg) and rank == 0
        self.mlflow = None
        self.run_id: str | None = None
        self._client = None
        self._last_log_time: float | None = None
        self._window_samples = 0.0
        self._window_target_tokens = 0.0
        self._window_steps = 0
        self._last_step = 0
        self._stop_event = threading.Event()
        self._log_lock = threading.Lock()
        self._tag_lock = threading.Lock()
        self._gpu_tags_logged: set[str] = set()
        self._hardware_thread: threading.Thread | None = None
        self._run_name = str(
            _get(cfg, "monitoring.run_name", "")
            or os.getenv("MLFLOW_RUN_NAME")
            or Path(exp_dir).name
            or "speechllm-run"
        )
        if not self.enabled:
            return

        try:
            import mlflow
            from mlflow import MlflowClient
        except ImportError as exc:
            raise RuntimeError(
                "MLflow monitoring is enabled, but mlflow is not installed. "
                "Install lalm/requirements-monitoring.txt or set monitoring.enabled=false."
            ) from exc

        self.mlflow = mlflow
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            str(_get(cfg, "monitoring.tracking_uri", "http://127.0.0.1:5000")),
        )
        experiment_name = str(
            os.getenv(
                "MLFLOW_EXPERIMENT_NAME",
                _get(cfg, "monitoring.experiment_name", "SpeechLLM R&D"),
            )
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        sampling_interval = int(
            _get(cfg, "monitoring.system_metrics_sampling_interval", 10)
        )
        mlflow.set_system_metrics_sampling_interval(sampling_interval)
        mlflow.set_system_metrics_samples_before_logging(
            int(_get(cfg, "monitoring.system_metrics_samples_before_logging", 1))
        )

        try:
            self._run = mlflow.start_run(
                run_name=self._run_name,
                log_system_metrics=bool(
                    _get(cfg, "monitoring.log_system_metrics", True)
                ),
                tags={
                    "project": "speechllm-rd",
                    "framework": "auden-lalm",
                    "host": os.uname().nodename,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not start the MLflow run at {tracking_uri}: {exc}"
            ) from exc

        self.run_id = self._run.info.run_id
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._log_parameters(cfg, model)
        if sampling_interval > 0:
            self._sample_hardware()
            self._hardware_thread = threading.Thread(
                target=self._hardware_poll_loop,
                args=(sampling_interval,),
                name="mlflow-gpu-telemetry",
                daemon=True,
            )
            self._hardware_thread.start()
        logging.info(
            "MLflow run %s is recording to %s (experiment %s)",
            self.run_id,
            tracking_uri,
            experiment_name,
        )

    def _log_parameters(self, cfg: Any, model: Any) -> None:
        if self.mlflow is None:
            return
        trainer_params = {
            "seed": _get(cfg, "seed"),
            "model.pretrained_model": _get(cfg, "model.pretrained_model"),
            "trainer.mixed_precision": _get(cfg, "trainer.mixed_precision"),
            "trainer.frozen_modules": _get(cfg, "trainer.frozen_modules"),
            "trainer.grad_accum_steps": _get(cfg, "trainer.grad_accum_steps"),
            "trainer.max_grad_norm": _get(cfg, "trainer.max_grad_norm"),
            "trainer.num_steps": _get(cfg, "trainer.num_steps"),
            "trainer.valid_interval": _get(cfg, "trainer.valid_interval"),
            "data.feature": _get(cfg, "data.feature"),
            "data.sampling_rate": _get(cfg, "data.sampling_rate"),
            "data.sampler.max_tokens": _get(cfg, "data.sampler.max_tokens"),
            "data.train_config_sha256": _file_sha256(
                _get(cfg, "data.train_data_config")
            ),
            "data.validation_config_sha256": _file_sha256(
                _get(cfg, "data.valid_data_config")
            ),
            "optimizer.type": _get(cfg, "trainer.optimizer.type"),
            "optimizer.lr": _get(cfg, "trainer.optimizer.lr"),
            "optimizer.weight_decay": _get(cfg, "trainer.optimizer.weight_decay"),
            "scheduler.type": _get(cfg, "trainer.scheduler.type"),
        }
        params = {
            key: str(value)
            for key, value in trainer_params.items()
            if value is not None
        }
        params["model.parameter_count"] = str(
            sum(parameter.numel() for parameter in model.parameters())
        )
        params["model.trainable_parameter_count"] = str(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )
        self.mlflow.log_params(params)
        repo = Path(__file__).resolve().parents[2]
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
            )
            self.mlflow.set_tags(
                {"git.commit": revision, "git.dirty": str(dirty).lower()}
            )
        except (OSError, subprocess.SubprocessError):
            logging.debug("Could not record Git revision for MLflow run", exc_info=True)

    def add_batch(self, samples: float, target_tokens: float) -> None:
        if not self.enabled:
            return
        self._window_samples += samples
        self._window_target_tokens += target_tokens
        self._window_steps += 1

    def log_training(
        self,
        step: int,
        epoch: int,
        batch_metrics: Any,
        average_metrics: Any,
        learning_rate: float,
        grad_scale: float,
        grad_norm: float | None,
        torch_module: Any,
        device: Any,
    ) -> None:
        if self.mlflow is None:
            return
        self._last_step = step
        metrics: dict[str, float] = {
            "train/epoch": float(epoch),
            "train/learning_rate": learning_rate,
            "train/grad_scale": grad_scale,
        }
        for key, value in getattr(batch_metrics, "_values", {}).items():
            number = _number(value)
            if number is not None:
                metrics[f"train/current/{key}"] = number
        for key, value in getattr(average_metrics, "_values", {}).items():
            number = _number(value)
            if number is not None:
                metrics[f"train/average/{key}"] = number
        if grad_norm is not None:
            metrics["train/gradient_norm_pre_clip"] = grad_norm

        now = time.monotonic()
        if self._last_log_time is not None:
            elapsed = max(now - self._last_log_time, 1e-9)
            metrics["train/seconds_per_batch"] = elapsed / max(self._window_steps, 1)
            metrics["train/examples_per_second"] = self._window_samples / elapsed
            metrics["train/target_tokens_per_second"] = (
                self._window_target_tokens / elapsed
            )
        self._last_log_time = now
        self._window_samples = 0.0
        self._window_target_tokens = 0.0
        self._window_steps = 0

        try:
            if torch_module.cuda.is_available():
                metrics["system/torch_peak_allocated_mib"] = (
                    torch_module.cuda.max_memory_allocated(device) / (1024**2)
                )
                metrics["system/torch_peak_reserved_mib"] = (
                    torch_module.cuda.max_memory_reserved(device) / (1024**2)
                )
                torch_module.cuda.reset_peak_memory_stats(device)
        except (AttributeError, RuntimeError) as exc:
            logging.debug("Could not read PyTorch CUDA memory metrics: %s", exc)

        metrics = {key: value for key, value in metrics.items() if math.isfinite(value)}
        with self._log_lock:
            try:
                self.mlflow.log_metrics(metrics, step=step)
            except Exception:
                logging.warning(
                    "MLflow metric delivery failed at step %s", step, exc_info=True
                )

    def log_validation(self, step: int, valid_name: str, metrics_tracker: Any) -> None:
        if self.mlflow is None:
            return
        metrics = {
            f"validation/{valid_name}/{key}": number
            for key, value in getattr(metrics_tracker, "_values", {}).items()
            if (number := _number(value)) is not None
        }
        if metrics:
            try:
                self.mlflow.log_metrics(metrics, step=step)
            except Exception:
                logging.warning(
                    "MLflow metric delivery failed at step %s", step, exc_info=True
                )

    def _hardware_poll_loop(self, interval: int) -> None:
        while not self._stop_event.wait(interval):
            self._sample_hardware()

    def _sample_hardware(self) -> None:
        if self._client is None or self.run_id is None:
            return
        metrics, tags = _gpu_metrics()
        if not metrics:
            return
        try:
            from mlflow.entities import Metric

            timestamp = int(time.time() * 1000)
            with self._log_lock:
                self._client.log_batch(
                    self.run_id,
                    metrics=[
                        Metric(key, value, timestamp, int(self._last_step))
                        for key, value in metrics.items()
                    ],
                    params=[],
                    tags=[],
                )
            with self._tag_lock:
                for key, value in tags.items():
                    if key in self._gpu_tags_logged:
                        continue
                    self._client.set_tag(self.run_id, key, value)
                    self._gpu_tags_logged.add(key)
        except Exception:
            logging.debug("Could not record MLflow hardware metrics", exc_info=True)

    def close(self, status: str = "FINISHED") -> None:
        if self.mlflow is None:
            return
        self._stop_event.set()
        if self._hardware_thread is not None:
            self._hardware_thread.join(timeout=6)
        try:
            self.mlflow.end_run(status=status)
        except Exception:
            logging.warning("Could not finalize MLflow run", exc_info=True)
        finally:
            self.mlflow = None
