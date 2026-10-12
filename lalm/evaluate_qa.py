"""QA evaluation / decoding script for lalm.

Assumes test manifests have been prepared by prepare_conversation_qa.py.

Outputs a plain-text results file (id | hyp | ref) and an accuracy summary.

Usage::

    python evaluate_qa.py \
        exp_dir=/path/to/exp \
        checkpoint.iter=5000 \
        data.test_data_config=configs/test_qa_data_config.yaml
"""

import hashlib
import logging
import os
import re
import string
import time
from pathlib import Path

import hydra
import torch
import yaml
from lalm_core.mlflow_monitoring import monitoring_enabled
from lalm_core.model import LALMConfig, LALMForConditionalGeneration, LALMProcessor
from lhotse import CutSet, set_audio_duration_mismatch_tolerance
from lhotse.dataset import DynamicBucketingSampler
from omegaconf import DictConfig, OmegaConf
from transformers.modeling_utils import no_init_weights

# ---------------------------------------------------------------------------
# Model loading  (reused from evaluate.py)
# ---------------------------------------------------------------------------


def _has_hf_weights(model_dir: str) -> bool:
    return any(
        os.path.exists(os.path.join(model_dir, n))
        for n in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def _resolve_checkpoint_path(cfg: DictConfig) -> tuple[str, str]:
    ckpt_cfg = cfg.checkpoint
    filename = ckpt_cfg.get("filename", None)
    if filename:
        path = (
            filename if os.path.isabs(filename) else os.path.join(cfg.exp_dir, filename)
        )
        return path, Path(filename).stem

    iters = int(ckpt_cfg.get("iter", 0))
    epoch = int(ckpt_cfg.get("epoch", 0))
    if iters > 0:
        return os.path.join(cfg.exp_dir, f"checkpoint-{iters}.pt"), f"iter-{iters}"
    if epoch > 0:
        return os.path.join(cfg.exp_dir, f"epoch-{epoch}.pt"), f"epoch-{epoch}"
    raise ValueError(
        "[evaluate_qa] Specify checkpoint.filename, checkpoint.iter, or checkpoint.epoch."
    )


def prepare_model_dir(cfg: DictConfig) -> tuple[str, str]:
    explicit_dir = cfg.checkpoint.get("model_dir", None)
    checkpoint_path, results_suffix = _resolve_checkpoint_path(cfg)

    model_dir = explicit_dir or os.path.join(cfg.exp_dir, "export", results_suffix)
    if explicit_dir:
        results_suffix = Path(explicit_dir).name

    if _has_hf_weights(model_dir):
        logging.info(f"[evaluate_qa] Using existing model dir: {model_dir}")
        return model_dir, results_suffix

    logging.info(f"[evaluate_qa] Exporting {checkpoint_path} -> {model_dir}")
    hf_dir = os.path.join(cfg.exp_dir, "hf")
    if not os.path.isdir(hf_dir):
        raise FileNotFoundError(
            f"[evaluate_qa] hf_dir not found: {hf_dir}. Run train.py first."
        )

    with no_init_weights():
        model = LALMForConditionalGeneration(LALMConfig.from_pretrained(hf_dir))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    LALMProcessor.from_pretrained(hf_dir).save_pretrained(model_dir)
    logging.info(f"[evaluate_qa] Saved to {model_dir}")
    del model, checkpoint, state_dict
    return model_dir, results_suffix


# ---------------------------------------------------------------------------
# QA matching with tolerance
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    """Lowercase, strip whitespace and trailing punctuation."""
    s = s.strip().lower()
    # Remove trailing punctuation (e.g., "Train." -> "train")
    s = s.rstrip(string.punctuation)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _match(hyp: str, ref: str) -> bool:
    """Tolerant matching for QA evaluation.

    Returns True if:
    1. Exact match after normalization, OR
    2. Normalized ref is contained in normalized hyp
       (handles "The answer is Train" matching ref "Train"), OR
    3. For multiple-choice: the ref appears as a standalone word in hyp.
    """
    norm_hyp = _normalize_answer(hyp)
    norm_ref = _normalize_answer(ref)

    if not norm_ref:
        return False

    # Exact match
    if norm_hyp == norm_ref:
        return True

    # Containment: ref appears in hyp
    if norm_ref in norm_hyp:
        return True

    # Word-boundary match (e.g., ref="a" should match "a" but not "a train")
    # Only apply for short refs (single word or letter) to avoid false positives
    if len(norm_ref.split()) == 1:
        pattern = r"\b" + re.escape(norm_ref) + r"\b"
        if re.search(pattern, norm_hyp):
            return True

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_ASST_PREFIX = "<|im_start|>assistant\n"


def _strip_last_assistant(rendered: str) -> str:
    idx = rendered.rfind(_ASST_PREFIX)
    if idx == -1:
        return rendered + _ASST_PREFIX
    return rendered[: idx + len(_ASST_PREFIX)]


def _file_sha256(path_value):
    path = Path(str(path_value))
    if not path.is_file():
        return "unavailable"
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _evaluate(cfg, model_dir, results_file_suffix, mlflow_module=None):
    dtype_name = cfg.get("dtype", "fp16")
    dtype = (
        torch.float16
        if dtype_name == "fp16"
        else torch.bfloat16 if dtype_name == "bf16" else torch.float32
    )

    model = LALMForConditionalGeneration.from_pretrained(model_dir, torch_dtype=dtype)
    processor = LALMProcessor.from_pretrained(model_dir)
    processor.tokenizer.padding_side = "left"
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device).eval()
    logging.info(
        f"[evaluate_qa] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )

    max_new_tokens = int(cfg.get("max_new_tokens", 200))
    generate_config = (
        dict(max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
        if cfg.decoding_method == "greedy_search"
        else dict(
            max_new_tokens=max_new_tokens,
            num_beams=int(cfg.get("num_beams", 4)),
            do_sample=False,
        )
    )

    sampling_rate = int(cfg.data.get("sampling_rate", 16000))

    with open(cfg.data.test_data_config) as f:
        test_data_config = yaml.load(f, Loader=yaml.FullLoader)

    res_dir = Path(cfg.exp_dir) / cfg.decoding_method
    os.makedirs(res_dir, exist_ok=True)

    suffix_str = f"-{results_file_suffix}" if results_file_suffix else ""

    for test_set in test_data_config:
        logging.info(f"[evaluate_qa] Test set: {test_set['name']}")
        cutset = CutSet.from_file(test_set["manifest"]).resample(sampling_rate)
        sampler = DynamicBucketingSampler(
            cutset, max_duration=cfg.data.max_duration, shuffle=False
        )

        all_results = []  # list of (cut_id, hyp_str, ref_str)
        num_cuts = 0
        generation_seconds = 0.0
        batch_generation_seconds = []

        for batch_idx, cuts in enumerate(sampler):
            cuts = cuts.sort_by_duration(ascending=False)

            audios = [cut.load_audio()[0] for cut in cuts]
            texts = [_strip_last_assistant(cut.rendered_conversation) for cut in cuts]

            inputs = processor(
                text=texts,
                audio=audios,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            ).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_started = time.perf_counter()
            output_ids = model.generate(**inputs, **generate_config)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_elapsed = time.perf_counter() - batch_started
            generation_seconds += batch_elapsed
            batch_generation_seconds.append(batch_elapsed)
            hyps = processor.batch_decode(output_ids, skip_special_tokens=True)
            refs = [
                cut.supervisions[0].custom.get("answer", cut.supervisions[0].text)
                for cut in cuts
            ]
            cut_ids = [cut.id for cut in cuts]

            for cut_id, ref, hyp in zip(cut_ids, refs, hyps):
                all_results.append((cut_id, hyp.strip(), ref.strip()))

            num_cuts += len(cut_ids)
            if batch_idx % 50 == 0:
                logging.info(f"  batch {batch_idx}, cuts: {num_cuts}")

        # --- Save results: id | hyp | ref ---
        results_path = res_dir / f"qa_results-{test_set['name']}{suffix_str}.txt"
        with open(results_path, "w", encoding="utf-8") as f:
            for cut_id, hyp, ref in sorted(all_results):
                f.write(f"{cut_id} | {hyp} | {ref}\n")
        logging.info(f"[evaluate_qa] Results saved to {results_path}")

        # --- Compute accuracy ---
        correct = sum(1 for _, hyp, ref in all_results if _match(hyp, ref))
        total = len(all_results)
        acc = correct / total * 100 if total > 0 else 0.0

        acc_path = res_dir / f"accuracy-{test_set['name']}{suffix_str}.txt"
        with open(acc_path, "w", encoding="utf-8") as f:
            f.write(f"Accuracy: {acc:.2f}% ({correct}/{total})\n")
        logging.info(
            f"[evaluate_qa] {test_set['name']} Accuracy: {acc:.2f}% ({correct}/{total})"
        )

        if mlflow_module is not None:
            metric_set = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", str(test_set["name"])
            ).strip("_")
            metric_set = metric_set or "unnamed"
            metrics = {
                f"eval/{metric_set}/accuracy_percent": acc,
                f"eval/{metric_set}/correct": float(correct),
                f"eval/{metric_set}/samples": float(total),
                f"eval/{metric_set}/generation_seconds": generation_seconds,
            }
            if generation_seconds > 0:
                metrics[f"eval/{metric_set}/examples_per_second"] = (
                    total / generation_seconds
                )
            if batch_generation_seconds:
                ordered = sorted(batch_generation_seconds)
                p95_index = min(int(0.95 * len(ordered)), len(ordered) - 1)
                metrics[f"eval/{metric_set}/batch_latency_p95_seconds"] = (
                    ordered[p95_index]
                )
                metrics[f"eval/{metric_set}/batch_latency_mean_seconds"] = (
                    generation_seconds / len(ordered)
                )
            try:
                mlflow_module.log_metrics(metrics, step=0)
            except Exception:
                logging.warning("MLflow evaluation metric delivery failed", exc_info=True)


@hydra.main(version_base=None, config_path="configs", config_name="evaluate_qa")
def main(cfg: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    set_audio_duration_mismatch_tolerance(0.1)
    model_dir, results_file_suffix = prepare_model_dir(cfg)

    monitoring = cfg.get("monitoring", {})
    if not monitoring_enabled(cfg):
        return _evaluate(cfg, model_dir, results_file_suffix)

    import mlflow

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        str(monitoring.get("tracking_uri", "http://127.0.0.1:5000")),
    )
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        str(monitoring.get("experiment_name", "SpeechLLM R&D")),
    )
    sampling_interval = int(monitoring.get("system_metrics_sampling_interval", 10))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.set_system_metrics_sampling_interval(sampling_interval)
    mlflow.set_system_metrics_samples_before_logging(
        int(monitoring.get("system_metrics_samples_before_logging", 1))
    )
    run_name = str(
        monitoring.get("run_name")
        or os.getenv("MLFLOW_RUN_NAME")
        or f"qa-{results_file_suffix}"
    )

    with mlflow.start_run(
        run_name=run_name,
        log_system_metrics=bool(monitoring.get("log_system_metrics", True)),
        tags={
            "project": "speechllm-rd",
            "run_type": "qa-evaluation",
            "checkpoint": results_file_suffix,
        },
    ):
        mlflow.log_params(
            {
                "evaluation.checkpoint": results_file_suffix,
                "evaluation.decoding_method": str(cfg.decoding_method),
                "evaluation.dtype": str(cfg.get("dtype", "fp16")),
                "evaluation.max_new_tokens": int(cfg.get("max_new_tokens", 200)),
                "evaluation.num_beams": int(cfg.get("num_beams", 1)),
                "evaluation.test_config_sha256": _file_sha256(
                    cfg.data.test_data_config
                ),
            }
        )
        _evaluate(cfg, model_dir, results_file_suffix, mlflow)


if __name__ == "__main__":
    main()
