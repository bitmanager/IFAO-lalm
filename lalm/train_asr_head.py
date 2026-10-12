"""Train a copied ASR readout over the frozen audio-language model."""

import argparse
import copy
import json
import logging
import random
from pathlib import Path

import jiwer
import mlflow
import torch
from lhotse import CutSet, MonoCut
from lhotse.dataset import AudioSamples
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Wav2Vec2FeatureExtractor,
    set_seed,
)

from lalm_core.data_module import LALMDataset
from lalm_core.model import LALMConfig, LALMForConditionalGeneration, LALMProcessor
from lalm_core.model.gigaam_adapter import GigaAMAudioTower
from lalm_core.text import ASR_NORMALIZER

SYSTEM = (
    "Transcribe the current audio verbatim in its original language. "
    "Use conversation history only as auxiliary context. "
    "Do not replace spoken words with expected ones or respond to the utterance. "
    "Output only the transcription."
)


def prepare_row(row, processor):
    cut = MonoCut.from_dict(copy.deepcopy(row["cut"]))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "<|audio|>\nTranscribe only the current audio."},
    ]
    prefix = processor.apply_chat_template(
        messages, tokenize=False, enable_thinking=False, add_generation_prompt=True
    )
    cut.rendered_conversation = prefix + row["target"] + processor.tokenizer.eos_token
    cut.task = "asr"
    expanded_prefix = processor.replace_multimodal_special_tokens(
        [prefix], iter([processor.audio_token_length_fn(cut.num_samples)])
    )[0]
    prefix_len = len(processor.tokenizer(expanded_prefix)["input_ids"])
    return dict(row, prepared_cut=cut, prefix_len=prefix_len)


def train(config, output):
    set_seed(config["seed"])
    # Keep the attention numerics used by the validated kl_iter1 pilot.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    language_model = AutoModelForCausalLM.from_pretrained(
        config["language_model"],
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"], local_files_only=True
    )
    encoder = GigaAMAudioTower.from_checkpoint(config["audio_checkpoint"])
    projector = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)[
        "projector"
    ]
    downsample_rate = config["projector_downsample_rate"]
    model = LALMForConditionalGeneration(
        LALMConfig(
            text_config=language_model.config,
            audio_config=encoder.config,
            audio_dim=projector["linear1.weight"].shape[1] // downsample_rate,
            text_dim=language_model.config.hidden_size,
            projector_downsample_rate=downsample_rate,
            audio_token_id=tokenizer.convert_tokens_to_ids("<|audio|>"),
        ),
        language_model=language_model,
        audio_tower=encoder,
    )
    device = next(language_model.parameters()).device
    # The compact checkpoint defines the widths of the existing projector layers.
    for name in ("linear1", "linear2"):
        weight = projector[f"{name}.weight"]
        setattr(
            model.projector, name, torch.nn.Linear(weight.shape[1], weight.shape[0])
        )
    model.projector.to(device=device, dtype=torch.float32)
    model.projector.load_state_dict(projector)
    model.requires_grad_(False).eval()
    head = (
        copy.deepcopy(language_model.get_output_embeddings())
        .float()
        .requires_grad_(True)
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config["lr"], weight_decay=0, foreach=False
    )
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    processor = LALMProcessor(
        Wav2Vec2FeatureExtractor(do_normalize=False, return_attention_mask=True),
        tokenizer,
        encoder_name="gigaam",
        projector_downsample_rate=downsample_rate,
    )
    dataset = LALMDataset(AudioSamples(), processor)
    data = json.loads(Path(config["data"]).read_text())
    prepared = {
        split: [prepare_row(row, processor) for row in data[split]]
        for split in ("train", "valid")
    }
    if not prepared["train"] or not prepared["valid"]:
        raise ValueError("Both train and validation splits must be nonempty")
    mlflow.log_params(
        {
            "train_rows": len(prepared["train"]),
            "valid_rows": len(prepared["valid"]),
            "trainable_parameters": sum(p.numel() for p in head.parameters()),
        }
    )

    @torch.no_grad()
    def encode(row):
        batch = dataset[CutSet.from_cuts([row["prepared_cut"]])]
        batch = {
            key: value.to(device)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        # Only the transcript and EOS contribute to the ASR loss.
        batch["labels"][:, : row["prefix_len"]] = loss_fn.ignore_index
        with torch.autocast(device.type, dtype=torch.bfloat16):
            embeddings = model._merge_input_ids_with_audio_features(
                batch["input_ids"],
                language_model.get_input_embeddings()(batch["input_ids"]),
                batch["features"],
                feature_lens=batch["feature_lens"],
            )
            hidden = language_model.model(
                inputs_embeds=embeddings,
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        labels = batch["labels"][:, 1:]
        mask = labels != loss_fn.ignore_index
        return hidden[:, :-1][mask], labels[mask], embeddings[:, : row["prefix_len"]]

    def losses(hidden, labels):
        for start in range(0, len(labels), config["ce_chunk_tokens"]):
            end = start + config["ce_chunk_tokens"]
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = head(hidden[start:end])
            yield loss_fn(logits.float(), labels[start:end])

    @torch.no_grad()
    def evaluate(step):
        rows = []
        total_loss = total_tokens = 0
        original_head = language_model.get_output_embeddings()
        language_model.set_output_embeddings(head)
        try:
            for row in prepared["valid"]:
                hidden, labels, prefix = encode(row)
                total_loss += sum(float(loss) for loss in losses(hidden, labels))
                total_tokens += len(labels)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    tokens = language_model.generate(
                        inputs_embeds=prefix,
                        attention_mask=torch.ones(
                            prefix.shape[:2], device=device, dtype=torch.long
                        ),
                        max_new_tokens=config["max_new_tokens"],
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )[0]
                rows.append(
                    {
                        "id": row["id"],
                        "reference": row["target"],
                        "text": tokenizer.decode(
                            tokens, skip_special_tokens=True
                        ).strip(),
                    }
                )
        finally:
            language_model.set_output_embeddings(original_head)
        references = ASR_NORMALIZER([row["reference"] for row in rows])
        hypotheses = ASR_NORMALIZER([row["text"] for row in rows])
        metrics = {
            "valid/CE": total_loss / total_tokens,
            "valid/WER": jiwer.wer(references, hypotheses),
            "valid/CER": jiwer.cer(references, hypotheses),
        }
        mlflow.log_metrics(metrics, step=step)
        report = output / f"validation-{step}.json"
        report.write_text(
            json.dumps({"metrics": metrics, "rows": rows}, ensure_ascii=False, indent=2)
        )
        mlflow.log_artifact(str(report))

    def save(name, step):
        path = output / name
        temporary = path.with_suffix(".tmp")
        torch.save(
            {"asr_head": head.state_dict(), "step": step, "config": config}, temporary
        )
        temporary.replace(path)
        return path

    step = seen_examples = 0
    evaluate(step)
    for epoch in range(config["epochs"]):
        order = list(prepared["train"])
        random.shuffle(order)
        for start in range(0, len(order), config["gradient_accumulation"]):
            batch = order[start : start + config["gradient_accumulation"]]
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for row in batch:
                hidden, labels, _ = encode(row)
                for loss in losses(hidden, labels):
                    loss = loss / len(labels) / len(batch)
                    loss.backward()
                    loss_value += float(loss.detach())
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(),
                config["max_grad_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            step += 1
            seen_examples += len(batch)
            mlflow.log_metrics(
                {
                    "train/CE": loss_value,
                    "train/grad_norm": float(gradient_norm),
                    "train/seen_examples": seen_examples,
                },
                step=step,
            )
            logging.info("epoch=%d step=%d loss=%.5f", epoch + 1, step, loss_value)
            if config["save_every"] and step % config["save_every"] == 0:
                save(f"asr-head-{step}.pt", step)
        evaluate(step)
    return save("asr_head.pt", step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    config = json.loads(parser.parse_args().config.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO)
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(
        run_id=config.get("mlflow_run_id"), run_name=config.get("run_name")
    ):
        mlflow.log_params(config)
        final_head = train(config, output)
        # All training tensors go out of scope before the potentially slow upload.
        torch.cuda.empty_cache()
        mlflow.log_artifact(str(final_head))


if __name__ == "__main__":
    main()
