"""Native GigaAM/projector frontend for SGLang's external-embedding API.

The server runs the unchanged Qwen weights; no HF language model is loaded here.
Requires language-model-only mode and disabled radix caching on the server.
"""
import json
import base64
import time
from pathlib import Path

import requests
import torch
from safetensors import safe_open

import kl_runtime as runtime
from lalm_core.model.modeling_lalm import LALMProjector
from lalm_core.model.gigaam_adapter import GigaAMAudioTower, gigaam_output_length


class SGLangClient:
    def __init__(self, checkpoint, endpoint, embedding_format="json", *,
                 encoder_name="gigaam", encoder_path=None):
        if embedding_format not in ("json", "bytes"):
            raise ValueError("Unknown embedding format")
        self.embedding_format = embedding_format
        self.endpoint = endpoint.rstrip("/")
        self.http = requests.Session()
        response = self.http.get(self.endpoint + "/server_info", timeout=30)
        response.raise_for_status()
        self.server_info = response.json()
        args = self.server_info.get("server_args", self.server_info)
        override = json.loads(args.get("json_model_override_args") or "{}")
        if not args.get("disable_radix_cache") or not (args.get("language_model_only") or override.get("language_model_only")):
            raise ValueError("External embeddings require no radix cache and language-model-only")
        snapshot = Path(json.loads((runtime.ROOT / "model_download.json").read_text())["path"])
        if Path(args["model_path"]) != snapshot:
            raise ValueError("SGLang is serving a different base model")
        # Do not fingerprint changing throughput/KV counters (or API credentials).
        self.server_recipe = {k: args.get(k) for k in (
            "model_path", "tokenizer_path", "dtype", "quantization", "kv_cache_dtype",
            "json_model_override_args", "language_model_only", "disable_radix_cache",
            "context_length", "attention_backend", "mamba_backend", "mamba_ssm_dtype",
            "cuda_graph_backend_decode", "cuda_graph_backend_prefill", "tp_size", "version")}
        self.server_recipe["wrapper_sha256"] = runtime.digest(Path(__file__).parent / "qwen3_vl.py")
        self.server_recipe["embedding_format"] = embedding_format
        if embedding_format == "bytes":
            patch = Path(__file__).parent / "input-embeds-bytes-v0.5.21.patch"
            self.server_recipe["transport_patch_sha256"] = runtime.digest(patch)
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        keys = [k for k in index if k.endswith("embed_tokens.weight")]
        if len(keys) != 1:
            raise ValueError("Expected one text embedding tensor")
        with safe_open(snapshot / index[keys[0]], framework="pt", device="cpu") as handle:
            self.embedding = torch.nn.Embedding.from_pretrained(
                handle.get_tensor(keys[0]).to(torch.bfloat16), freeze=True)
        self.tokenizer = runtime.processor().tokenizer
        self.encoder_name = encoder_name
        if encoder_name == "gigaam":
            if encoder_path is not None:
                raise ValueError("GigaAM benchmark uses the frozen runtime checkpoint")
            self.encoder = GigaAMAudioTower.from_checkpoint(runtime.CKPT).requires_grad_(False).eval()
            self.output_length = gigaam_output_length
            audio_dim = 768
        elif encoder_name == "gemma":
            from lalm_core.model.gemma_adapter import GemmaAudioTower, gemma_output_length
            if encoder_path is None:
                raise ValueError("Gemma requires the frozen encoder snapshot path")
            # Match the native Gemma training/validation frontend (not the server).
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            self.encoder = GemmaAudioTower.from_pretrained(encoder_path).to("cuda").requires_grad_(False).eval()
            self.output_length = gemma_output_length
            audio_dim = self.encoder.audio_dim
            self.server_recipe["audio_encoder"] = {
                "name": encoder_name, "path": str(encoder_path),
                "adapter_sha256": runtime.digest(Path(__import__(
                    "lalm_core.model.gemma_adapter", fromlist=["__file__"]).__file__)),
                "files": {p.name: runtime.digest(p) for p in sorted(Path(encoder_path).iterdir())
                          if p.is_file() and (p.suffix in (".json", ".safetensors"))},
            }
        else:
            raise ValueError("Unknown audio encoder")
        self.projector = LALMProjector(audio_dim, 5120, downsample_rate=2, hidden_size=6144)
        self.projector.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["projector"])
        self.projector.to("cuda", dtype=torch.float32).requires_grad_(False).eval()

    @torch.inference_mode()
    def embeddings(self, prefix, wave=None):
        ids = torch.tensor(prefix)
        embeds = self.embedding(ids)
        if wave is not None:
            lengths = torch.tensor([wave.numel()], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.encoder_name == "gemma"):
                encoded = self.encoder(wave.unsqueeze(0).cuda(), lengths)
            projected = self.projector(encoded.float())[0].to(torch.bfloat16).cpu()
            mask = ids == self.tokenizer.convert_tokens_to_ids("<|audio|>")
            expected = int(self.output_length(lengths)[0]) // 2
            if int(mask.sum()) != expected or len(projected) != expected:
                raise ValueError("Audio placeholder/projector lengths differ")
            embeds[mask] = projected
        return embeds.float().numpy()

    def generate(self, prefix, wave=None, *, text_embeds=False, embedding_format=None):
        import orjson

        started = time.monotonic()
        suppressed = sorted({t for marker in ("<think>", "</think>")
                             for t in self.tokenizer.encode(marker, add_special_tokens=False)})
        payload = {"sampling_params": {
            "temperature": 0, "max_new_tokens": 384,
            "stop_token_ids": [self.tokenizer.eos_token_id],
            "logit_bias": {str(t): -1e9 for t in suppressed},
        }}
        if wave is None and not text_embeds:
            payload["input_ids"] = prefix
        else:
            embeds = self.embeddings(prefix, wave)
            if (embedding_format or self.embedding_format) == "bytes":
                payload["input_embeds_bytes"] = base64.b64encode(embeds.tobytes()).decode("ascii")
                payload["input_embeds_shape"] = list(embeds.shape)
            else:
                payload["input_embeds"] = embeds
        response = self.http.post(
            self.endpoint + "/generate",
            data=orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY),
            headers={"Content-Type": "application/json"}, timeout=600)
        response.raise_for_status()
        result = response.json()
        meta = result["meta_info"]
        finish = meta["finish_reason"]
        if finish["type"] not in ("stop", "length"):
            raise RuntimeError(f"SGLang generation failed: {finish}")
        text = result["text"].strip()
        if "<think>" in text or "</think>" in text:
            raise ValueError("Thinking delimiters escaped suppression")
        return dict(text=text, complete=finish["type"] == "stop",
                    tokens=meta["completion_tokens"], seconds=time.monotonic() - started,
                    finish_reason=finish)


def generated(model, tok, prefix, wave=None):
    return model.generate(prefix, wave)
