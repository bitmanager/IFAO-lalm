"""Demo adapter around the existing GigaAM and IFAO inference implementations."""
import io
import math
import threading
import time
from pathlib import Path

import hydra
import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf
from scipy.signal import resample_poly

from lalm_core.model import LALMForConditionalGeneration, LALMProcessor


def read_audio(payload, channel=0):
    audio, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    if channel >= audio.shape[1]:
        raise ValueError("Выбранного канала нет в записи")
    audio = audio[:, channel]
    if not np.isfinite(audio).all() or not 0.5 <= len(audio) / rate <= 30:
        raise ValueError("Нужна запись от 0,5 до 30 секунд без повреждённых отсчётов")
    divisor = math.gcd(rate, 16000)
    if rate != 16000:
        audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
    return audio


class DemoEngine:
    def __init__(self, checkpoint, rnnt_checkpoint):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Демо должно видеть ровно одну GPU")
        self.checkpoint = str(Path(checkpoint).resolve())
        self.lock = threading.Lock()
        self.model = LALMForConditionalGeneration.from_pretrained(
            self.checkpoint, torch_dtype=torch.bfloat16,
        ).cuda().eval()
        self.processor = LALMProcessor.from_pretrained(self.checkpoint)
        self.processor.tokenizer.padding_side = "left"
        # The frozen encoder is shared. Restore only the official RNNT head.
        raw = torch.load(rnnt_checkpoint, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(raw["hyper_parameters"]["model_cfg"])
        encoder_cfg = OmegaConf.create(self.model.config.audio_config.gigaam_config)
        cfg.encoder.flash_attn = encoder_cfg.encoder.flash_attn
        if cfg.encoder != encoder_cfg.encoder or cfg.preprocessor != encoder_cfg.preprocessor:
            raise ValueError("RNNT and projector checkpoints use different encoder configurations")
        self.head = hydra.utils.instantiate(cfg.head)
        self.head.load_state_dict({k.removeprefix("head."): v for k, v in raw["state_dict"].items()
                                   if k.startswith("head.")}, strict=True)
        self.head = self.head.cuda().eval()
        self.decoding = hydra.utils.instantiate(cfg.decoding)

    @torch.inference_mode()
    def baseline(self, audio):
        with self.lock:
            started = time.monotonic()
            tower = self.model.audio_tower.model
            tower.preprocessor.float()
            wav = torch.from_numpy(audio).unsqueeze(0).cuda()
            lengths = torch.tensor([len(audio)], device="cuda")
            with torch.autocast("cuda", enabled=False):
                encoded, lengths = tower(wav.float(), lengths)
                text = self.decoding.decode(self.head, encoded.float(), lengths)[0][0]
            return {"text": text, "seconds": time.monotonic() - started}

    @torch.inference_mode()
    def generate(self, audio, history, system, *, asr=False, text=None):
        with self.lock:
            started = time.monotonic()
            for turn in history:
                if turn["role"] not in ("user", "assistant") or "<|" in turn["content"]:
                    raise ValueError("История содержит недопустимую роль или служебные токены")
            if "<|" in system or (text is not None and "<|" in text):
                raise ValueError("В тексте присутствуют служебные токены")
            messages = [{"role": "system", "content": system}, *history]
            content = [{"type": "audio"}]
            if asr:
                content.append({"type": "text", "text": "Дословно расшифруй текущую аудиозапись. Выведи только её текст."})
            messages.append({"role": "user", "content": text if text is not None else content})
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, audio=None if text is not None else [audio],
                                    sampling_rate=16000, padding=True, return_tensors="pt").to("cuda")
            if inputs.input_ids.shape[-1] + 256 > 8192:
                raise ValueError("История превысила контекст 8192 токена. Начните новый разговор.")
            tokens = self.model.generate(**inputs, asr=asr, max_new_tokens=256, do_sample=False)
            return {"text": self.processor.batch_decode(tokens, skip_special_tokens=True)[0],
                    "seconds": time.monotonic() - started, "tokens": tokens.shape[-1],
                    "limit_reached": tokens.shape[-1] >= 256}
