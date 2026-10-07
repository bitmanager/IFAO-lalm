"""Experimental audio-history adapter; encoder, projector and decoding stay native."""
import time

import torch


@torch.inference_mode()
def encode_turn(engine, audio):
    inputs = engine.processor(text=engine.processor.audio_token, audio=[audio],
                              sampling_rate=16000, return_tensors="pt").to("cuda")
    projected = engine.model.projector(engine.model.encode_audio(
        inputs["input_features"], inputs["feature_lens"]))
    length = int(engine.model._get_audio_output_lengths(inputs["feature_lens"])[0])
    length //= engine.model.projector.downsample_rate
    if length <= 0 or projected.shape[1] != length:
        raise ValueError("Audio turn has an invalid projected length")
    return projected[0]


@torch.inference_mode()
def generate_audio_history(engine, messages, audio_embeddings, system):
    started = time.monotonic()
    prompt = engine.processor.apply_chat_template(
        [{"role": "system", "content": system}, *messages], add_generation_prompt=True)
    if prompt.count(engine.processor.audio_token) != len(audio_embeddings):
        raise ValueError("Every audio placeholder requires exactly one encoded turn")
    rendered = engine.processor.replace_multimodal_special_tokens(
        [prompt], iter(len(embedding) for embedding in audio_embeddings))
    inputs = engine.processor.tokenizer(rendered, return_tensors="pt").to("cuda")
    context_limit = engine.model.config.text_config.max_position_embeddings
    if inputs.input_ids.shape[-1] + 256 > context_limit:
        raise ValueError("Full audio history exceeds the model context; no truncation")
    embeds = engine.model.language_model.get_input_embeddings()(inputs.input_ids)
    mask = inputs.input_ids == engine.model.config.audio_token_id
    audio = torch.cat(audio_embeddings).to(embeds.dtype)
    if int(mask.sum()) != len(audio):
        raise ValueError("Audio-history frame count does not match placeholders")
    embeds[mask] = audio
    tokens = engine.model.language_model.generate(
        inputs_embeds=embeds, attention_mask=inputs.attention_mask,
        max_new_tokens=256, do_sample=False)
    return {"text": engine.processor.batch_decode(tokens, skip_special_tokens=True)[0],
            "seconds": time.monotonic() - started, "tokens": tokens.shape[-1],
            "input_tokens": inputs.input_ids.shape[-1],
            "audio_tokens": len(audio), "audio_turns": len(audio_embeddings),
            "limit_reached": tokens.shape[-1] >= 256}
