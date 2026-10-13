"""No-thinking response rendering and frozen-backbone activation recomputation."""

from functools import partial

import torch
from lalm_core.model import LALMProcessor
from lalm_core.task_loss import IGNORE_INDEX
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor


def render(conversation, tokenizer, audio_token="<|audio|>"):
    assert conversation[-1]["role"] == "assistant"
    chunks = []
    for turn in conversation[:-1]:
        content = turn["content"]
        if isinstance(content, list):
            content = " ".join(
                audio_token if item["type"] == "audio" else str(item["text"])
                for item in content
            )
        chunks.append({"role": turn["role"], "content": content})
    prefix = tokenizer.apply_chat_template(
        chunks, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert prefix.endswith("<think>\n\n</think>\n\n")
    answer = conversation[-1]["content"]
    assert isinstance(answer, str) and answer.strip()
    assert all(s not in answer for s in ["<|", "<think>", "</think>"])
    return prefix + answer + "<|im_end|>\n"


def install_rendering():
    import lalm_core.response_kl
    import prepare_conversation

    prepare_conversation._render_conversation = render
    lalm_core.response_kl._render_conversation = render


class KLProcessor(LALMProcessor):
    def _prepare_labels(self, input_ids, attention_mask):
        labels = super()._prepare_labels(input_ids, attention_mask)
        scaffold = self.tokenizer.encode(
            "<think>\n\n</think>\n\n", add_special_tokens=False
        )
        for row in labels:
            pos = (row != IGNORE_INDEX).nonzero().flatten()
            assert len(pos) > len(scaffold)
            assert row[pos[: len(scaffold)]].tolist() == scaffold, (
                "Expected masked no-thinking scaffold"
            )
            row[pos[: len(scaffold)]] = IGNORE_INDEX
        return labels


def processor(tokenizer_path, downsample_rate=2):
    install_rendering()
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    tok.add_special_tokens({"additional_special_tokens": ["<|audio|>"]})
    feature = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=False,
        return_attention_mask=True,
    )
    return KLProcessor(
        feature, tok, encoder_name="gigaam", projector_downsample_rate=downsample_rate
    )


def _checkpointed_eval_forward(original, hidden_states, *args, **kwargs):
    if torch.is_grad_enabled() and hidden_states.requires_grad:
        return checkpoint(original, hidden_states, *args, use_reentrant=False, **kwargs)
    return original(hidden_states, *args, **kwargs)


def install_eval_checkpointing(lm):
    """Activation recomputation works while frozen LM remains eval (no dropout)."""
    for layer in lm.model.layers:
        layer.forward = partial(_checkpointed_eval_forward, layer.forward)
