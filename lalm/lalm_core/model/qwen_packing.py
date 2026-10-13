"""Keep packed Qwen3.8 samples isolated in its temporal convolution and DeltaNet.

The stock reference kernels accept extra kwargs but ignore sequence boundaries.
Reuse those exact kernels per segment; full attention and tokenwise layers remain
packed. No pretrained parameters or numerical operations are changed.
"""

import contextvars
import functools

import torch

_installed = False
_segments = contextvars.ContextVar("qwen_packed_segments", default=None)


def install_segment_linear_routes(language_model):
    """Use identical BF16 GEMM shapes per sample, including recomputation.

    Blackwell's packed and single-sample GEMMs amplify rounding differences
    through this deep backbone. Attention still runs in one packed pass.
    """
    if getattr(language_model, "_segment_linear_routes", False):
        return

    def scoped_forward(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            token = _segments.set(kwargs.get("packed_segment_lens", _segments.get()))
            try:
                return original(*args, **kwargs)
            finally:
                _segments.reset(token)

        return wrapped

    language_model.forward = scoped_forward(language_model.forward)
    for layer in language_model.model.layers:
        # HF checkpointing re-enters each layer after the outer forward returns.
        layer.forward = scoped_forward(layer.forward)
    for module in language_model.modules():
        if isinstance(module, torch.nn.Linear):
            original = module.forward

            def linear(x, original=original):
                lengths = _segments.get()
                if (
                    lengths
                    and len(lengths) > 1
                    and x.ndim == 3
                    and x.shape[:2] == (1, sum(lengths))
                ):
                    return torch.cat(
                        [original(part) for part in x.split(lengths, dim=1)], dim=1
                    )
                return original(x)

            module.forward = linear
    language_model._segment_linear_routes = True


def install_packed_qwen_kernels():
    global _installed
    if _installed:
        return
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native

    original_conv = native.causal_conv1d_fn
    original_delta = native.torch_chunk_gated_delta_rule

    def conv(hidden_states, weight, bias=None, activation=None, **kwargs):
        lengths = kwargs.pop("packed_segment_lens", None)
        if not lengths or len(lengths) == 1:
            return original_conv(hidden_states, weight, bias, activation, **kwargs)
        if hidden_states.shape[0] != 1 or hidden_states.shape[-1] != sum(lengths):
            raise ValueError("Packed convolution boundaries do not match its input")
        pieces = torch.split(hidden_states, lengths, dim=-1)
        return torch.cat(
            [original_conv(p, weight, bias, activation, **kwargs) for p in pieces],
            dim=-1,
        )

    def delta(query, key, value, g, beta, **kwargs):
        lengths = kwargs.pop("packed_segment_lens", None)
        if not lengths or len(lengths) == 1:
            return original_delta(query, key, value, g, beta, **kwargs)
        if kwargs.get("initial_state") is not None or kwargs.get(
            "output_final_state", False
        ):
            raise ValueError("Multi-sample packing requires cache disabled")
        if query.shape[0] != 1 or query.shape[1] != sum(lengths):
            raise ValueError("Packed DeltaNet boundaries do not match its input")
        pieces = [torch.split(t, lengths, dim=1) for t in (query, key, value, g, beta)]
        outputs = [original_delta(*values, **kwargs)[0] for values in zip(*pieces)]
        return torch.cat(outputs, dim=1), None

    native.causal_conv1d_fn = conv
    native.torch_chunk_gated_delta_rule = delta
    _installed = True
