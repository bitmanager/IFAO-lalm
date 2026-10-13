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


def _scoped_forward(original, *args, **kwargs):
    token = _segments.set(kwargs.get("packed_segment_lens", _segments.get()))
    try:
        return original(*args, **kwargs)
    finally:
        _segments.reset(token)


def _segment_linear_forward(original, x):
    lengths = _segments.get()
    if (
        lengths
        and len(lengths) > 1
        and x.ndim == 3
        and x.shape[:2] == (1, sum(lengths))
    ):
        return torch.cat([original(part) for part in x.split(lengths, dim=1)], dim=1)
    return original(x)


def install_segment_linear_routes(language_model):
    """Use identical BF16 GEMM shapes per sample, including recomputation.

    Blackwell's packed and single-sample GEMMs amplify rounding differences
    through this deep backbone. Attention still runs in one packed pass.
    """
    if getattr(language_model, "_segment_linear_routes", False):
        return

    language_model.forward = functools.update_wrapper(
        functools.partial(_scoped_forward, language_model.forward), language_model.forward
    )
    for layer in language_model.model.layers:
        # HF checkpointing re-enters each layer after the outer forward returns.
        layer.forward = functools.update_wrapper(
            functools.partial(_scoped_forward, layer.forward), layer.forward
        )
    for module in language_model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = functools.partial(_segment_linear_forward, module.forward)
    language_model._segment_linear_routes = True


def _packed_conv_forward(original, hidden_states, weight, bias=None, activation=None, **kwargs):
    lengths = kwargs.pop("packed_segment_lens", None)
    if not lengths or len(lengths) == 1:
        return original(hidden_states, weight, bias, activation, **kwargs)
    if hidden_states.shape[0] != 1 or hidden_states.shape[-1] != sum(lengths):
        raise ValueError("Packed convolution boundaries do not match its input")
    pieces = torch.split(hidden_states, lengths, dim=-1)
    return torch.cat(
        [original(p, weight, bias, activation, **kwargs) for p in pieces], dim=-1
    )


def _packed_delta_forward(original, query, key, value, g, beta, **kwargs):
    lengths = kwargs.pop("packed_segment_lens", None)
    if not lengths or len(lengths) == 1:
        return original(query, key, value, g, beta, **kwargs)
    if kwargs.get("initial_state") is not None or kwargs.get("output_final_state", False):
        raise ValueError("Multi-sample packing requires cache disabled")
    if query.shape[0] != 1 or query.shape[1] != sum(lengths):
        raise ValueError("Packed DeltaNet boundaries do not match its input")
    pieces = [torch.split(t, lengths, dim=1) for t in (query, key, value, g, beta)]
    outputs = [original(*values, **kwargs)[0] for values in zip(*pieces)]
    return torch.cat(outputs, dim=1), None


def install_packed_qwen_kernels():
    global _installed
    if _installed:
        return
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native

    native.causal_conv1d_fn = functools.partial(_packed_conv_forward, native.causal_conv1d_fn)
    native.torch_chunk_gated_delta_rule = functools.partial(
        _packed_delta_forward, native.torch_chunk_gated_delta_rule
    )
    _installed = True
