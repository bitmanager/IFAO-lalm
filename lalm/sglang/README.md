# Qwen external embeddings on SGLang

This adapter serves the same Qwen3.8-27B checkpoint (whose config uses the
`Qwen3_5ForConditionalGeneration` architecture), BF16, with the existing FP32
GigaAM projector. It does not substitute a text transcript for audio.

SGLang v0.5.21 accepts `/generate.input_embeds`; its Qwen3.5 backbone also accepts
them, but the inherited Qwen3-VL outer wrapper does not forward them. The included
four-line patch connects those existing interfaces, restricted to standalone
`language_model_only=True` model configuration. Model layers, weight loading, logits processor and
generation remain SGLang's implementation. Apply the patch to the pinned image's
source and bind mount that file; do not modify the shared image or training code.
The config is set with `--json-model-override-args`: v0.5.21's separate
`--language-model-only` CLI flag has an architecture allowlist excluding Qwen,
while its model-config path and Qwen wrapper already implement this mode.

Existing implementation references:

- [SGLang sampling API](https://github.com/sgl-project/sglang/blob/main/docs/docs/basic_usage/sampling_params.mdx)
- [NLA inference external-embedding client](https://github.com/kitft/nla-inference):
  load only the embedding table and use the native generation endpoint.
- [Analogous Gemma wrapper fix](https://github.com/kitft/natural_language_autoencoders/blob/main/patches/nla_gemma3_mm_input_embeds.patch):
  a precedent, not a drop-in patch for Qwen.
- [SGLang Omni prefill adapter](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/model_runner/prefill_inputs.py):
  a larger official speech pipeline, unnecessary for this standalone projector.

`run-server.sh` reserves only the authorized GPU1 by UUID, binds HTTP to loopback,
disables radix caching (different audio can have identical placeholder IDs), and
uses a private MPS pipe path. GigaAM/projector run separately on the same card.
No second 27B model is loaded in the client. Model data and caches stay on exp.

Run `canary.py` before benchmarking: text IDs versus equivalent text embeddings,
real audio versus silence, and saved HF reference outputs. This is a correctness
gate, not evidence of improved model quality. Kernels and accumulation differ
from HF; semantic/metric comparisons must remain separate even when greedy text
is identical. The native 384-token generation limit and no-thinking chat template
are retained; SGLang's native logit bias (-1e9) suppresses the two thinking markers.

Use `benchmark_qwen_calls.py --sglang-url http://127.0.0.1:18139` with a separate
output directory. Backend metadata is recorded in each result fingerprint.
Never overwrite or combine the completed HF run with these outputs.

## Binary transport experiment

`input-embeds-bytes-v0.5.21.patch` ports the existing implementation from
[SGLang PR #20207](https://github.com/sgl-project/sglang/pull/20207) (including
its NumPy IPC prerequisite, PR #20205). This is an unmerged upstream proposal,
not a feature of the stock release. Port adjustments use v0.5.21's existing
`Annotated[..., Base64Bytes()]` marker, retain current chunk slicing and its
memory-config accessor, and preserve current unrelated request checks.
GPU testing also exposed a PyTorch incompatibility in the proposed PR:
`torch.tensor(numpy_array, pin_memory=True)` fails in this runtime. The port
uses the standard two-step `torch.from_numpy(...).pin_memory()` operation
(pinning only when the existing `_pin` flag is enabled), then the same device copy.

Apply this patch to the pinned image's three manager files under
`bytes-pr/python/sglang/srt/managers/`; the launcher bind-mounts those files.
The original stock image stays intact. `test_input_embeds_bytes.py` tests the
HTTP schema, exact float32 reconstruction and malformed request rejection.
`canary.py --bytes` additionally requires identical JSON/binary generated text.
Select binary transport with `--sglang-embedding-format bytes`; no quantization
or embedding normalization is introduced. Keep its outputs in a separate run.

Motivation measured on this host: approximately 15 of 20 seconds sampled in the
HTTP process were spent in Pydantic validation of nested float lists. Four JSON
clients did not outperform the previous serial HF run on the initial matched
subset. Speed improvements must be measured, not inferred from engine choice.

Initial binary run: 223 matched segments in 76.6 seconds, versus 185.7 seconds
of recorded serial HF generation for the same segments (approximately 2.43x;
preliminary, not a full-run comparison). Median request latency was 0.939 seconds
versus 4.306 seconds in the four-client JSON run on that matched subset.

The three controlled transport cases passed text parity. This does **not** prove
global deterministic generation: on a separate ambiguous fragment, repeated JSON
requests with the exact same precomputed vectors produced both `фобия` and
`фотобанк`; binary requests produced `фобия`. The isolated probe is retained on
exp as `transport-disagreement-probe.json`. Do not describe all outputs as
identical or assume checkpoint metrics are backend-invariant; complete WER/judge
results are exported separately for this runtime.
