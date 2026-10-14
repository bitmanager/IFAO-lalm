# Native audio API: one-example smoke test, 2026-10-14

A direct `v1internal:generateContent` request accepted a WAV `inlineData` part
and returned a transcription, with no model tool step. This verifies transport
for this account, endpoint and model; it does not establish production quality,
quota savings, payload limits, or support for every model and audio format.

## Experiment

- Executed inside the existing `agy-03` container on dev-1, through its current network.
- Read existing Google OAuth credentials locally; credentials were not copied to
  a proxy or written into the repository. The access token was already valid.
- Discovery: `loadCodeAssist`, metadata `{"ideType":"ANTIGRAVITY"}`.
  An initial discovery with platform `LINUX` returned HTTP 400 (invalid enum);
  the corrected metadata returned HTTP 200 and a managed project.
- Generation: `https://daily-cloudcode-pa.googleapis.com/v1internal:generateContent`,
  model `gemini-3.8-flash-low`, native `contents[].parts[]`, one text and one WAV.
- Audio: SHA256 `6af1638bbb3e992bc009ef09713e37a597698e3ab4436f5fe7ff9df4dc63f5c1`,
  1.17 seconds, 37,484 bytes. User previously confirmed the speech as Russian
  **«я умею»**. The request did not include that reference or a Russian language hint.
- One generation POST, no tools and no retries. Discovery was separate from
  generation and its project response was cached locally.

## Result

HTTP **200**, **6.5 seconds**, response `modelVersion=gemini-3.8-flash`.
Usage: 246 prompt tokens, 41 candidate tokens, 415 thought tokens, 702 total.

Returned `audio_heard=true`, `language=uk`, `transcript="Я вмію"`, no uncertainty.
This is wrong against the user's listening audit. The Russian precision gate
must reject it. Audio acceptance does not prove correct language identification.
Despite `responseMimeType=application/json`, the response text contained a JSON
Markdown fence; an adapter would need to handle and validate the actual response.

The saved earlier CLI attempt on this same audio returned Arabic `يوميو`.
Its trace contains two distinct `agent_response` steps with a `view_file` step
between them, while the summary reports only one **user turn**. Thus `num_turns=1`
does not mean a single generation step. The CLI trace does not directly count
upstream HTTP POSTs, so exact HTTP reduction is not measured there.

The earlier CLI attempt used 8,383 uncached input tokens, 16,249 cache-read tokens
and 110 output tokens. Its full worker time was 7.48 seconds. These were different
invocation times and request contexts, so the latency and token numbers are not
a controlled benchmark or a measurement of subscription quota consumption.

## Reproduce

`native_audio_probe.py` makes one generation request using a locally cached
discovery response and an existing OAuth token. Run it inside the existing
account container with its usual network and `PYTHONPATH=/opt/speech`:

```bash
python native_audio_probe.py \
  --audio /private/control.wav \
  --token-file /home/worker/.gemini/antigravity-cli/antigravity-oauth-token \
  --discovery-file /private/discovery.json \
  --output-dir /private/new-probe-run
```

The output directory must not exist. Raw responses stay in that private directory.
The script does not refresh auth, retry, update the queue, or accept data for training.
It has not been connected to production workers. CLI authentication and project
discovery remain native account setup; no new proxy stack is required for this test.

## Sources checked

- [Community gateway specification](https://github.com/NoeFabris/opencode-antigravity-auth/blob/main/docs/ANTIGRAVITY_API_SPEC.md):
  describes `generateContent`, the native envelope and discovery. Its old example
  headers and model list are not evidence of current runtime compatibility.
- [Current discovery metadata code](https://github.com/cortexkit/antigravity-auth/blob/main/packages/core/src/fingerprint.ts):
  uses only `ideType: ANTIGRAVITY` for discovery, consistent with the successful test.
- [Current native model definitions](https://github.com/cortexkit/antigravity-auth/blob/main/packages/core/src/model-registry.ts):
  includes the exact `gemini-3.8-flash-low` wire ID.
