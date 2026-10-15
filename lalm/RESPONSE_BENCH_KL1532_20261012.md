# KL1532: completed response benchmark, 12 October 2026

## What ran

Actual checkpoint inference on **exp-srv-rtx6000x4-02, GPU3**, then stock
**promptfoo 0.124.1 llm-rubric** using OpenRouter `openai/gpt-6-luna`.
No new infrastructure on prod/test. GPU3 was released after inference.
Only a format adapter was added; inference reuses the native checkpoint runner.

Checkpoint: `response-kl-best-step1532-860552e6f018.pt`.
SHA256: `860552e6f0183bd43f27cb79691c5b8b6f2995ad009d220dc2d2b1b253ce3310`.
Frozen GigaAM → trained projector (40,905,728 parameters) → frozen Qwen3.8-27B.
This checkpoint has no separate ASR head. We evaluate **agent responses**.

48/48 native generations completed: 10 main recordings plus two conditions of
one separate «учусь» diagnostic, each through four input paths. Each answer
received two binary checks: task fulfilment and factual grounding.
The final judge run completed **96 checks with zero technical errors**.
Eval ID: `eval-VLk-2026-10-12T19:22:28`.

## Raw judge scores, main ten recordings

| Input to the same Qwen | Task fulfilled | Grounded | Both pass |
|---|---:|---:|---:|
| Gemini reference text | 9/10 | 6/10 | 6/10 |
| Offline GigaAM text | 8/10 | 8/10 | 7/10 |
| Current audio/projector checkpoint | 7/10 | 8/10 | 5/10 |
| Recorded online ASR text | 9/10 | 9/10 | 8/10 |

These are **raw automated pilot scores**, not production accuracy. Ten cases
were purpose-selected; Gemini text is a pseudo-reference, not human gold.
The recordings are full user-channel WAVs ≤30 seconds, not necessarily single
online commits. Two suspicious references were excluded before inference.
No general superiority over GigaAM has been demonstrated.

## Manual review

- «Учусь», with recorded history: audio response was «Поняла, учитесь. А для
  каких целей рассматриваете курс — для работы, для себя, или чтобы сменить
  сферу?» Offline Giga/reference also understood studying. Online «мучаюсь»
  replay asked again whether the user works or studies. This is a real positive
  example against the old online transcript, not a win over offline Giga.
- Without history, audio merely answered «Понял». Its passing grade does not
  prove it understood studying. Online text produced «Сочувствую. Что случилось?»
- Joke request: audio answered «Привет! Да, конечно.» but told no joke;
  all text paths told a joke. Task failure, not hallucination.
- «...модель... блин»: audio invented a model named **Blin** and elaborated on
  local model deployment. Text paths did not make that mistake.
- Course price: audio interpreted “курс” as exchange rate. No-history ambiguity
  needs audio/context adjudication before treating this as a strict failure.

Both diagnostic conditions passed both checks for audio, reference and offline
Giga; both failed overall for the online-text path. These two conditions are
one recording and are deliberately excluded from the main denominator.
Recorded history has not been independently clock-aligned: no causal memory-gain
claim is made.

## Judge audit and limitations

Version 1 conflated missing task execution with unsupported claims and rejected
honest explanations of missing external tools. Version 2 explicitly separates
these criteria and accepts a brief acknowledgement when no action was requested.
Native outputs and datasets were not changed; both judge reports are retained.

A remaining false positive is visible: the judge rejects the reference branch's
suggestion to use the official pizza app as unsupported, while accepting similar
advice in the audio branch. **Grounding is not yet a validated release gate.**
Do not rank checkpoints from this small pilot alone. Preserve raw reasons and
manually adjudicate disagreements; rubric changes require rerunning every branch.
The judge is a public model alias, not a pinned dated upstream model version.

## Artifacts and repeatability

All full artifacts are on exp under:
`/mnt/local/drive2/checkpoint-response-bench-v1/`

- `native/manifest.json`, `native/results.json`: exact inputs and native outputs.
- `prepared-v2.json`: stock promptfoo providerOutput tests and complete rubric.
- `results-v2.html`, `results-v2.json`, `judge-v2.log`: final report and reasons.
- `results.html`, `results.json`: retained first grading run.
- `run-metadata.json`, `tools/deployment.json`: provenance and framework versions.

```sh
cd /mnt/local/drive2/checkpoint-response-bench-v1
tools/run-promptfoo eval -c prepared-v2.json --no-cache --no-share \
  --max-concurrency 8 --output repeat.json --output repeat.html
```

For another checkpoint, run the existing native runner on the identical manifest
cases/history/decoding with the new verified checkpoint SHA, then export via
`lalm/export_response_promptfoo.py`. Use a new result directory. Do not reuse
cached answers from the previous checkpoint. Promptfoo exit 100 means failing
quality assertions; inspect the separate evaluation error count.
