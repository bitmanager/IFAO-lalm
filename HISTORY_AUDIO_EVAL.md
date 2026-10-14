# Fixed history facts: audio evaluation

This evaluation adapts the existing 16 cases from
`/home/alexesn/slam-asr-ru/validation/context-text/cases.jsonl` without changing
their messages or expected-fact rubrics. Six A/B pairs have identical current
utterances and different histories; four additional cases remove history.
Only the current user utterance is synthesized. All identical utterances
reuse exactly the same WAV, including no-history controls.

`lalm/prepare_history_audio_eval.py` calls the existing Lychee
`Teachers.synthesize` client and uses native Lhotse resampling and
`prepare_cut` / `prepare_asr_cut`. There is no new speech generator, evaluator,
model, trainer, teacher-answer generation or decoder. `prepare_asr_cut` is the
existing contextual task helper; the isolated `asr_cut` format validator is
not applicable to cases that deliberately include history.

The authored `expected` strings are saved as the answer reference and as
`expected_facts` metadata. They are concise factual rubrics, not canonical
natural-language answers. Native `evaluate_qa.py` can generate and save full
answers, but its exact/containment metric is not factual correctness here.
Compare the actual city/date/name/count and requested abstention against
`cases.jsonl`, retaining complete answers and unchanged references. No new
scoring framework is introduced.

## Ready artifacts

Exp container root:
`/runs/dev-storage/ifao-data/context-history-audio-eval-v1`.
Dev-1 root: `/mnt/local/drive1/ifao-data/context-history-audio-eval-v1`.

* `history-answer16.jsonl.gz`: all 16 answer cases.
* `history-asr16.jsonl.gz`: the same cases with transcript-task labels.
* `eval-answer.yaml` and `eval-asr.yaml`: separate with-history (12) and
  no-history (4) sets using the unchanged native evaluator.
* `cases.jsonl`: byte-for-byte copy of the fixed source cases.
* `raw-source/`, `audio/`, `sources.jsonl`: original 24 kHz TTS audio/client
  metadata, six 16 kHz WAVs, scripts, durations and hashes.
* `provenance.json`, `readiness.json`: TTS/voice/client revisions and CPU QC.

There are **six unique WAVs, 13.92 seconds** of unique audio and **36.48
seconds per task view** across 16 cases. Both views together contain 32 cuts
and 72.96 seconds of exposure, not additional unique speech.

The already running dev TTS at `http://10.20.0.11:19099` served
`bitmanagerai/Qwen3TTS-v3.7-mix50`, revision
`384ffff264f0407498f3ca7138871b9cf03f69f6`. All requests use its existing
ordinary `anastasia-1` voice from voice-bank revision
`860451b106bc799d5235f392a9380b509c7c2d31`. No service or GPU assignment was
changed. The unchanged existing client was copied from dev-1
`/home/se@bitmanager.ai/lychee-ru-pipeline/adapters.py` into the execution
directory; its hash is recorded. No student speech tokenizer was instantiated.

Every cut has `split=test`, `evaluation_only=true`,
`training_eligible=false`, and `main_training_connected=false`. Histories
remain text; no acoustic enrollment or history audio is supplied. Fields
`case_id`, `case_type`, `source_group_id` and `expected_facts` retain the
case/control relationship. There is no training manifest or training-config
reference to these data. The single shared voice does not test unseen voices.

## Reproduce the format adapter

Run with `CUDA_VISIBLE_DEVICES=''` and existing CPU dependencies. The adapter
uses the remote TTS service, not an exp GPU; output must not already exist.

```sh
python lalm/prepare_history_audio_eval.py \
  --cases assets/context_history_eval_cases.jsonl \
  --teacher-client /runs/dev-storage/ifao-data/context-history-eval-code/existing_teachers.py \
  --tts-url http://10.20.0.11:19099 --voice anastasia-1 \
  --tokenizer /runs/ifao-balalaika-export/epoch-2 \
  --output /runs/dev-storage/ifao-data/context-history-audio-eval-v1
```

The log is at sibling `context-history-eval-code/export.log`. Four CPU tests
pass, covering all 16 fixed cases, shared A/B/control audio, exact history
and system preservation, isolated task labels, and the existing ASR tests.
All WAVs decode to finite samples, with maximum peak 0.873955. A real native
CPU batch passed all 32 cuts: 16 ASR flags, exact decoded final-target labels,
masked history, and EOS token 151645 in every supervised target. The copied
source cases hash is
`4eb081d883d38af9a559ddd8567682b32ea8ca18a9aac684cd994e4c3b43a658`.
No exp GPU inference or human listening was performed. Acoustic ASR fidelity
remains pending; the TTS scripts alone do not certify what was spoken.
The unchanged upstream GigaAM evaluator can consume
`asr-qc-gigaam-v1/six-current-utterances.tsv` directly (six rows, standard
`path/duration/transcription` columns). Its frozen input hash and handoff
status are in `asr-qc-gigaam-v1/input-provenance.json`. The agent owning the
separately scheduled GPU1 evaluation slot is responsible for that single
baseline batch; this preparation task starts no exp GPU job.

For a scheduled native answer run, reuse the allocated epoch export and
ordinary `evaluate_qa.py` with
`data.test_data_config=/runs/dev-storage/ifao-data/context-history-audio-eval-v1/eval-answer.yaml`.
Do not set `+asr=true` for answer generation. For the independent ASR view use
`eval-asr.yaml` and `+asr=true`. Use the established native runtime
`PYTHONPATH=/ifao-auden/src:/deps:/gpu:/review/vendor:/olddeps:/flash` and keep
decoding settings fixed across epochs. This preparation task does not launch
those GPU evaluations.
