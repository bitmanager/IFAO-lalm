# Two-account native API canary, 2026-10-14

The later 30-audio, four-account run is documented in
[`native-api-expanded-20261014.md`](native-api-expanded-20261014.md).

The native audio transport completed a bounded experiment on `agy-03` and
`agy-16`: six identical audios per slot, three independent prompts per audio,
at most five simultaneous requests per slot. Both slots automatically resumed
their normal CLI queue afterward. This experiment is review-only and is not a
training-data release or a decision to switch the production transport.

## Transport and response checks

| Metric | agy-03 | agy-16 |
|---|---:|---:|
| Generation POSTs, no retries | 18 | 18 |
| Responses passing strict validation | 18 | 18 |
| Request failures | 0 | 0 |
| Median generation seconds | 5.195 | 3.976 |
| Maximum generation seconds | 8.641 | 6.514 |
| Generation phase wall seconds, concurrency 5 | 20.746 | 17.395 |
| Prompt tokens over all 18 calls | 6,488 | 6,488 |
| Thought tokens | 3,456 | 3,575 |
| Total tokens | 10,890 | 10,975 |
| Three-pass Russian candidates | 3/6 | 2/6 |

Every result reported actual `modelVersion=gemini-3.8-flash`; the requested wire
ID was `gemini-3.8-flash-low`. Each pass was one `generateContent` POST containing
the real WAV bytes, with no tools. A native catalog/auth check and project
discovery ran once per batch. They are separate from generation counts.

Validation required the expected model version, a single STOP candidate,
`audio_heard=true`, JSON field/type validation and absence of output tool calls.
Native prompts derive the existing English blind, Russian blind and Russian
source-correction contracts, replacing file opening with an attached audio.
The source correction received only the original attached annotation, not another
model pass. All three prompts were run on these controls for diagnosis; the
ordinary production pipeline skips follow-up calls when the first pass is rejected.

The first detached launch found no module in the frozen CLI runtime and exited
before reservation or generation. The two experimental modules were then made
available in the actual account code mount. No inference was repeated.

## Precision checks

| Audio control | agy-03 | agy-16 | Listening evidence |
|---|---|---|---|
| «я умею» | All three: `ru`, «Я умею»; candidate | Blind: `ru`, «Ё-моё»; other two: «Я умею»; rejected disagreement | User confirmed Russian «я умею» |
| Nintendo Switch fragment | All three `pt`; rejected first | All three `pt`; rejected first | User confirmed foreign speech |
| «Я не знаю, ну, має бути…» | All three `uk`; rejected first | All three `uk`; rejected first | User confirmed Ukrainian speech |
| Ambiguous «плитка на ванне» | Blind `uk`, uncertainty; rejected first | Blind «плиткование», others «плитка на ванне»; rejected disagreement | User said unclear; no gold label |
| «Хорошо» | Three-way agreement, candidate | Three-way agreement, candidate | Earlier model agreement; no human-gold claim |
| «Ну я согласен, что если» | Three-way agreement, candidate | Three-way agreement, candidate | Earlier model agreement; no human-gold claim |

Neither of the two confirmed foreign audios became a Russian candidate on either
slot (four repeated control trials, two unique audios). The bad Russian guess
«Ё-моё» and the ambiguous fragment were also rejected by disagreement/uncertainty.
Only one accepted Russian audio here has an explicit listening-confirmed text;
the other accepted examples remain model-agreement candidates. This tiny,
selected set cannot establish aggregate production precision or recall.

There are nine cached CLI pass results whose prompt contract hashes match the
current contracts. Their historical median worker time was 16.01 seconds, versus
3.697 seconds for the matching native passes on 03 and 4.143 on 16. The CLI results
were collected at different times, under different load and system contexts;
this is a useful diagnostic comparison, not a concurrent speed benchmark.

## Quota snapshots

| Slot | Before, 12:37 UTC: weekly / five-hour remaining | After, 12:51 UTC: weekly / five-hour remaining |
|---|---|---|
| agy-03 | 84.28% / 92.01% | 84.10% / 91.27% |
| agy-16 | 61.87% / 79.62% | 61.59% / 99.72% |

The snapshot interval includes normal CLI work before and after the reserved
batch. Slot 16 shares a quota identity with another active slot, and its five-hour
window reset during this interval. These changes do not isolate the native batch
quota cost and do not prove a token-to-subscription-quota formula. The TUI's
selected-model header is also separate from the actual model used by these requests.

## Artifacts and operational checks

Private annotation S3 prefix:
`evaluation/native-api-canary-v1/20261014-pair-01/`.
It contains the manifest, each account's `results.json`, and raw responses under
`<account>/raw/<audio_sha256>/<prompt_variant>.json`. No credentials are stored
there. Production jobs, original annotations and source audio are untouched.

The reservation pauses fresh CLI claims until existing jobs drain, then restores
the prior queue cooldown. A 15-minute expiry is the fallback if the process dies;
rollback preserves newer quota blocks. At 12:54 UTC both slots were enabled with
normal concurrency 10 and had 9/10 active CLI jobs. The global queue had 12,108
pending jobs, 1,491 saved results per five minutes and zero expired leases.

Validation: 20 native/contract/agreement unit checks locally, 14 existing worker
checks in the deployed container, and the real PostgreSQL reservation/rollback
check in a rolled-back private schema. The CLI contract hash stayed unchanged.
