# GigaAM + one Gemini pass: phone1 pilot

## Current deployment: SOVA CTC, 2026-10-16 19:44 UTC

At the user's request, new eligible SOVA work now uses one blind Gemini pass
and a separate, pinned GigaAM-v3-CTC reference. This produces review candidates,
not human-verified labels or a training release. The historical phone1/OpenSTT
analysis below remains diagnostic only; OpenSTT is excluded from new work.

The stock source preparer joins the existing 81,607-row SOVA diagnostic Parquet
by exact source Parquet key and row, verifies the original encoded-audio SHA256,
and attaches the CTC reference separately from the original source annotation.
The reference file's SHA256 is checked before use. Only a mono window covering
the complete source recording receives `annotation_policy=gigaam-single-pass-v1`.
Short windows of a longer recording and separate channels never inherit a
whole-file label under this rule. Other samples retain the existing triad flow.

The existing Airflow reviewer emits `candidate_gigaam_agreement` only when the
first answer is Russian, nonempty, has no reported uncertainty, and its normalized
words exactly match the pinned CTC text, without conflicting prior model output.
Disagreement/foreign/uncertain answers are rejected without coverage or correction
requests. The blind prompt and inference contract are unchanged. All reports
retain `human_review_required=true` and `automatic_publication=false`.

The selector prioritizes SOVA before resuming the retained Golos cursor. The
preparer was restarted gracefully and selected SOVA train shard 00009; at the
19:44 check it was waiting behind 9,813 existing queued/running/retry requests.
The 6,000 pending-work threshold remains unchanged. Selection is deployed;
first-manifest and completed one-pass output checks remain to be observed.

Validation: 23 contract/preparation tests passed, including whole-recording vs
partial/multichannel coverage, source identity, uncertainty, language, negation
and prior conflict. Real PostgreSQL fixture checks verify source selection and
that both one-pass matches and disagreements create no follow-up jobs; fixtures
roll back and submit no inference.

The saved-output audits below still apply: model agreement is not a measured
precision guarantee. Keep these candidates separate from three-Gemini agreement
and audit them by listening before any training publication.

At 19:55:53 UTC Airflow imported the first new SOVA manifest as 128 distinct
pending, one-pass `verbatim` jobs. The complete manifest audit checked its hash,
all 128 exact CTC source joins and two stored audio hashes. A prior read-only
reproduction of the first 32 source recordings yielded one full mono window per
recording, with all original hashes matching the CTC layer. The imported jobs
had no follow-ups at the 19:57 check; inference and the subsequent scheduled
review still need observation. Receipts:

- `speech-datasets/diagnostics/gigaam-single-pass-20261016/source32-b3eccf308c60306dc231c08466d37b0255d24067ae29182a4d4abe5bc7a8f1ed.json`
- `speech-datasets/diagnostics/gigaam-single-pass-20261016/first-manifest-c508b337e6942249c7fd676e2099b83d79bab8c1547ce3e80f07e3c866642680.json`
- `speech-datasets/diagnostics/gigaam-single-pass-20261016/import128-80ba0bd59aa8b310b65ef5bb57a1306799fb1f318c5ffd122e566d0ed8026aa4.json`

The shared progress notebook now recognizes the one-pass policy separately,
shows its GigaAM text and candidate hours, and keeps it outside the three-pass
yield denominator. Select `decision_filter='candidate_gigaam_agreement'` to
inspect these matches once saved outputs enter an inspection snapshot. Existing
snapshots without the new pass columns remain supported. PostgreSQL view and
small notebook fixtures checked matches, disagreement, foreign language,
incorrect source identity and legacy triad snapshots without executing a user
kernel. Six notebook cells changed after a Jupyter checkpoint; the other cells
and metadata were preserved. Deployed source digest:
`fd3e36f42c14756d1ea5ddf929c0b21224b49a6c9459866280d66ce85e6886d0`.
The appended pass-view fields become visible in the next scheduled snapshot:
`speech-datasets/diagnostics/gigaam-single-pass-20261016/inspection-view-634191569bf54decf9ab993eb7106e61e317b532dcf2ba4a4d64fef1e1b6108b.json`.

Checked 2026-10-15 (MSK) against the immutable inspection snapshot from
2026-10-14 22:02:53 UTC. This is an analysis of saved outputs; it made no new
Gemini requests and did not publish training labels.

The current `openstt-phone1` source annotation in the labeling queue is the
original OpenSTT archive text. A separate GigaAM-RNNT prediction set exists at
`/mnt/local/drive1/ifao-data/phone1-remaining-rnnt-v2/shard-*/predictions/input/shard-*/gigaam-sip/preds.jsonl`
on `dev-1`. Its seven shards contain 200,016 predictions. The recorded
checkpoint SHA256 is
`607e71d4d9fa87fadbdd779357fc3d1d064e1c273e7080cafd325ce474ba591e`.
The set excludes a separate 6,000-example partition; it must not be joined to
that partition or to benchmark audio.

Predictions were joined to `openstt-phone1` by original archive member path.
Only a zero-start segment whose processed duration differed from the GigaAM
input duration by less than 0.35 s entered the comparison. This yielded
192,289 distinct audio SHA256 values. The first Gemini pass was the blind
`verbatim` pass, so it did not receive GigaAM text. For the one-pass rule, its
language had to be `ru`, its uncertainty list empty, and its normalized words
identical to GigaAM's.

| Measure | Count |
| --- | ---: |
| Exactly linked audio | 192,289 |
| First Gemini pass matched GigaAM | 49,152 (25.6%) |
| Matches with a second, independent Gemini answer | 47,613 |
| Those where the second answer disagreed | 9,141 (19.2%) |
| Matches with all three Gemini texts present | 46,430 |
| Those where all three Gemini texts agreed | 31,531 (67.9%) |

These disagreement rates are **not error rates**: neither the second pass nor
three-pass agreement is human gold. However, agreement between one Gemini pass
and GigaAM is not strong enough evidence to mark a label clean under the
project's precision-first rule. None of these examples is marked
`human_verified`. Keep this rule as a review-only candidate selector until a
stratified listening audit measures actual precision, including short phrases
and cases where the independent second pass disagrees.

The dataset catalog mentions GigaAM-RNNT for `phone1`, but the copied archive
and CSV contain the original OpenSTT labels. `phone2` has original OpenSTT ASR
labels with the model unconfirmed; the queue has no verified GigaAM layer for
it. The small `bitmanager-sessions` catalog also mentions client GigaAM-RNNT,
but its source index is not proof that every stored audio has an attached label.

## Additional sources located on 2026-10-15

The shared inventory spans three independent stores. A live read from the
second store (`https://100.119.47.17:19010`) confirmed a substantial **CTC**
layer, distinct from the RNNT predictions evaluated above:

- `speech-corpora/staging/gigaam-original-all/20261013-v1/manifests/verified-snapshot-20261013T132951Z.json`:
  3,475,789 canonical rows in 2,434 Parquet files, plus 81,607 SOVA rows;
  3,557,396 rows total. Its recorded model is `v3_ctc`, weights SHA256
  `a20a9b5b8ed929c4a60141753cd3dd1284138879a46fb4a1cef6430ec019a02f`.
  The manifest reports complete SHA256 readback, zero missing IDs and zero
  duplicate source IDs. These are the producer's checks, not a new full audit.
- A freshly read canonical Parquet has `audio_sha256`, `source_endpoint`,
  `source_bucket`, `source_key`, `duration`, `gigaam_transcription`,
  `gigaam_segments`, `gigaam_model` and `gigaam_weights_sha256`. Its rows are
  references and predictions; they do not contain new waveform bytes.
- `speech-corpora/staging/sova-rudevices-gigaam-v3-ctc/20261013-v1/manifest.json`
  reports all 81,607 train examples, retaining source text and adding separate
  `gigaam_transcription`, model provenance, source row and audio SHA256.
  A CTC-versus-Gemini comparison is still required; the RNNT disagreement
  percentages above must not be presented as CTC measurements.

Other existing catalog evidence identifies these candidates:

| Source | Available GigaAM evidence | Current restriction |
| --- | --- | --- |
| `lab260/tuchniyzhab_balalaika` | 1,347 paired audio/JSON records with `giga_rnnt.txt`, CTC and other ASR fields | Cross-corpus and benchmark overlap unverified |
| `crm-history175` | 187,758 turns / 174.83 h, GigaAM-SIP RNNT | Private archive; existing holdout audit and admission restrictions |
| `exp-local-calls159` | 4,526 turns / 5.394 h, GigaAM-SIP RNNT | Preserve 75 train / 25 dev / 26 test / 33 control conversation splits |
| `YO-CPT-ru` pilot | 3,558 examples / about 14.1 h; publisher describes GigaAM/Whisper/Vosk ensemble | Ensemble text is not automatically an independent GigaAM hypothesis; source overlap unresolved |

This discovery does not enqueue restricted sources or promote single-pass
matches to clean labels. The current high-precision flow remains in place
pending an exact audio join and a separate quality comparison of the CTC layer.

### Saved SOVA CTC comparison, 2026-10-15 01:45:43 UTC

All 81,607 source-linked CTC rows were extracted using Parquet column reads;
no audio was retranscribed. The separate diagnostic artifact is
`speech-datasets/diagnostics/gigaam-ctc-sova-20261015/cc693e4862f452a2d7a56f586b28acec3e38e578a5ad89302a958d84d2f5aedd.parquet`.
Its filename is its SHA256. Original SOVA labels were retained unchanged.

`scripts/analyze_gigaam_ctc_saved.py` compares saved
`precision-ru-triad-v1%` passes against the exact source Parquet key and row.
The query is read-only and sends no model requests.

| Measure | Count |
| --- | ---: |
| Source-linked examples with first Gemini text | 4,121 |
| Those with identical original audio SHA256 in both layers | 4,121 |
| Russian, uncertainty-free first Gemini text matches CTC | 2,161 |
| Those with an independent second text | 2,115 |
| Independent second text disagrees | 257 (12.2%) |
| Those with all three texts | 2,096 |
| All three agree | 1,682 (80.2%) |

The query saw 11,820 saved pass rows, including 115 from a separate server
pilot that could not be joined to this source locator scheme. Prepared-audio
SHA256 values differ from the CTC input hashes after preprocessing. A follow-up
comparison against the separate `jobs.sample.source_audio_sha256` field verified
the original input bytes for all 4,121 linked examples, with the same agreement
counts. This verifies source identity, not full-segment coverage after
preparation; segmentation boundaries still need validation for an admission
rule. The evaluated examples
are also a selected subset, not a random SOVA sample. These are agreement
statistics, not human-gold accuracy. The result does not justify automatically
publishing half the queue under a one-pass rule.
