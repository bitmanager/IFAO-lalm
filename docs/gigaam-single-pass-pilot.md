# GigaAM + one Gemini pass: phone1 pilot

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
