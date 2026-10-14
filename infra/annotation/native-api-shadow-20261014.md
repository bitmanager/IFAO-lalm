# First parallel native audio shadow batch, 2026-10-14

While the normal CLI annotation queue continued on the other accounts,
`agy-03` handled a private, review-only batch of 100 audios through the native
Gemini endpoint. The account drained its existing CLI jobs before the batch
and restored normal CLI claims afterward. This was a bounded live trial, not a
switch of the production annotation transport or a training-data publication.

The batch used six earlier listening controls and 94 fresh, distinct,
unprotected training audios. Recent CLI language tags selected 80 `ru` and 14
other examples; these tags are sampling hints, not ground truth. Every sample
had an attached source annotation. The generator excluded all 30 audios from
the previous four-account trial and kept one fresh audio per conversation.
The manifest is private at
`s3://speech-annotations/evaluation/native-api-canary-v1/20261014-shadow-01/manifest.json`.

At most five native requests ran concurrently. The adaptive protocol made a
blind pass on all 100 audios, then two more independent passes only after a
usable Russian first result. There were 264 generation POSTs over 203.811
seconds; median POST time was 2.941 seconds. 263 responses passed the strict
native parser. One HTTP 200 response was invalid JSON and was rejected without
an automatic retry. No 401, 403, 429, server error, transport failure or
timeout stopped the batch. No extra `agy models` catalog call was needed.

| Decision | Audios |
|---|---:|
| Three-pass candidate | 58 |
| First pass rejected | 17 |
| Three-pass disagreement | 20 |
| Quality check rejected | 4 |
| First pass invalid, pending manual review | 1 |

Both user-confirmed foreign controls were rejected. The ambiguous recording
was rejected. The user-confirmed Russian «я умею» was rejected for disagreement:
its blind response was «Ё-моё» on this run. This is a recall failure on a known
Russian example, although it does not create a false accepted label. The other
94 fresh audios have no listening-confirmed gold. Thus `58/100` is a candidate
yield, not an accuracy or precision measurement. The candidates remain out of
training until a listening audit.

The raw gateway responses and per-audio report are under
`s3://speech-annotations/evaluation/native-api-canary-v1/20261014-shadow-01/agy-03/`.
The shared Jupyter notebook can show this batch by setting
`native_run = 'shadow_01'`; it presents all outputs and plays the selected
audio without marking it human-verified.

The first launch encountered a short account cooldown and exited before any
inference. The retry started when the account was ready. The reservation now
waits up to two minutes for short cooldowns; longer quota blocks and disabled
accounts still stop the batch before audio is sent. Its guarded rollback was
checked against a disposable PostgreSQL transaction.
