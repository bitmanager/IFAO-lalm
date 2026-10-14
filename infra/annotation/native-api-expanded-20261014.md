# Native audio canary expanded to four accounts, 2026-10-14

The same 30 unprotected training audios ran through `agy-03`, `agy-16`,
`agy-08`, and `agy-21` as an isolated review experiment. Six were earlier
controls; 24 were fresh, unverified phone recordings. Each account used its
own existing VPN and OAuth home. No production queue result or source label
was modified. The account reservation paused new CLI claims, waited for the
existing jobs to drain, then restored the normal CLI queue on completion.

The adaptive protocol made a blind pass first. It asked for the Russian blind
and source correction passes only after a usable Russian first pass. Each
generation was exactly one POST with the audio inline, at most five concurrent
per account, with no automatic retry. The existing prompt contract was used
without changing its CLI hash. The source correction saw the attached original
label, not the other model outputs. All results remain review-only.

| Account | POSTs | Complete | Format rejection | Median POST seconds | Generation wall seconds | Three-pass candidates |
|---|---:|---:|---:|---:|---:|---:|
| agy-03 | 74 | 73 | 1 | 3.306 | 63.827 | 13/30 |
| agy-16 | 68 | 68 | 0 | 3.562 | 71.699 | 11/30 |
| agy-08 | 72 | 72 | 0 | 2.977 | 53.845 | 11/30 |
| agy-21 | 72 | 72 | 0 | 3.609 | 76.154 | 10/30 |

There were 286 generation POSTs instead of the 360 that three passes for
every sample would require. Healthy cached access tokens were reused; none
of these four batches needed an extra `agy models` catalog request. The one
`agy-03` failure returned HTTP 200 but echoed a JSON schema with the wrong
field set. Validation rejected it and did not automatically replay it or
publish a label.

The two recordings the user identified as non-Russian were rejected in all
eight account trials. The ambiguous «плитка на ванне» was rejected in all four.
The user-confirmed «я умею» became a three-pass candidate on 03 and 16 but
was rejected for disagreement on 08 and 21. Ten of the 30 audios became
three-pass candidates on all four accounts. Their blind texts matched across
accounts except punctuation and `е`/`ё`; these ten include two controls whose
earlier evidence was model agreement only. The other eight have no human gold.
These are repeated judgments on 30 selected audios, not a precision estimate
for a population or evidence that the candidates are acoustically correct.
Keep them out of the training publication until a listening audit establishes
the error rate.

The private reports and raw gateway responses are under
`s3://speech-annotations/evaluation/native-api-canary-v1/20261014-expanded-01/`.
The 30-audio manifest and container logs remain private on `dev-1`. Credentials
are absent from the report, raw-response prefix, and Git.

The same investigation found three account-local OAuth JSON files with extra
trailing bytes after a complete object (`06`, `13`, `18`). An exact, private
backup was saved in each account's home; only the complete first object was
restored. All three subsequently passed `agy models`. The source now checks
this narrow corruption before inference and shares a one-minute auth catalog
result across threads to avoid an auth outage spawning many simultaneous
catalog calls. Incomplete or ambiguous files remain blocked before inference.
After restarting the three workers, `06` and `18` each had zero new errors in
a five-minute window and resumed saving records; `13` resumed 20 active jobs
but still had ten errors in that window and needs further observation.

Validation: seven local native API/auth tests, sixteen worker tests in the
deployed container, `git diff --check`, and the four live bounded batches.
