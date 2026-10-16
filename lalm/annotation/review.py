"""Compare blind passes and request an explicitly identified third-pass audit."""
import hashlib
import json
from difflib import SequenceMatcher
from .store import connect, s3, bucket
from .queue import enqueue
from .contract import source_text, TRIAD_VERSION, triad_decision, gigaam_single_pass_decision
from .gigaam_reference import POLICY as GIGAAM_POLICY
from .inspection import normalized_words


def compare(a, b):
    left, right = normalized_words(a), normalized_words(b)
    diffs = [{'kind': tag, 'a_span': [i, j], 'b_span': [k, l],
              'a_text': ' '.join(left[i:j]), 'b_text': ' '.join(right[k:l])}
             for tag, i, j, k, l in SequenceMatcher(None, left, right, autojunk=False).get_opcodes()
             if tag != 'equal']
    return {'normalized_agreement': not diffs, 'a_words': len(left), 'b_words': len(right),
            'differences': diffs, 'human_review_required': True}


def _enqueue_followups(samples):
    # Reuse the existing guarded/idempotent enqueue path, committing small
    # batches instead of opening a connection and committing for every job.
    for offset in range(0, len(samples), 128):
        with connect() as conn:
            for sample in samples[offset:offset + 128]:
                enqueue(conn, sample)


def reconcile(max_pilots=200):
    # Native calls have one pilot per conversation; 25 per five-minute cycle
    # falls behind their input rate. Keep the batch bounded and caller-tunable.
    if type(max_pilots) is not int or not 1 <= max_pilots <= 1000:
        raise ValueError('invalid_review_batch_limit')
    with connect() as c:
        started = c.execute('SELECT clock_timestamp() AS stamp').fetchone()['stamp']
        changed = c.execute("SELECT j.sample->>'pilot_id' AS pilot_id FROM jobs j "
            "LEFT JOIN annotation_review_cursors r ON r.pilot_id=j.sample->>'pilot_id' "
            "WHERE j.sample->>'pilot_id' LIKE %s AND j.updated_at>COALESCE(r.checked_at,'epoch'::timestamptz) "
            "GROUP BY j.sample->>'pilot_id' ORDER BY min(j.updated_at),j.sample->>'pilot_id' LIMIT %s",
            (TRIAD_VERSION + '-%', max_pilots)).fetchall()
        pilots = [r['pilot_id'] for r in changed]
        if not pilots:
            return {'reports': [], 'third_passes_requested': 0}
        jobs = c.execute("SELECT id,sample,state,result,error_code FROM jobs "
                         "WHERE sample->>'pilot_id'=ANY(%s) ORDER BY id", (pilots,)).fetchall()
    with connect() as c:
        # A pilot's own answers are excluded from previous_texts below. Avoid
        # fetching them in the common single-pilot run, but preserve all history
        # across changed pilots and preserve legacy rows with a NULL pilot ID.
        excluded_pilot = pilots[0] if len(pilots) == 1 else None
        historical = c.execute("SELECT sample->>'sample_id' AS sample_id, sample->>'pilot_id' AS pilot_id, "
            "result->>'verbatim_text' AS text FROM jobs WHERE result IS NOT NULL "
            "AND (%s::text IS NULL OR sample->>'pilot_id' IS DISTINCT FROM %s::text)",
            (excluded_pilot, excluded_pilot)).fetchall()
    previous_by_sample = {}
    for old in historical:
        previous_by_sample.setdefault(old['sample_id'], []).append(old)
    groups = {}
    for j in jobs:
        s = j['sample']
        groups.setdefault((s['pilot_id'], s['sample_id']), {})[s.get('prompt_variant', 'verbatim')] = j
    reports = {}
    followups = []
    for (pilot, sample_id), passes in groups.items():
        source = next(iter(passes.values()))['sample']
        row = {'sample_id': sample_id, 'audio_key': source['audio_key'],
               'audio_bucket': source.get('audio_bucket', bucket()),
               'source_provenance': source.get('source_provenance'), 'passes': {k: {'job_id': v['id'], 'state': v['state'],
               'error_code': v['error_code'], 'result': v['result']} for k, v in passes.items()}}
        if not pilot.startswith(TRIAD_VERSION + '-'):
            continue  # Historical prompts are immutable evidence; never schedule old passes.
        results = {k: v.get('result') for k, v in passes.items()}
        previous = [r['text'] for r in previous_by_sample.get(sample_id, []) if r['pilot_id'] != pilot]
        if source.get('annotation_policy') == GIGAAM_POLICY:
            row['decision'] = gigaam_single_pass_decision(source, results, previous)
            row['annotation_policy'] = GIGAAM_POLICY
        else:
            row['decision'] = triad_decision(results, previous)
        row['human_review_required'] = True
        row['automatic_publication'] = False
        if row['decision'] == 'waiting_remaining':
            if source_text(source) is None:
                row['decision'] = 'missing_source_annotation'
            else:
                for variant in ('coverage', 'source_correction'):
                    if variant not in passes:
                        followups.append(dict(source, prompt_variant=variant, review_only=True))
        if any(v['state'] in ('NEEDS_REVIEW', 'PERMANENT_FAILED') and not v.get('result')
               for v in passes.values()):
            row['decision'] = 'rejected_failed_pass'
        if all(results.get(v) for v in ('verbatim', 'coverage', 'source_correction')):
            # The report only needs equality. Keep detailed alignment in compare()
            # for callers that actually consume the word-level differences.
            words = normalized_words(results['verbatim']['verbatim_text'])
            row['normalized_agreement'] = all(
                words == normalized_words(results[v]['verbatim_text'])
                for v in ('coverage', 'source_correction'))
        reports.setdefault(pilot, []).append(row)
    # Do not publish reports or advance cursors if an enqueue guard fails.
    _enqueue_followups(followups)
    summaries = []
    for pilot, rows in reports.items():
        rows.sort(key=lambda r: r['sample_id'])
        summary = {'pilot_id': pilot, 'samples': len(rows),
                   'paired': sum('normalized_agreement' in r for r in rows),
                   'disagreements': sum(r.get('normalized_agreement') is False for r in rows),
                   'candidates_3of3': sum(r.get('decision') == 'candidate_3of3' for r in rows),
                   'candidates_gigaam_agreement': sum(r.get('decision') == 'candidate_gigaam_agreement' for r in rows)}
        body = json.dumps({'summary': summary, 'rows': rows}, ensure_ascii=False, sort_keys=True).encode()
        sha = hashlib.sha256(body).hexdigest()
        key = f'reviews/{pilot}/{sha}.json'
        s3().put_object(Bucket=bucket(), Key=key, Body=body)
        with connect() as c:
            c.execute('INSERT INTO annotation_review_cursors(pilot_id,checked_at) VALUES (%s,%s) '
                      'ON CONFLICT(pilot_id) DO UPDATE SET checked_at=excluded.checked_at', (pilot, started))
        summaries.append({**summary, 'report_key': key})
    return {'reports': summaries, 'third_passes_requested': len(followups)}
