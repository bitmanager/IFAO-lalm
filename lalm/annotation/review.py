"""Compare blind passes and request an explicitly identified third-pass audit."""
import hashlib
import json
import re
from difflib import SequenceMatcher
from .store import connect, s3, bucket
from .queue import enqueue
from .contract import source_text


def compare(a, b):
    def words(s):
        return re.findall(r"[\w]+", s.lower().replace('ё', 'е'))
    left, right = words(a), words(b)
    diffs = [{'kind': tag, 'a_span': [i, j], 'b_span': [k, l],
              'a_text': ' '.join(left[i:j]), 'b_text': ' '.join(right[k:l])}
             for tag, i, j, k, l in SequenceMatcher(None, left, right, autojunk=False).get_opcodes()
             if tag != 'equal']
    return {'normalized_agreement': not diffs, 'a_words': len(left), 'b_words': len(right),
            'differences': diffs, 'human_review_required': True}


def reconcile():
    with connect() as c:
        started = c.execute('SELECT clock_timestamp() AS stamp').fetchone()['stamp']
        changed = c.execute("SELECT j.sample->>'pilot_id' AS pilot_id FROM jobs j "
            "LEFT JOIN annotation_review_cursors r ON r.pilot_id=j.sample->>'pilot_id' "
            "WHERE j.sample ? 'pilot_id' AND j.updated_at>COALESCE(r.checked_at,'epoch'::timestamptz) "
            "GROUP BY j.sample->>'pilot_id' ORDER BY min(j.updated_at) LIMIT 25").fetchall()
        pilots = [r['pilot_id'] for r in changed]
        jobs = c.execute("SELECT id,sample,state,result,error_code FROM jobs "
                         "WHERE sample->>'pilot_id'=ANY(%s) ORDER BY id", (pilots,)).fetchall()
    groups = {}
    for j in jobs:
        s = j['sample']
        groups.setdefault((s['pilot_id'], s['sample_id']), {})[s.get('prompt_variant', 'verbatim')] = j
    reports = {}
    third_requested = 0
    for (pilot, sample_id), passes in groups.items():
        source = next(iter(passes.values()))['sample']
        row = {'sample_id': sample_id, 'audio_key': source['audio_key'],
               'audio_bucket': source.get('audio_bucket', bucket()),
               'source_provenance': source.get('source_provenance'), 'passes': {k: {'job_id': v['id'], 'state': v['state'],
               'error_code': v['error_code'], 'result': v['result']} for k, v in passes.items()}}
        if all(passes.get(k, {}).get('result') for k in ('verbatim', 'coverage')):
            row.update(compare(passes['verbatim']['result']['verbatim_text'],
                               passes['coverage']['result']['verbatim_text']))
            seed = passes['verbatim']['sample']
            if not row['normalized_agreement'] and seed.get('adaptive_third_pass'):
                # Retain existing third jobs; never add a fourth pass on upgrade.
                variant = ('literal_audit' if 'literal_audit' in passes else
                           'source_correction' if source_text(seed) is not None else 'literal_audit')
                row['third_pass_variant'] = variant
                row['third_pass_independent'] = variant != 'source_correction'
                if variant not in passes:
                    sample = dict(seed, prompt_variant=variant)
                    with connect() as c:
                        enqueue(c, sample)
                    third_requested += 1
                    row['third_pass_pending'] = True
                elif passes[variant].get('result'):
                    third = passes[variant]['result']['verbatim_text']
                    row['third_vs_verbatim'] = compare(passes['verbatim']['result']['verbatim_text'], third)
                    row['third_vs_coverage'] = compare(passes['coverage']['result']['verbatim_text'], third)
                else:
                    row['third_pass_pending'] = passes[variant]['state'] in ('PENDING', 'RUNNING', 'RETRYABLE_FAILED')
        else:
            row['incomplete'] = True
        reports.setdefault(pilot, []).append(row)
    summaries = []
    for pilot, rows in reports.items():
        rows.sort(key=lambda r: r['sample_id'])
        summary = {'pilot_id': pilot, 'samples': len(rows),
                   'paired': sum('normalized_agreement' in r for r in rows),
                   'disagreements': sum(r.get('normalized_agreement') is False for r in rows)}
        body = json.dumps({'summary': summary, 'rows': rows}, ensure_ascii=False, sort_keys=True).encode()
        sha = hashlib.sha256(body).hexdigest()
        key = f'reviews/{pilot}/{sha}.json'
        s3().put_object(Bucket=bucket(), Key=key, Body=body)
        with connect() as c:
            c.execute('INSERT INTO annotation_review_cursors(pilot_id,checked_at) VALUES (%s,%s) '
                      'ON CONFLICT(pilot_id) DO UPDATE SET checked_at=excluded.checked_at', (pilot, started))
        summaries.append({**summary, 'report_key': key})
    return {'reports': summaries, 'third_passes_requested': third_requested}
