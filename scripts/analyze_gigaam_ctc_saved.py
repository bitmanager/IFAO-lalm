"""Compare saved Gemini passes with a source-linked GigaAM CTC layer.

Read-only analysis: source locator agreement is not waveform verification or
permission to publish labels. This makes no model requests.
"""
import argparse
from collections import Counter
import hashlib
import io
import json

import pyarrow.parquet as pq

from lalm.annotation.inspection import normalized_words
from lalm.annotation.store import connect, dataset_bucket, s3


def analyze(key, expected_sha):
    body = s3().get_object(Bucket=dataset_bucket(), Key=key)['Body'].read()
    if hashlib.sha256(body).hexdigest() != expected_sha:
        raise ValueError('CTC layer checksum mismatch')
    table = pq.read_table(io.BytesIO(body))
    labels = {}
    for row in table.to_pylist():
        locator = (row['source_s3_key'], str(row['source_row']))
        if locator in labels:
            raise ValueError('Duplicate CTC source locator')
        labels[locator] = row
    counts, groups = Counter(), {}
    with connect() as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        conn.execute("SET LOCAL statement_timeout='100s'")
        captured = conn.execute('SELECT transaction_timestamp() AS t').fetchone()['t'].isoformat()
        with conn.cursor(name='gigaam_ctc_saved_comparison') as cursor:
            cursor.execute("""SELECT batch_id,sample_id,audio_sha256,prompt_variant,
                gemini_text,language,uncertain_spans,source_provenance,
                (SELECT sample->>'source_audio_sha256' FROM jobs j
                  WHERE j.id=review_passes.job_id) AS source_audio_sha256
                FROM review_passes WHERE corpus='sova-rudevices'
                AND batch_id LIKE 'precision-ru-triad-v1%'
                AND gemini_text IS NOT NULL""")
            while rows := cursor.fetchmany(5000):
                for row in rows:
                    counts['saved_pass_rows'] += 1
                    source = row['source_provenance']
                    ctc = labels.get((source.get('key'), str(source.get('member'))))
                    if ctc is None:
                        counts['unlinked_pass_rows'] += 1
                        continue
                    if source.get('start_seconds', 0) != 0:
                        counts['nonzero_start_pass_rows'] += 1
                        continue
                    identity = (row['batch_id'], row['sample_id'])
                    group = groups.setdefault(identity, {
                        'ctc': normalized_words(ctc['gigaam_transcription']),
                        'exact_hash': row['audio_sha256'] == ctc['audio_sha256'],
                        'source_hash_matches': row['source_audio_sha256'] == ctc['audio_sha256'],
                        'passes': {}})
                    group['passes'][row['prompt_variant']] = (
                        normalized_words(row['gemini_text']), row['language'],
                        not row['uncertain_spans'])
    comparisons = {}
    for scope in ('source_locator_only', 'same_source_audio_sha256', 'same_encoded_audio_sha256'):
        out = Counter()
        for group in groups.values():
            if scope == 'same_encoded_audio_sha256' and not group['exact_hash']:
                continue
            if scope == 'same_source_audio_sha256' and not group['source_hash_matches']:
                continue
            out['samples'] += 1
            passes = group['passes']
            first = passes.get('verbatim')
            if not first:
                continue
            out['with_first_text'] += 1
            if first[1] != 'ru' or not first[2] or not group['ctc'] or first[0] != group['ctc']:
                continue
            out['first_matches_ctc_ru_without_uncertainty'] += 1
            independent = passes.get('coverage')
            if independent:
                out['with_independent_second_text'] += 1
                if independent[0] != first[0]:
                    out['independent_second_disagrees'] += 1
            three = [passes.get(name) for name in ('verbatim', 'coverage', 'source_correction')]
            if all(three):
                out['with_three_texts'] += 1
                if all(value[0] == first[0] for value in three):
                    out['three_agree'] += 1
        comparisons[scope] = dict(out)
    return {'captured_at': captured, 'purpose': 'review_only_diagnostic',
            'ctc_key': key, 'ctc_sha256': expected_sha,
            'ctc_rows': table.num_rows, 'counts': dict(counts),
            'comparisons': comparisons,
            'limitations': ['No human-gold accuracy measurement',
                'Source-hash joins verify original bytes but not full-segment coverage after preparation',
                'Saved-pass selection is not a random corpus sample']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.key, args.sha256), sort_keys=True))
