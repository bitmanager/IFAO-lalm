"""Run notebook decision cells on small fixtures, without S3 or a user kernel."""
import json
from pathlib import Path

import pandas as pd

from lalm.annotation.contract import TRIAD_VERSION, triad_decision, gigaam_single_pass_decision
from lalm.annotation.gigaam_reference import POLICY as GIGAAM_POLICY, REFERENCE_KEY, REFERENCE_SHA


notebook = json.loads(Path('notebooks/speech_data_progress.ipynb').read_text())
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), f'cell-{index}', 'exec')
source = ''.join(notebook['cells'][1]['source'])
decision_code = source[source.index('records = []'):]


def row(sample, variant='verbatim', text='да', language='ru', policy=GIGAAM_POLICY):
    reference = {'reference_key': REFERENCE_KEY, 'reference_sha256': REFERENCE_SHA,
                 'full_source_coverage': True, 'prepared_audio_sha256': 'a' * 64,
                 'source_audio_sha256': 'b' * 64, 'model': 'v3_ctc',
                 'weights_sha256': 'a20a9b5b8ed929c4a60141753cd3dd1284138879a46fb4a1cef6430ec019a02f',
                 'text': 'да'}
    return {'batch_id': TRIAD_VERSION + '-fixture', 'sample_id': sample,
            'prompt_variant': variant, 'gemini_text': text, 'language': language,
            'uncertain_spans': '[]', 'processing_state': 'NEEDS_REVIEW',
            'original_annotation_status': 'attached', 'audio_sha256': 'a' * 64,
            'source_audio_sha256': 'b' * 64, 'annotation_policy': policy,
            'source_provenance': json.dumps({'gigaam_reference': reference}) if policy else None}


rows = [row('single-match'), row('single-mismatch', text='нет'),
        row('single-foreign', language='uk'), row('single-wrong-hash')]
rows[-1]['source_audio_sha256'] = 'c' * 64
rows += [row('triad', variant=v, policy=None) for v in ('verbatim', 'coverage', 'source_correction')]
current = pd.DataFrame(rows)
examples = current[['batch_id', 'sample_id']].drop_duplicates()
namespace = dict(globals(), current=current, examples=examples, old_texts={})
exec(compile(decision_code, 'notebook-decision-fixture', 'exec'), namespace)
actual = namespace['examples'].set_index('sample_id')
assert actual.decision.to_dict() == {
    'single-match': 'candidate_gigaam_agreement',
    'single-mismatch': 'rejected_gigaam_disagreement',
    'single-foreign': 'rejected_gigaam_quality',
    'single-wrong-hash': 'rejected_gigaam_reference', 'triad': 'candidate_3of3'}
assert actual.loc['single-match', 'gigaam_text'] == 'да'
assert actual.loc['triad', 'annotation_policy'] == TRIAD_VERSION

# Older immutable snapshots lack the appended columns and still load normally.
legacy = current[current.sample_id.eq('triad')].drop(columns=['annotation_policy', 'source_audio_sha256'])
namespace.update(current=legacy, examples=legacy[['batch_id', 'sample_id']].drop_duplicates())
exec(compile(decision_code, 'notebook-legacy-fixture', 'exec'), namespace)
assert namespace['examples'].decision.tolist() == ['candidate_3of3']
print('Notebook one-pass, triad and older-snapshot fixtures passed; no external I/O')
