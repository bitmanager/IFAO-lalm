import unittest

from lalm.annotation.contextual_calls import build_records, index_key


def row(ident, start, end, channel=0, conversation='call', uncertain=False):
    return {'job_id': ident, 'sample': {
        'sample_id': ident, 'conversation_id': conversation,
        'audio_bucket': 'audio', 'audio_key': ident, 'audio_sha256': ident,
        'source_provenance': {'start_seconds': start, 'end_seconds': end,
            'channel': channel, 'group_identity_verified': True,
            'bucket': 'indices', 'key': 'source', 'original_label': 'whole call, future included'}},
        'result': {'language': 'ru', 'verbatim_text': ident,
                   'uncertain_spans': ['uncertain'] if uncertain else [],
                   'model': 'model', 'contract_sha256': 'contract'}}


class ContextualCallsTests(unittest.TestCase):
    def test_canonical_index_is_per_immutable_source_manifest(self):
        key = 'incoming/' + 'a' * 64 + '.jsonl'
        self.assertTrue(index_key(key).endswith('/manifests/' + 'a' * 64 + '.json'))
        with self.assertRaises(ValueError):
            index_key('incoming/not-an-immutable-manifest.jsonl')

    def test_history_excludes_future_overlap_other_calls_and_uncertainty(self):
        records = build_records([
            row('past', 0, 2, channel=1), row('overlap', 1, 4),
            row('current', 3, 5), row('future', 6, 7),
            row('other', 0, 1, conversation='other-call'),
            row('uncertain', 0, 1, uncertain=True)])
        current = next(r for r in records if r['job_id'] == 'current')
        self.assertEqual([h['source_job_id'] for h in current['history']], ['past'])
        self.assertEqual(current['history'][0]['speaker_id'], 'channel_1')
        self.assertEqual(current['history'][0]['role'], 'unknown')
        self.assertEqual(current['target']['verbatim_text'], 'current')
        self.assertFalse(current['metadata']['training_release'])
        self.assertFalse(current['metadata']['human_verified'])

    def test_history_length_and_chronological_order(self):
        rows = [row(str(i), i, i + 1) for i in range(8)]
        records = build_records(list(reversed(rows)), max_history=2)
        self.assertEqual(records, build_records(rows, max_history=2))
        self.assertEqual([h['text'] for h in records[-1]['history']], ['5', '6'])
        self.assertEqual(records[0]['history'], [])

    def test_foreign_and_empty_predictions_are_not_targets_or_history(self):
        foreign, empty = row('foreign', 0, 1), row('empty', 1, 2)
        foreign['result']['language'] = 'uk'
        empty['result']['verbatim_text'] = ''
        records = build_records([foreign, empty, row('current', 2, 3)])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['history'], [])

    def test_invalid_identity_times_and_duplicates_fail_closed(self):
        for start, end in ((2, 1), (-1, 2), (0, float('nan'))):
            with self.assertRaises(ValueError):
                build_records([row('bad', start, end)])
        duplicate = row('same', 0, 1)
        with self.assertRaises(ValueError):
            build_records([duplicate, duplicate])
        duplicate['sample']['source_provenance']['group_identity_verified'] = False
        with self.assertRaises(ValueError):
            build_records([duplicate])
