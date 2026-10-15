import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lalm.annotation.organization_calls import current_dialogues, segment_sample, stage_pilot


class OrganizationCallsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dialog = {'id': 'dialog-1', 'organization_id': 'org-1',
                       'current_relation_id': 'current', 'conversation_id': 'conversation-1',
                       'text_source': 'whole conversation, never a segment label'}
        self.current = {'id': 'current', 'dialog_id': 'dialog-1',
                        'organization_id': 'org-1', 'audio_s3_key': 'calls/current.mp3'}

    def write(self, name, rows):
        with (self.root / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def load(self, relations):
        self.write('dialogs-native.csv', [self.dialog])
        self.write('relations-native.csv', relations)
        return current_dialogues(self.root)

    def test_current_relation_is_selected_and_native_text_retained_separately(self):
        historic = dict(self.current, id='historic', audio_s3_key='calls/old.mp3')
        record = self.load([historic, self.current])['dialog-1']
        self.assertEqual(record['relation']['audio_s3_key'], 'calls/current.mp3')
        self.assertEqual(record['dialog']['text_source'], self.dialog['text_source'])
        sample = segment_sample(record, {'bucket': 'private', 'key': 'staging/meteor/index.json',
                                         'sha256': 'a' * 64},
                                {'sha256': 'b' * 64}, 1, 18.5, 38.5, 'c' * 64)
        self.assertEqual(sample['conversation_id'], 'conversation-1')
        self.assertEqual(sample['source_provenance']['channel'], 1)
        self.assertEqual(sample['source_provenance']['start_seconds'], 18.5)
        self.assertEqual(sample['source_provenance']['channel_role'], 'unverified')
        self.assertIsNone(sample['source_provenance']['original_label'])
        self.assertNotIn(self.dialog['text_source'], json.dumps(sample))
        self.assertTrue(sample['review_only'])
        self.assertFalse(sample['source_provenance']['canonical_pcm_overlap_checked'])

    def test_wrong_organization_or_dialogue_join_is_rejected(self):
        for changed in [dict(self.current, organization_id='other'),
                        dict(self.current, dialog_id='other')]:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'join_mismatch'):
                self.load([changed])

    def test_missing_or_ambiguous_current_relation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing_current_relation'):
            self.load([dict(self.current, id='historic')])
        with self.assertRaisesRegex(ValueError, 'duplicate_native_relation'):
            self.load([self.current, self.current])

    def test_missing_conversation_identity_is_rejected(self):
        self.dialog['conversation_id'] = ''
        record = self.load([self.current])['dialog-1']
        with self.assertRaisesRegex(ValueError, 'missing_native_conversation_id'):
            segment_sample(record, {'bucket': 'private', 'key': 'index', 'sha256': 'a' * 64},
                           {'sha256': 'b' * 64}, 0, 0, 20, 'c' * 64)

    def test_late_input_failure_does_not_publish_an_incoming_manifest(self):
        self.load([self.current])
        raw = b'encoded source audio'
        (self.root / 'call.mp3').write_bytes(raw)
        source = {'dialog_id': self.dialog['id'], 'relation_id': self.current['id'],
                  'conversation_id': self.dialog['conversation_id'],
                  'source_audio_key': self.current['audio_s3_key'], 'organization': 'Meteor',
                  'local_path': '/original/call.mp3', 'sha256': hashlib.sha256(raw).hexdigest()}
        selection = self.root / 'selection.json'
        selection.write_text(json.dumps([source, dict(source, sha256='0' * 64)]))
        client, connection = MagicMock(), MagicMock()
        client.get_object.return_value = {'Body': io.BytesIO(json.dumps(
            {'conversation_ids': [], 'audio_sha256': []}).encode())}
        modules = {
            'lalm.annotation.prepare': SimpleNamespace(segments=lambda _: [(0, 0, 1, b'fake WAV')]),
            'lalm.annotation.queue': SimpleNamespace(protected=lambda *args: False),
            'lalm.annotation.store': SimpleNamespace(connect=connection,
                dataset_bucket=lambda: 'datasets', s3=lambda: client)}
        with patch.dict('sys.modules', modules), self.assertRaisesRegex(ValueError, 'changed_pilot_audio'):
            stage_pilot(self.root, selection, self.root)
        self.assertFalse(any(call.kwargs['Key'].startswith('incoming/')
                             for call in client.put_object.call_args_list))
        connection.return_value.__enter__.return_value.execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
