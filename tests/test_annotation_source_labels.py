import json
import sqlite3
import unittest

from lalm.annotation.source_labels import build_index, lookup
from lalm.annotation.contract import source_text


class SourceLabelsTests(unittest.TestCase):
    def test_hash_lookup_preserves_provenance_and_refuses_conflicting_text(self):
        rows = [
            {'source_audio_sha256': 'a'*64, 'original_metadata': {'text': 'слово'},
             'audio_filepath': '/untrusted/path', 'training_eligible': False},
            {'source_audio_sha256': 'b'*64, 'original_metadata': {'text': 'пять'}},
            {'source_audio_sha256': 'b'*64, 'original_manifest_record': {'text': 'семь'}},
        ]
        with sqlite3.connect(':memory:') as conn:
            build_index(conn, [json.dumps(r).encode() for r in rows])
            reference = {'bucket': 'test', 'key': 'index.jsonl', 'etag': 'version'}
            label = lookup(conn, 'a'*64, reference)
            self.assertEqual(label['records'][0], rows[0])
            self.assertEqual(label['text'], 'слово')
            conflict = lookup(conn, 'b'*64, reference)
            self.assertTrue(conflict['text_conflict'])
            self.assertEqual(len(conflict['records']), 2)
            self.assertIsNone(source_text({'source_provenance': {'original_label': conflict}}))
            self.assertIsNone(lookup(conn, 'c'*64, reference))


if __name__ == '__main__':
    unittest.main()
