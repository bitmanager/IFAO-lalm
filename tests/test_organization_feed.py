import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from lalm.annotation.organization_feed import checkpoint_key, feed


class OrganizationFeedTests(unittest.TestCase):
    def test_checkpoint_binds_selection_and_native_metadata(self):
        self.assertEqual(checkpoint_key([{'a': 1}], {'b': 'hash'}),
                         checkpoint_key([{'a': 1}], {'b': 'hash'}))
        self.assertNotEqual(checkpoint_key([{'a': 1}], {'b': 'hash'}),
                            checkpoint_key([{'a': 1}], {'b': 'other'}))

    def test_restart_checks_saved_manifests_and_does_not_stage_again(self):
        import hashlib
        objects, counts = {}, {}

        def get_object(Bucket, Key):
            if Key not in objects:
                raise ClientError({'ResponseMetadata': {'HTTPStatusCode': 404},
                                   'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
            return {'Body': io.BytesIO(objects[Key])}

        client = Mock()
        client.get_object.side_effect = get_object
        client.put_object.side_effect = lambda **kw: objects.update({kw['Key']: kw['Body']})

        def stage(metadata, selection, audio):
            rows = json.loads(Path(selection).read_text())
            raw = json.dumps(rows, sort_keys=True).encode()
            key = 'incoming/' + hashlib.sha256(raw).hexdigest() + '.jsonl'
            objects[key], counts[key] = raw, len(rows)
            return {'manifest_key': key, 'segments': len(rows), 'calls': len(rows)}

        stage = Mock(side_effect=stage)

        class Conn:
            def execute(self, query, args=None):
                result = {'ok': True} if args is None else {'row_count': counts[args[0]]}
                return types.SimpleNamespace(fetchone=lambda: result)

        conn = Conn()
        modules = {
            'lalm.annotation.store': types.SimpleNamespace(
                connect=lambda: contextlib.nullcontext(conn), s3=lambda: client,
                dataset_bucket=lambda: 'datasets'),
            'lalm.annotation.prepare': types.SimpleNamespace(pending_work=Mock(side_effect=[6000, 0, 0])),
            'lalm.annotation.organization_calls': types.SimpleNamespace(stage_pilot=stage)}
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', modules), \
                patch('lalm.annotation.organization_feed.time.sleep') as sleep:
            for name in ('dialogs-native.csv', 'relations-native.csv'):
                (Path(tmp) / name).write_text(name)
            selection = Path(tmp) / 'selected.json'
            selection.write_text(json.dumps([{'conversation_id': str(i), 'sha256': str(i)}
                                             for i in range(13)]))
            feed(tmp, selection, tmp)
            self.assertEqual(stage.call_count, 2)
            self.assertEqual(sorted(counts.values()), [1, 12])
            sleep.assert_called_once_with(30)
            stage.reset_mock()
            feed(tmp, selection, tmp)
            stage.assert_not_called()
            manifest = next(key for key in objects if key.startswith('incoming/'))
            objects[manifest] = b'corrupted'
            with self.assertRaises(AssertionError):
                feed(tmp, selection, tmp)
            stage.assert_not_called()
