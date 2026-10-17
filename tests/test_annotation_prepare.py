import hashlib
import io
import json
import tarfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import pyarrow as pa

from lalm.annotation.prepare import segments, tar_audio_rows, parquet_source_kind, prepare_asset, audio_rows


def wav(values, rate=16000):
    stream = io.BytesIO()
    sf.write(stream, values, rate, format='WAV', subtype='PCM_16')
    return stream.getvalue()


class SourcePreparationTests(unittest.TestCase):
    def test_storage_full_retries_same_source_without_quarantining_or_losing_cursor(self):
        from botocore.exceptions import ClientError
        from lalm.annotation import prepare
        asset = {'id': 'source', 'cursor_json': {'rows': 128, 'samples': 100}}
        failure = ClientError({'Error': {'Code': 'XMinioStorageFull'}}, 'PutObject')
        calls = []

        def preparing(row):
            calls.append(row)
            if len(calls) == 1:
                raise failure
            prepare.STOP = True

        with patch.object(prepare, 'STOP', False), \
                patch.object(prepare.signal, 'signal'), \
                patch.object(prepare, 'connect') as db, \
                patch.object(prepare, 'select_source', return_value=asset), \
                patch.object(prepare, 'prepare_asset', side_effect=preparing), \
                patch.object(prepare.time, 'sleep') as sleep, \
                patch('builtins.print'):
            db.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = {'ok': True}
            prepare.main()
            self.assertEqual(calls, [asset, asset])
            self.assertEqual(asset['cursor_json'], {'rows': 128, 'samples': 100})
            sleep.assert_called_once_with(60)
            writes = db.return_value.__enter__.return_value.execute.call_args_list
            self.assertFalse(any('NEEDS_REVIEW' in call.args[0] for call in writes))
            self.assertFalse(any('cursor_json=' in call.args[0] for call in writes))

    def test_filtered_golos_rows_advance_without_waiting_for_queue_capacity(self):
        asset = {'id': 'source', 'codec': 'audio', 'cursor_json': {},
                 'object_key': 'staging/golos/rev/crowd/train/train-00000.tar'}
        for protected in (False, True):
            with self.subTest(protected=protected), \
                    patch('lalm.annotation.prepare.s3') as client, \
                    patch('lalm.annotation.prepare.connect') as db, \
                    patch('lalm.annotation.prepare.load_reference_index'), \
                    patch('lalm.annotation.prepare.audio_rows', return_value=iter([
                        ('a', b'audio-a', None), ('b', b'audio-b', None)])), \
                    patch('lalm.annotation.prepare.gigaam_reference',
                          return_value={'text': 'reference'} if protected else None), \
                    patch('lalm.annotation.prepare.pending_work',
                          side_effect=AssertionError('filtered input must not wait')) as pending, \
                    patch('lalm.annotation.prepare.segments') as decode:
                hashes = [hashlib.sha256(raw).hexdigest() for raw in
                          (b'audio-a', b'audio-b')] if protected else []
                client.return_value.get_object.return_value = {'Body': io.BytesIO(
                    json.dumps({'audio_sha256': hashes, 'conversation_ids': []}).encode())}
                prepare_asset(asset)
                pending.assert_not_called()
                decode.assert_not_called()
                client.return_value.put_object.assert_not_called()
                writes = db.return_value.__enter__.return_value.execute.call_args_list
                cursor = next(c.args[1][0].obj for c in writes if 'SET cursor_json=' in c.args[0])
                self.assertEqual((cursor['rows'], cursor['samples'], cursor['errors']), (2, 0, 0))
                self.assertEqual(cursor['unreferenced_rows'], 0 if protected else 2)

    def test_eligible_audio_still_waits_before_decode_or_upload(self):
        from types import SimpleNamespace
        asset = {'id': 'source', 'bucket': 'speech-corpora', 'codec': 'audio',
                 'object_key': 'staging/golos/rev/crowd/train/train-00000.tar',
                 'bytes': 10, 'etag': 'fixed', 'cursor_json': {}}
        with patch('lalm.annotation.prepare.s3') as client, \
                patch('lalm.annotation.prepare.connect'), \
                patch('lalm.annotation.prepare.load_reference_index'), \
                patch('lalm.annotation.prepare.audio_rows', return_value=iter([('0', b'raw', None)])), \
                patch('lalm.annotation.prepare.gigaam_reference',
                      return_value={'text': 'reference', 'source_duration': 2}), \
                patch('lalm.annotation.prepare.sf.info', return_value=SimpleNamespace(channels=1, duration=2)), \
                patch('lalm.annotation.prepare.segments', return_value=iter([(0, 0, 2, b'prepared')])) as decode, \
                patch('lalm.annotation.prepare.pending_work', side_effect=[6000, 0]) as pending, \
                patch('lalm.annotation.prepare.time.sleep') as sleep, \
                patch.dict('os.environ', {'PREPARE_MAX_PENDING': '6000'}):
            client.return_value.get_object.return_value = {'Body': io.BytesIO(
                json.dumps({'audio_sha256': [], 'conversation_ids': []}).encode())}
            def waiting(_):
                decode.assert_not_called()
                client.return_value.put_object.assert_not_called()
            sleep.side_effect = waiting
            prepare_asset(asset)
            self.assertEqual(pending.call_count, 2)
            sleep.assert_called_once_with(10)
            manifests = [c.kwargs['Body'] for c in client.return_value.put_object.call_args_list
                         if c.kwargs['Key'].startswith('incoming/')]
            self.assertEqual(len(manifests), 1)
            self.assertEqual(json.loads(manifests[0])['annotation_policy'], 'gigaam-single-pass-v1')

    def test_failed_reference_checksum_does_not_skip_source_rows(self):
        asset = {'id': 'source', 'codec': 'audio',
                 'object_key': 'staging/golos/rev/crowd/train/train-00000.tar'}
        with patch('lalm.annotation.prepare.s3') as client, \
                patch('lalm.annotation.prepare.audio_rows') as audio, \
                patch('lalm.annotation.prepare.load_reference_index', side_effect=ValueError('checksum')):
            client.return_value.get_object.return_value = {'Body': io.BytesIO(
                json.dumps({'audio_sha256': [], 'conversation_ids': []}).encode())}
            with self.assertRaisesRegex(ValueError, 'checksum'):
                prepare_asset(asset)
            audio.assert_not_called()

    def test_golos_reference_only_skips_unmapped_and_partial_inputs_with_durable_cursor(self):
        from types import SimpleNamespace
        asset = {'id': 'source', 'bucket': 'speech-corpora', 'codec': 'audio',
                 'object_key': 'staging/golos/rev/crowd/train/train-00000.tar',
                 'bytes': 10, 'etag': 'fixed', 'cursor_json': {}}
        for kind, reference, end, expected in [('missing', None, 2, 0),
                ('partial', {'text': 'GigaAM', 'source_duration': 2}, 1, 0),
                ('wrong-scope', {'text': 'GigaAM', 'source_duration': 1}, 2, 0),
                ('full', {'text': 'GigaAM', 'source_duration': 2}, 2, 1)]:
            with self.subTest(kind=kind), patch('lalm.annotation.prepare.s3') as client, \
                    patch('lalm.annotation.prepare.load_reference_index'), \
                    patch('lalm.annotation.prepare.connect') as db, \
                    patch('lalm.annotation.prepare.audio_rows', return_value=iter([('0', b'raw', 'source text')])), \
                    patch('lalm.annotation.prepare.gigaam_reference', return_value=reference), \
                    patch('lalm.annotation.prepare.sf.info', return_value=SimpleNamespace(channels=1, duration=2)), \
                    patch('lalm.annotation.prepare.segments', return_value=iter([(0, 0, end, b'prepared')])):
                client.return_value.get_object.return_value = {'Body': io.BytesIO(
                    json.dumps({'audio_sha256': [], 'conversation_ids': []}).encode())}
                db.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = {'n': 0}
                prepare_asset(asset)
                manifests = [c.kwargs['Body'] for c in client.return_value.put_object.call_args_list
                             if c.kwargs['Key'].startswith('incoming/')]
                self.assertEqual(len(manifests), expected)
                if manifests:
                    self.assertEqual(json.loads(manifests[0])['annotation_policy'], 'gigaam-single-pass-v1')
                writes = db.return_value.__enter__.return_value.execute.call_args_list
                cursor = next(c.args[1][0].obj for c in writes if 'SET cursor_json=' in c.args[0])
                self.assertEqual(cursor['rows'], 1)
                self.assertEqual(cursor['errors'], 0)
                self.assertEqual(cursor['selection_policy'], 'gigaam-reference-only-v1')
                self.assertEqual(cursor['selection_policy_start_row'], 0)
                self.assertEqual(cursor['unreferenced_rows'], int(kind == 'missing'))
                self.assertEqual(cursor['unaligned_windows'], int(kind in ('partial', 'wrong-scope')))

    def test_gigaam_whole_file_label_is_not_attached_to_partial_or_multichannel_audio(self):
        from types import SimpleNamespace
        asset = {'id': 'source', 'bucket': 'speech-corpora', 'codec': 'audio', 'bytes': 10, 'etag': 'fixed',
                 'object_key': 'staging/sova-rudevices/revision/train.parquet', 'cursor_json': {}}
        for channels, start, end, expected in [(1, 0, 2, True), (1, 0, 1, False),
                                               (1, 1, 2, False), (2, 0, 2, False)]:
            with self.subTest(channels=channels, start=start, end=end), \
                    patch('lalm.annotation.prepare.load_reference_index'), \
                    patch('lalm.annotation.prepare.s3') as client, \
                    patch('lalm.annotation.prepare.connect') as db, \
                    patch('lalm.annotation.prepare.audio_rows', return_value=iter([('0', b'raw', 'source text')])), \
                    patch('lalm.annotation.prepare.gigaam_reference', return_value={'text': 'GigaAM'}), \
                    patch('lalm.annotation.prepare.sf.info', return_value=SimpleNamespace(channels=channels, duration=2)), \
                    patch('lalm.annotation.prepare.segments', return_value=iter([(0, start, end, b'prepared')])):
                client.return_value.get_object.return_value = {'Body': io.BytesIO(
                    json.dumps({'audio_sha256': [], 'conversation_ids': []}).encode())}
                db.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = {'n': 0}
                prepare_asset(asset)
                manifest = next(call.kwargs['Body'] for call in client.return_value.put_object.call_args_list
                                if call.kwargs['Key'].startswith('incoming/'))
                row = json.loads(manifest)
                self.assertEqual(row.get('annotation_policy') == 'gigaam-single-pass-v1', expected)
                self.assertEqual('gigaam_reference' in row['source_provenance'], expected)
                self.assertEqual(row['source_provenance']['original_label'], 'source text')

    def test_protected_session_is_excluded_before_audio_download(self):
        session = '089e27e9-89b4-4200-9129-db0813970e2a'
        asset = {'id': 'source', 'codec': 'audio',
                 'object_key': f'staging/bitmanager-sessions/revision/sessions/2026/04/07/{session}/customer.opus'}
        registry = {'audio_sha256': [], 'conversation_ids': [session]}
        with patch('lalm.annotation.prepare.s3') as client, patch('lalm.annotation.prepare.connect') as db:
            client.return_value.get_object.return_value = {'Body': io.BytesIO(json.dumps(registry).encode())}
            prepare_asset(asset)
            client.return_value.get_object.assert_called_once()
            self.assertEqual(client.return_value.get_object.call_args.kwargs['Key'], 'benchmarks/holdouts.json')
            statement = db.return_value.__enter__.return_value.execute.call_args.args[0]
            self.assertIn('EXCLUDED_HOLDOUT', statement)

    def test_parquet_metadata_is_not_treated_as_broken_audio(self):
        self.assertEqual(parquet_source_kind(pa.schema([
            ('filepath', pa.string()), ('gigaam-v3-ctc', pa.string())])), 'reference_metadata')
        self.assertEqual(parquet_source_kind(pa.schema([
            ('audio', pa.struct([('bytes', pa.binary()), ('path', pa.string())])),
            ('text', pa.string())])), 'embedded_audio')
        with self.assertRaisesRegex(ValueError, 'unsupported_parquet_audio_schema'):
            parquet_source_kind(pa.schema([('audio', pa.string()), ('filepath', pa.string())]))
        with self.assertRaisesRegex(ValueError, 'unsupported_parquet_source_schema'):
            parquet_source_kind(pa.schema([('text', pa.string())]))

    def test_long_speech_has_full_coverage_and_bounded_overlap(self):
        with patch('lalm.annotation.prepare.webrtcvad.Vad') as vad:
            vad.return_value.is_speech.return_value = True
            chunks = list(segments(wav(np.full(45*16000, 0.2))))
        self.assertEqual(chunks[0][1], 0)
        self.assertEqual(chunks[-1][2], 45)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertGreater(current[1], previous[1])
            self.assertLessEqual(current[1], previous[2])
            self.assertLessEqual(previous[2]-current[1], 0.5)
        for _, start, end, audio in chunks:
            self.assertLessEqual(end-start, 20)
            values, rate = sf.read(io.BytesIO(audio))
            self.assertEqual(rate, 16000)
            self.assertEqual(len(values), round((end-start)*16000))

    def test_channels_remain_separate_during_resampling(self):
        source = np.column_stack([np.full(8000, 0.2), np.full(8000, -0.3)])
        with patch('lalm.annotation.prepare.webrtcvad.Vad') as vad:
            vad.return_value.is_speech.return_value = True
            chunks = list(segments(wav(source, 8000)))
        self.assertEqual([r[0] for r in chunks], [0, 1])
        for row, expected in zip(chunks, (0.2, -0.3)):
            values, rate = sf.read(io.BytesIO(row[3]))
            self.assertEqual(rate, 16000)
            self.assertEqual(len(values), 16000)
            self.assertAlmostEqual(float(values[100:-100].mean()), expected, places=3)

    def test_silence_is_not_queued(self):
        self.assertEqual(list(segments(wav(np.zeros(32000)))), [])


class ArchiveLabelTests(unittest.TestCase):
    def test_sidecars_before_and_after_audio_keep_exact_identity(self):
        data = io.BytesIO()
        entries = [('a.wav', b'audio-a'), ('a.txt', b'original a'),
                   ('b.json', b'{"text":"original b","model":"legacy"}'),
                   ('b.flac', b'audio-b'), ('c.opus', b'audio-c')]
        with tarfile.open(fileobj=data, mode='w') as archive:
            for name, content in entries:
                member = tarfile.TarInfo(name); member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        data.seek(0)
        rows = list(tar_audio_rows(data))
        self.assertEqual([r[0] for r in rows], ['a.wav', 'b.flac', 'c.opus'])
        self.assertEqual(rows[0][2]['archive_sidecars'][0],
                         {'member': 'a.txt', 'value': 'original a'})
        self.assertEqual(rows[1][2]['archive_sidecars'][0]['value']['model'], 'legacy')
        self.assertIsNone(rows[2][2])


class BoundedArchiveReadTests(unittest.TestCase):
    def test_generator_pause_does_not_leave_an_http_response_open(self):
        data = io.BytesIO()
        entries = [('a.wav', b'a'*1200000), ('b.wav', b'b'*1200000)]
        with tarfile.open(fileobj=data, mode='w') as archive:
            for name, content in entries:
                member = tarfile.TarInfo(name); member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        payload = data.getvalue()
        class Client:
            bodies = []
            def head_object(self, **kwargs):
                return {'ContentLength': len(payload), 'ETag': 'immutable'}
            def get_object(self, **kwargs):
                self.assert_match = kwargs['IfMatch']
                self.assert_range = kwargs['Range']
                start, end = map(int, kwargs['Range'].removeprefix('bytes=').split('-'))
                body = io.BytesIO(payload[start:end+1]); self.bodies.append(body)
                return {'Body': body}
        client = Client()
        rows = audio_rows(client, {'codec': 'tar', 'bucket': 'source', 'object_key': 'audio.tar', 'etag': 'immutable'})
        first = next(rows)
        self.assertEqual(first[:2], entries[0])
        self.assertTrue(all(body.closed for body in client.bodies))
        self.assertEqual(client.assert_match, 'immutable')
        # Resume after arbitrary backpressure without keeping a socket alive.
        self.assertEqual(next(rows)[:2], entries[1])
        with self.assertRaises(StopIteration): next(rows)
        self.assertTrue(all(body.closed for body in client.bodies))

    def test_changed_archive_is_rejected_before_reading_members(self):
        from unittest.mock import Mock
        client = Mock()
        client.head_object.return_value = {'ContentLength': 1000, 'ETag': 'changed'}
        with self.assertRaisesRegex(ValueError, 'source_changed'):
            next(audio_rows(client, {'codec': 'tar', 'bucket': 'source', 'object_key': 'audio.tar', 'etag': 'original'}))
        client.get_object.assert_not_called()


if __name__ == '__main__':
    unittest.main()
