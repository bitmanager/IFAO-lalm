import importlib.util
import json
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / 'scripts/build_golos_gigaam_reference.py'
spec = importlib.util.spec_from_file_location('golos_reference_builder', path)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class GolosReferenceTests(unittest.TestCase):
    source = 'staging/golos/rev/crowd/train/train-00000.tar'

    def row(self):
        meta = {'audio_uri': 's3://speech-corpora/' + self.source,
                'audio_locator': 'member=audio.opus', 'audio_sha256': 'a' * 64,
                'duration': 5.0, 'start': 0.0, 'corpus': 'golos', 'split': 'train',
                'transcript_kind': 'GigaAM-v3-CTC pseudo-label',
                'transcript_weights_sha256': builder.WEIGHTS,
                'gigaam_transcript': 'не согласен', 'qwen_answer': 'другой ответ'}
        return {'metadata_json': json.dumps(meta), 'sample_rate': 16000,
                'num_samples': 80000, 'corpus': 'golos', 'split': 'train'}

    def test_keeps_explicit_gigaam_label_and_source_identity(self):
        result = builder.extract(self.row(), {self.source})
        self.assertEqual(result['gigaam_transcription'], 'не согласен')
        self.assertEqual(result['source_row'], 'audio.opus')
        self.assertEqual(result['audio_sha256'], 'a' * 64)
        self.assertEqual(result['gigaam_model'], 'v3_ctc')

    def test_rejects_crop_wrong_model_unlinked_audio_and_invalid_labels(self):
        for field, value in [('start', 0.2), ('duration', 4.8),
                             ('transcript_weights_sha256', 'b' * 64),
                             ('audio_uri', 's3://speech-corpora/staging/golos/rev/test/a.tar'),
                             ('audio_locator', 'member=../audio.opus'),
                             ('audio_sha256', 'invalid'), ('gigaam_transcript', '')]:
            with self.subTest(field=field):
                row = self.row();meta = json.loads(row['metadata_json']);meta[field] = value
                row['metadata_json'] = json.dumps(meta)
                with self.assertRaises(ValueError):
                    builder.extract(row, {self.source})
        row = self.row();row['split'] = 'validation'
        with self.assertRaises(ValueError):
            builder.extract(row, {self.source})
        row = self.row();row['metadata_json'] = '[]'
        with self.assertRaises(ValueError):
            builder.extract(row, {self.source})


if __name__ == '__main__':
    unittest.main()
