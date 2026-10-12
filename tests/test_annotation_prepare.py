import io
import tarfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from lalm.annotation.prepare import segments, tar_audio_rows


def wav(values, rate=16000):
    stream = io.BytesIO()
    sf.write(stream, values, rate, format='WAV', subtype='PCM_16')
    return stream.getvalue()


class SourcePreparationTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
