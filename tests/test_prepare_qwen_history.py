import json
import tempfile
import unittest
from pathlib import Path
import wave

from lalm.prepare_qwen_history import digest, prepare_call, speech_segments


class QwenHistoryPreparationContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.audio = Path(self.directory.name) / "user.wav"
        with wave.open(str(self.audio), "wb") as file:
            file.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            file.writeframes(b"\x01\x00" * 5 * 16000)
        self.sha = digest(self.audio)
        self.asset = dict(media_id="media", user_sha256=self.sha, call_ids=["call"])
        self.call = dict(call_id="call", online_turns=[
            dict(role="client", text="A future transcript must never leak", at=.1),
            dict(role="agent", text="Past agent context", at=.2),
            dict(role="agent", text="Current agent context", at=1.9),
            dict(role="agent", text="Future agent context", at=4),
        ])
        self.vad = dict(audio_sha256=self.sha, duration_seconds=5, speech_detected=True,
                        timestamps=[dict(start=2, end=3)], backend="silero-vad-onnx", version="6.2.3", config={})

    def prepared(self):
        return prepare_call("test", self.asset, self.call, self.vad, self.audio, {"vad": "v" * 64})

    def test_only_past_agent_context_enters_inputs(self):
        prepared = self.prepared()
        history = prepared["segments"][0]["history"]
        self.assertEqual([h["content"] for h in history], ["Past agent context"])
        self.assertTrue(all(h["role"] == "assistant" for h in history))
        self.assertNotIn("future transcript", json.dumps(prepared).lower())
        self.assertFalse(prepared["history_policy"]["original_client_text_included"])

    def test_aliases_hold_all_original_agent_history(self):
        self.asset["call_ids"].append("other-call")
        prepared = self.prepared()
        self.assertEqual(prepared["segments"][0]["history"], [])
        self.assertIn("agent_history_unverified", prepared["history_policy"]["flags"])
        self.assertEqual(prepared["source_call_ids"], ["call", "other-call"])

    def test_invalid_clock_is_explicit_and_does_not_silently_add_history(self):
        for bad_time in (None, float("nan"), -1, 7):
            self.call["online_turns"][0]["at"] = bad_time
            with self.subTest(bad_time=bad_time):
                prepared = self.prepared()
                self.assertEqual(prepared["segments"][0]["history"], [])
                self.assertTrue(prepared["history_policy"]["original_agent_history_held"])

    def test_close_native_spans_preserve_coverage_without_duplicate_samples(self):
        raw = [dict(start=0, end=1), dict(start=1.1, end=2), dict(start=2, end=5)]
        segments = speech_segments(raw, 5)
        self.assertEqual(segments[0]["start"], 0)
        self.assertEqual(segments[-1]["end"], 5)
        for segment, span in zip(segments, raw):
            self.assertLessEqual(segment["start"], span["start"])
            self.assertGreaterEqual(segment["end"], span["end"])
        for first, second in zip(segments, segments[1:]):
            self.assertLessEqual(first["end"], second["start"])

    def test_changed_audio_fails_cached_vad_identity(self):
        with self.audio.open("ab") as file:
            file.write(b"modified")
        with self.assertRaisesRegex(ValueError, "hash/duration"):
            self.prepared()

    def test_no_speech_or_mismatched_native_identity_cannot_be_transcribed(self):
        self.vad["speech_detected"] = False
        with self.assertRaisesRegex(ValueError, "gate/source identity"):
            self.prepared()
        self.vad.update(speech_detected=True, audio_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "gate/source identity"):
            self.prepared()

    def test_long_or_overlapping_spans_fail_instead_of_silent_truncation(self):
        with self.assertRaisesRegex(ValueError, "exceeds 30s"):
            speech_segments([dict(start=0, end=31)], 31)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            speech_segments([dict(start=0, end=2), dict(start=1, end=3)], 4)


if __name__ == "__main__":
    unittest.main()
