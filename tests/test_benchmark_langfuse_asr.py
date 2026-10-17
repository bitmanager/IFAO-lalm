import json
import csv
import io
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lalm.benchmark_langfuse_asr import MODEL, agy_result, download, incomplete_gemini_status, native_cli, prepare, report, sha, zero_user_audio
from lalm.benchmark_gemini_api import MODEL as API_MODEL, billed_cost, validate


class BenchmarkContracts(unittest.TestCase):
    def presence_report_fixture(self, root, changes):
        calls, assets, checks = [], [], []
        for i, delta in enumerate(changes):
            mid = str(i)
            calls.append({"call_id": "test-" + mid, "trace_id": mid,
                          "media_id": mid, "source_environment": "test"})
            assets.append({"media_id": mid, "status": "complete", "user_sha256": "sha-" + mid})
            checks.append({"source_keys": ["test:" + mid], "audio_sha256": "sha-" + mid,
                           "model": MODEL, "status": "complete", "native_validated": True,
                           "vad_speech_detected": True, "speech_status": "speech", "language": "ru",
                           "language_conflict": False, **delta})
        for name, rows in (("calls", calls), ("assets", assets), ("presence", checks)):
            (root / (name + ".jsonl")).write_text("".join(json.dumps(x) + "\n" for x in rows))
        return SimpleNamespace(root=root, presence_index=root / "presence.jsonl")

    def test_review_gate_excludes_empty_unconfirmed_foreign_and_pending_audio(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = self.presence_report_fixture(root, [{}, {"vad_speech_detected": False},
                {"speech_status": "no_speech"}, {"language": "en"},
                {"status": "not_started"}, {"native_validated": False}, {"language_conflict": True}])
            report(args)
            review = next(root.glob("review-*.csv"))
            with review.open() as f:
                self.assertEqual([r["call_id"] for r in csv.DictReader(f)], ["test-0"])
            metadata = json.loads(review.with_suffix(".meta.json").read_text())
            self.assertEqual(metadata["presence_index_sha256"], sha(args.presence_index))
            self.assertEqual(metadata["quality_gate"], "silero_and_gemini_russian_speech")

    def test_review_gate_rejects_stale_audio_and_ambiguous_source_keys(self):
        for changes in ([{"audio_sha256": "stale"}], [{}, {"source_keys": ["test:0"]}]):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as d:
                args = self.presence_report_fixture(Path(d), changes)
                with self.assertRaises(ValueError):
                    report(args)

    def test_review_gate_holds_conflicting_full_transcript_language(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = self.presence_report_fixture(root, [{}])
            cache = root / "gemini/blind/0/result.json"
            cache.parent.mkdir(parents=True)
            cache.write_text(json.dumps({"audio_sha256": "sha-0", "model": MODEL,
                "variant": "blind", "prompt_sha256": "prompt", "status": "complete",
                "native_result": {"status": "SUCCESS", "structured_output": {
                    "language": "ru, uz", "transcript": "Mixed-language speech"}}}))
            report(args)
            with next(root.glob("review-*.csv")).open() as f:
                self.assertEqual(list(csv.DictReader(f)), [])

    def test_zero_pcm_gate_preserves_any_nonzero_samples(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "user.wav"
            for samples, expected in ((b"\0\0" * 1000, True), (b"\0\0" * 1000 + b"\1\0", False)):
                with wave.open(str(p), "wb") as audio:
                    audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    audio.writeframes(samples)
                self.assertEqual(zero_user_audio(p), expected)

    def test_zero_pcm_gate_rejects_unconverted_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "user.wav"
            with wave.open(str(p), "wb") as audio:
                audio.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0" * 4000)
            with self.assertRaises(ValueError):
                zero_user_audio(p)

    def export(self, path, rows):
        path.write_text(json.dumps({"status": 200, "body": {
            "data": rows, "meta": {"truncated": False}}}))

    def root(self, trace="a", start="2026-09-12T00:00:00Z"):
        return {"id": "root-" + trace, "traceId": trace, "startTime": start,
                "tags": ["call", "ru"], "metadata": "{}"}

    def recording(self, trace="a", media="same"):
        return {"id": "recording-" + trace, "traceId": trace,
                "startTime": "2026-09-12T00:01:00Z", "tags": ["call", "ru"],
                "metadata": "{}", "output": json.dumps({
                    "recording": f"@@@langfuseMedia:type=audio/wav|id={media}|source=bytes@@@",
                    "transcript": [{"role": "client", "text": "Начни разговор"},
                                   {"role": "client", "text": "я учусь"}]})}

    def prepared(self, recordings, roots):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            self.export(p / "recordings.json", recordings)
            self.export(p / "roots.json", roots)
            prepare(SimpleNamespace(root=p / "out", environment="test",
                                    recordings=p / "recordings.json", roots=p / "roots.json",
                                    start="2026-09-11T21:00:00Z", end="2026-10-11T21:00:00Z"))
            calls = [json.loads(x) for x in (p / "out/calls.jsonl").read_text().splitlines()]
            media = [json.loads(x) for x in (p / "out/media.jsonl").read_text().splitlines()]
            return calls, media

    def test_duplicate_audio_keeps_distinct_calls(self):
        calls, media = self.prepared([self.recording("a"), self.recording("b")],
                                    [self.root("a"), self.root("b")])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(media), 1)
        self.assertEqual(len(media[0]["call_ids"]), 2)
        self.assertEqual(calls[0]["online_text"], "я учусь")

    def test_mp3_marker_keeps_identity_and_mime(self):
        recording = self.recording(media="mp3-id")
        recording["output"] = recording["output"].replace("audio/wav", "audio/mpeg")
        calls, media = self.prepared([recording], [self.root()])
        self.assertEqual(calls[0]["media_id"], "mp3-id")
        self.assertEqual(calls[0]["media_content_type"], "audio/mpeg")
        self.assertEqual(media, [{"media_id": "mp3-id", "content_type": "audio/mpeg",
                                 "call_ids": ["test-a"]}])

    def test_same_media_with_conflicting_mime_fails_loud(self):
        mp3 = self.recording("b")
        mp3["output"] = mp3["output"].replace("audio/wav", "audio/mpeg")
        with self.assertRaises(ValueError):
            self.prepared([self.recording("a"), mp3], [self.root("a"), self.root("b")])

    def download_args(self, root):
        (root / "media.jsonl").write_text(json.dumps({"media_id": "mp3-id", "content_type": "audio/mpeg",
                                                    "call_ids": ["test-a"]}) + "\n")
        return SimpleNamespace(root=root, api_container="native-api", api_version="4.10.0",
                               ffmpeg="ffmpeg", media_proxy=None, workers=1)

    def test_download_rejects_descriptor_identity_or_mime_before_network(self):
        for identity, mime in (("other-id", "audio/mpeg"), ("mp3-id", "audio/wav")):
            with self.subTest(identity=identity, mime=mime), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                args = self.download_args(root)
                with patch("lalm.benchmark_langfuse_asr.native_cli", return_value={"mediaId": identity, "contentType": mime}), \
                        patch("lalm.benchmark_langfuse_asr.urllib.request.urlopen") as request:
                    download(args)
                request.assert_not_called()
                result = json.loads((root / "download/mp3-id.json").read_text())
                self.assertEqual((result["status"], result["error"]),
                                 ("failed", "Media descriptor identity/type mismatch"))

    @unittest.skipUnless(shutil.which("ffmpeg"), "Native FFmpeg required for compressed-audio regression")
    def test_mp3_native_decode_preserves_compressed_sha_and_rechecks_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "fixture.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"".join(struct.pack("<h", 4000 if i % 80 < 40 else -4000) for i in range(16000)))
            mp3 = root / "fixture.mp3"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), str(mp3)], check=True)
            body = mp3.read_bytes()
            descriptor = {"mediaId": "mp3-id", "contentType": "audio/mpeg", "contentLength": len(body),
                          "url": "https://example.invalid/audio?secret=do-not-persist"}
            args = self.download_args(root)
            with patch("lalm.benchmark_langfuse_asr.native_cli", return_value=descriptor), \
                    patch("lalm.benchmark_langfuse_asr.urllib.request.urlopen", return_value=io.BytesIO(body)):
                download(args)
            result = json.loads((root / "download/mp3-id.json").read_text())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["source_sha256"], sha(mp3))
            self.assertEqual(result["source_content_type"], "audio/mpeg")
            self.assertEqual(Path(result["source_audio_path"]).read_bytes(), body)
            self.assertNotEqual(result["sha256"], result["source_sha256"])
            self.assertFalse(zero_user_audio(Path(result["user_audio_path"])))
            native = json.loads((root / "giga/data/calls.raw.jsonl").read_text())
            self.assertEqual(native["provenance"]["source_sha256"], sha(mp3))
            self.assertNotIn("do-not-persist", (root / "download/mp3-id.json").read_text())
            Path(result["source_audio_path"]).write_bytes(b"tampered compressed source")
            with patch("lalm.benchmark_langfuse_asr.native_cli", side_effect=LookupError("media_not_uploaded")) as cli:
                download(args)
            cli.assert_called_once()
            self.assertEqual(json.loads((root / "download/mp3-id.json").read_text())["status"], "media_not_uploaded")

    @unittest.skipUnless(shutil.which("ffmpeg"), "Native FFmpeg required for compressed-audio regression")
    def test_invalid_mp3_does_not_fall_back_or_leak_url(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = self.download_args(root)
            body = b"not an mp3 audio stream"
            descriptor = {"mediaId": "mp3-id", "contentType": "audio/mpeg", "contentLength": len(body),
                          "url": "https://example.invalid/audio?secret=do-not-persist"}
            with patch("lalm.benchmark_langfuse_asr.native_cli", return_value=descriptor), \
                    patch("lalm.benchmark_langfuse_asr.urllib.request.urlopen", return_value=io.BytesIO(body)):
                download(args)
            saved = (root / "download/mp3-id.json").read_text()
            self.assertEqual(json.loads(saved)["status"], "failed")
            self.assertNotIn("do-not-persist", saved)
            self.assertEqual((root / "giga/data/calls.raw.jsonl").read_text(), "")

    @unittest.skipUnless(shutil.which("ffmpeg"), "Native FFmpeg required for compressed-audio regression")
    def test_zero_mp3_removes_compressed_and_derived_audio_keeps_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "silent.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\0\0" * 16000)
            mp3 = root / "silent.mp3"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), str(mp3)], check=True)
            body = mp3.read_bytes()
            descriptor = {"mediaId": "mp3-id", "contentType": "audio/mpeg", "contentLength": len(body),
                          "url": "https://example.invalid/audio"}
            args = self.download_args(root)
            with patch("lalm.benchmark_langfuse_asr.native_cli", return_value=descriptor), \
                    patch("lalm.benchmark_langfuse_asr.urllib.request.urlopen", return_value=io.BytesIO(body)):
                download(args)
            result = json.loads((root / "download/mp3-id.json").read_text())
            self.assertEqual(result["status"], "excluded_no_user_speech")
            self.assertEqual(result["source_sha256"], sha(mp3))
            self.assertTrue(result["audio_deleted"])
            for name in ("source-audio/mp3-id.mp3", "audio/mp3-id.wav", "user-audio/mp3-id.wav"):
                self.assertFalse((root / name).exists())
            self.assertEqual((root / "giga/data/calls.raw.jsonl").read_text(), "")

    def test_excluded_cache_cannot_hide_a_manifest_mime_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = self.download_args(root)
            (root / "download").mkdir()
            (root / "download/mp3-id.json").write_text(json.dumps({
                "status": "excluded_no_user_speech", "content_type": "audio/wav"}))
            with patch("lalm.benchmark_langfuse_asr.native_cli") as cli:
                download(args)
            cli.assert_not_called()
            result = json.loads((root / "download/mp3-id.json").read_text())
            self.assertEqual((result["status"], result["error"]),
                             ("failed", "Cached media MIME does not match manifest"))

    def test_missing_recording_remains_in_coverage(self):
        calls, media = self.prepared([], [self.root()])
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["media_id"])
        self.assertEqual(media, [])

    def test_unsupported_nonempty_media_marker_fails_loud(self):
        recording = self.recording()
        output = json.loads(recording["output"])
        output["recording"] = "unrecognized-audio-link"
        recording["output"] = json.dumps(output)
        with self.assertRaises(ValueError):
            self.prepared([recording], [self.root()])

    def test_upload_in_window_does_not_admit_earlier_call(self):
        calls, _ = self.prepared([self.recording()], [self.root(start="2026-09-11T20:59:59Z")])
        self.assertEqual(calls, [])

    def test_custom_call_window_uses_inclusive_start_exclusive_end(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            self.export(p / "recordings.json", [self.recording("inside"), self.recording("end")])
            self.export(p / "roots.json", [
                self.root("inside", "2026-10-12T10:00:00Z"),
                self.root("end", "2026-10-13T10:00:00Z"),
            ])
            prepare(SimpleNamespace(root=p / "out", environment="prod",
                                    recordings=p / "recordings.json", roots=p / "roots.json",
                                    start="2026-10-12T10:00:00Z", end="2026-10-13T10:00:00Z"))
            calls = [json.loads(x) for x in (p / "out/calls.jsonl").read_text().splitlines()]
            cohort = json.loads((p / "out/cohort.json").read_text())
            self.assertEqual([c["trace_id"] for c in calls], ["inside"])
            self.assertEqual(cohort["start"], "2026-10-12T10:00:00Z")
            self.assertEqual(cohort["end_exclusive"], "2026-10-13T10:00:00Z")

    def test_reversed_call_window_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            self.export(p / "recordings.json", [])
            with self.assertRaises(ValueError):
                prepare(SimpleNamespace(root=p / "out", environment="test",
                                        recordings=p / "recordings.json", roots=None,
                                        start="2026-10-13T10:00:00Z", end="2026-10-12T10:00:00Z"))

    def test_native_langfuse_cli_uses_auto_api_version(self):
        response = SimpleNamespace(returncode=0, stdout='{"status":200,"body":{}}', stderr="")
        with patch("lalm.benchmark_langfuse_asr.subprocess.run", return_value=response) as run:
            native_cli("langfuse-api", "media", "get", "media-id")
        command = run.call_args.args[0]
        self.assertIn("--api-version", command)
        self.assertEqual(command[command.index("--api-version") + 1], "auto")

    def test_native_langfuse_cli_accepts_explicit_api_snapshot(self):
        response = SimpleNamespace(returncode=0, stdout='{"status":200,"body":{}}', stderr="")
        with patch("lalm.benchmark_langfuse_asr.subprocess.run", return_value=response) as run:
            native_cli("langfuse-api", "media", "get", "media-id", api_version="4.10.0")
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--api-version") + 1], "4.10.0")

    def events(self, steps):
        result = {"status": "SUCCESS", "num_turns": 1, "structured_output": {
            "audio_heard": True, "language": "ru", "transcript": "я учусь",
            "uncertain_fragments": [], "context_consistency_notes": []}}
        return "\n".join(json.dumps(e) for e in [
            {"event": "init", "init": {"model": MODEL}},
            *[{"event": "step_update", "step_update": s} for s in steps],
            {"event": "result", "result": result}])

    def tool(self, name="view_file", state="DONE", path="/audio.wav"):
        return {"step_type": "tool", "tool_name": name, "state": state,
                "tool_info": {"parameters": {"AbsolutePath": path}}}

    def test_native_audio_success(self):
        result = agy_result(self.events([self.tool()]), Path("/audio.wav"))
        self.assertEqual(result["structured_output"]["transcript"], "я учусь")

    def test_requested_argon_rejects_fallback_to_flash(self):
        raw = self.events([self.tool()])
        with self.assertRaises(ValueError):
            agy_result(raw, Path('/audio.wav'), expected_model='gemini-4-argon-low')
        argon = raw.replace(MODEL, 'gemini-4-argon-low')
        result = agy_result(argon, Path('/audio.wav'), expected_model='gemini-4-argon-low')
        self.assertTrue(result['structured_output']['audio_heard'])

    def test_soft_denial_is_not_asr_success(self):
        for steps in ([], [self.tool(state="ERROR")]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                agy_result(self.events(steps), Path("/audio.wav"))

    def test_native_schema_finalization_is_not_external_asr(self):
        result = agy_result(self.events([self.tool(), self.tool(name="finish")]), Path("/audio.wav"))
        self.assertEqual(result["structured_output"]["transcript"], "я учусь")

    def test_external_asr_and_extra_file_reads_rejected(self):
        for extra in (self.tool(name="run_command"), self.tool(path="/transcript.txt")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                agy_result(self.events([self.tool(), extra]), Path("/audio.wav"))

    def test_filtered_native_success_is_not_an_unstarted_job(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            self.assertEqual(incomplete_gemini_status(p), "not_completed")
            native = {"event": "result", "result": {
                "status": "SUCCESS", "num_turns": 2,
                "response": "This request was blocked by Gemini's filters.\n{\"transcript\":\"text\"}"}}
            (p / "attempt-2.stream.jsonl").write_text(json.dumps(native))
            (p / "attempt-2.status.json").write_text('{"exit_code": 0}')
            self.assertEqual(incomplete_gemini_status(p), "provider_filtered")
            (p / "attempt-10.stream.jsonl").write_text('{"event":')
            self.assertEqual(incomplete_gemini_status(p), "attempt_pending_validation")
            (p / "attempt-10.stream.jsonl").write_text("")
            (p / "attempt-10.status.json").write_text('{"exit_code": 1}')
            self.assertEqual(incomplete_gemini_status(p), "failed_attempt")
            (p / "attempt-11.stream.jsonl").write_text(json.dumps({
                "event": "result", "result": {"error": "Individual quota reached"}}))
            (p / "attempt-11.status.json").write_text('{"exit_code": 0}')
            self.assertEqual(incomplete_gemini_status(p), "quota_blocked_attempt")

    def api_response(self, **changes):
        native = {"model": API_MODEL, "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"audio_heard": True, "language": "ru", "transcript": "",
                                   "uncertain_fragments": [], "context_consistency_notes": []})}}]}
        native.update(changes)
        return native

    def test_direct_api_silence_is_a_valid_empty_transcript(self):
        self.assertEqual(validate(self.api_response())["transcript"], "")

    def test_direct_api_model_substitution_and_truncation_rejected(self):
        for native in (self.api_response(model="other-model"), self.api_response(choices=[])):
            with self.subTest(native=native), self.assertRaises(ValueError):
                validate(native)
        native = self.api_response()
        native["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            validate(native)

    def test_direct_api_audio_unavailable_is_not_success(self):
        native = self.api_response()
        output = json.loads(native["choices"][0]["message"]["content"])
        output["audio_heard"] = False
        native["choices"][0]["message"]["content"] = json.dumps(output)
        with self.assertRaises(ValueError):
            validate(native)

    def test_unknown_or_nonfinite_api_cost_fails_loud(self):
        for cost in (None, True, "0.1", -1, float("nan"), float("inf")):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                billed_cost({"usage": {"cost": cost}})
        self.assertEqual(billed_cost({"usage": {"cost": 0}}), 0)


if __name__ == "__main__":
    unittest.main()
