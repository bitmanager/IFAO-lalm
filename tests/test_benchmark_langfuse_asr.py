import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lalm.benchmark_langfuse_asr import MODEL, agy_result, prepare


class BenchmarkContracts(unittest.TestCase):
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
                                    recordings=p / "recordings.json", roots=p / "roots.json"))
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

    def test_missing_recording_remains_in_coverage(self):
        calls, media = self.prepared([], [self.root()])
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["media_id"])
        self.assertEqual(media, [])

    def test_upload_in_window_does_not_admit_earlier_call(self):
        calls, _ = self.prepared([self.recording()], [self.root(start="2026-09-11T20:59:59Z")])
        self.assertEqual(calls, [])

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

    def test_soft_denial_is_not_asr_success(self):
        for steps in ([], [self.tool(state="ERROR")]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                agy_result(self.events(steps), Path("/audio.wav"))

    def test_external_asr_and_extra_file_reads_rejected(self):
        for extra in (self.tool(name="run_command"), self.tool(path="/transcript.txt")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                agy_result(self.events([self.tool(), extra]), Path("/audio.wav"))


if __name__ == "__main__":
    unittest.main()
