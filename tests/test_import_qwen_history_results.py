import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lalm.import_qwen_history_results import (
    CHECKPOINT_SHA256,
    import_item,
    task_fields,
    validate_result,
)


class QwenHistoryImportContracts(unittest.TestCase):
    def setUp(self):
        self.audio_sha = "a" * 64
        self.call = {
            "environment": "test", "media_id": "media-1", "call_id": "call-1",
            "user_sha256": self.audio_sha, "source_call_ids": ["call-1"],
            "history_policy": {"mode": "past_only"},
        }
        self.manifest = {"history_mode": "self_transcripts_and_agent", "calls": [self.call]}
        self.calls = {("test", "media-1"): self.call}
        self.source = {"source_environment": "test", "media_id": "media-1", "audio_sha256": self.audio_sha}
        self.planned = {("test", "media-1"): self.source}
        self.result = {
            "key": "result-1", "status": "complete", "environment": "test", "media_id": "media-1",
            "call_id": "call-1", "audio_sha256": self.audio_sha,
            "recipe": {"checkpoint_sha256": CHECKPOINT_SHA256, "history_mode": "self_transcripts_and_agent"},
            "provenance": {"user_sha256": self.audio_sha, "history_policy": {"mode": "past_only"}},
            "segments": [{"output": {"text": "Учусь.", "complete": True}}],
        }

    def _validate(self, result):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result-1.json"
            path.write_text(json.dumps(result, ensure_ascii=False))
            return validate_result(path, result, self.manifest, self.calls, self.planned)

    def test_complete_result_keeps_history_mode_and_audio_provenance(self):
        item = self._validate(self.result)
        fields = task_fields(item)
        self.assertEqual(fields["qwen_history_status"], "complete")
        self.assertEqual(fields["qwen_history_text"], "Учусь.")
        self.assertEqual(fields["qwen_history_mode"], "self_transcripts_and_agent")
        self.assertEqual(fields["qwen_history_audio_sha256"], self.audio_sha)

    def test_truncated_result_is_explicitly_partial_not_complete(self):
        partial = dict(self.result, status="truncated")
        item = self._validate(partial)
        fields = task_fields(item)
        self.assertEqual(fields["qwen_history_status"], "truncated")
        self.assertTrue(fields["qwen_history_display"].startswith("ЧАСТИЧНЫЙ ВЫВОД:"))
        self.assertNotEqual(fields["qwen_history_status"], "complete")

    def test_incomplete_segment_cannot_claim_complete(self):
        broken = json.loads(json.dumps(self.result))
        broken["segments"][0]["output"]["complete"] = False
        with self.assertRaisesRegex(ValueError, "incomplete segments"):
            self._validate(broken)

    def test_audio_identity_mismatch_is_rejected(self):
        broken = dict(self.result, audio_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self._validate(broken)

    def test_wrong_runner_version_cannot_overwrite_selected_recipe(self):
        self.manifest["runner_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "Runner version"):
            self._validate(self.result)

    def test_per_call_self_only_policy_overrides_generic_manifest_mode(self):
        call = dict(self.call, history_policy={
            "selection": "Self-only prior completed Qwen transcripts; no original agent/client text",
            "original_agent_history_held": True,
        })
        manifest = {"history_mode": "self_transcripts_and_agent", "calls": [call]}
        item_calls = {("test", "media-1"): call}
        result = json.loads(json.dumps(self.result))
        result["provenance"]["history_policy"] = call["history_policy"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result-1.json"
            path.write_text(json.dumps(result, ensure_ascii=False))
            item = validate_result(path, result, manifest, item_calls, self.planned)
        fields = task_fields(item)
        self.assertEqual(fields["qwen_history_mode"], "self_transcripts_only")
        self.assertIn("Только предыдущие расшифровки Qwen", fields["qwen_history_mode_display"])

    def test_import_patches_only_task_data_and_preserves_human_work(self):
        item = self._validate(self.result)
        existing = {
            "id": 44,
            "data": {"source_environment": "test", "media_id": "media-1", "audio_sha256": self.audio_sha},
            "annotations": [{"id": 8, "result": [{"value": "human"}], "was_cancelled": False}],
            "drafts": [{"id": 9, "result": [{"value": "draft"}], "lead_time": 3.2}],
        }
        updated = json.loads(json.dumps(existing))
        updated["data"].update(task_fields(item))

        def fake_request(api_url, token, path, method="GET", payload=None, cookie_header=None):
            if method == "GET" and path == "/api/tasks/44":
                return existing if not hasattr(fake_request, "patched") else updated
            if method == "PATCH":
                self.assertEqual(set(payload), {"data"})
                updated["data"] = payload["data"]
                fake_request.patched = True
                return None
            self.fail(f"Unexpected request: {method} {path}")

        summary = {"id": 44}
        with patch("lalm.import_qwen_history_results.request_json", side_effect=fake_request):
            self.assertEqual(import_item("https://ls", "secret-unused", summary, item), "imported")
        self.assertEqual(existing["annotations"], updated["annotations"])
        self.assertEqual(existing["drafts"], updated["drafts"])

    def test_same_audio_complete_candidate_is_archived_before_explicit_supersession(self):
        self.check_supersession("self_transcripts_and_agent")

    def test_v2_recipe_migration_archives_old_output_and_preserves_labels(self):
        self.check_supersession("reference_history_v2")

    def test_v3_recipe_migration_archives_old_output_and_preserves_labels(self):
        self.check_supersession("reference_history_v3")

    def test_failed_new_recipe_is_visible_instead_of_stale_old_success(self):
        self.check_supersession("reference_history_v3", "truncated")

    def check_supersession(self, mode, status="complete"):
        item = self._validate(self.result)
        item["effective_history_mode"] = mode
        item["recipe"]["history_mode"] = mode
        item["status"] = status
        previous_sha = "b" * 64
        existing = {
            "id": 44,
            "data": {
                "source_environment": "test", "media_id": "media-1", "audio_sha256": self.audio_sha,
                "qwen_history_status": "complete", "qwen_history_text": "Earlier candidate.",
                "qwen_history_mode": "self_transcripts_and_agent", "qwen_history_call_id": "call-1",
                "qwen_history_audio_sha256": self.audio_sha,
                "qwen_history_checkpoint_sha256": CHECKPOINT_SHA256,
                "qwen_history_result_key": "canary-result", "qwen_history_result_sha256": previous_sha,
                "qwen_history_provenance": {"history_mode": "self_transcripts_and_agent"},
            },
            "annotations": [{"id": 8, "result": []}], "drafts": [{"id": 9, "result": []}],
        }
        updated = json.loads(json.dumps(existing))

        def fake_request(api_url, token, path, method="GET", payload=None, cookie_header=None):
            if method == "GET" and path == "/api/tasks/44":
                return existing if not hasattr(fake_request, "patched") else updated
            if method == "PATCH":
                updated["data"] = payload["data"]
                fake_request.patched = True
                return None
            self.fail(f"Unexpected request: {method} {path}")

        with patch("lalm.import_qwen_history_results.request_json", side_effect=fake_request):
            self.assertEqual(import_item("https://ls", "unused", {"id": 44}, item), "conflict")
            self.assertEqual(import_item("https://ls", "unused", {"id": 44}, item, supersede_complete=True), "imported")
        archive = updated["data"]["qwen_history_previous_candidates"]
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive[0]["result_sha256"], previous_sha)
        self.assertEqual(archive[0]["text"], "Earlier candidate.")
        self.assertEqual(updated["annotations"], existing["annotations"])
        self.assertEqual(updated["drafts"], existing["drafts"])
        self.assertEqual(updated["data"]["qwen_history_status"], status)


if __name__ == "__main__":
    unittest.main()
