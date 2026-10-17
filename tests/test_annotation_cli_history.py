import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from lalm.annotation.cli_history import archive


class CliHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.home = Path(self.temp)
        self.root = self.home/'.gemini/antigravity-cli'
        self.root.mkdir(parents=True)
        self.profile = {'antigravity-oauth-token': b'test-auth', 'settings.json': b'{}'}
        self.history = {'conversation_summaries.db': b'old-index', 'jetbox_summaries_proto.pb': b'old-summary'}
        for name, data in {**self.profile, **self.history}.items():
            (self.root/name).write_bytes(data)
        self.now = datetime(2026, 10, 15, tzinfo=timezone.utc)
        self.hold = self.now + timedelta(minutes=15)
        self.connection = Mock()
        self.connection.execute.side_effect = [
            Mock(fetchone=Mock(return_value={'next_request_at': self.hold, 'now': self.now})),
            Mock(fetchone=Mock(return_value={'n': 0}))]
        self.enterContext(patch('lalm.annotation.cli_history.connect')).return_value.__enter__.return_value = self.connection
        self.enterContext(patch.object(Path, 'home', return_value=self.home))
        self.enterContext(patch.dict('os.environ', {'ACCOUNT_ID': 'agy-test'}))
        self.active = self.enterContext(patch('lalm.annotation.cli_history.agy_active', return_value=False))

    def assert_original_state(self):
        for name, data in {**self.profile, **self.history}.items():
            self.assertEqual((self.root/name).read_bytes(), data)

    def test_history_is_archived_and_profile_is_preserved(self):
        result = archive(self.hold, apply=True)
        backup = Path(result['backup'])
        self.assertTrue(result['profile_unchanged'])
        self.assertEqual(backup.stat().st_mode & 0o777, 0o700)
        for name, data in self.profile.items():
            self.assertEqual((self.root/name).read_bytes(), data)
            self.assertFalse((backup/name).exists())
        for name, data in self.history.items():
            self.assertEqual((backup/name).read_bytes(), data)
            self.assertFalse((self.root/name).exists())
        self.assertEqual(json.loads((backup/'archive-manifest.json').read_text()), result)

    def test_wrong_hold_does_not_touch_history(self):
        with self.assertRaises(RuntimeError):
            archive(self.hold + timedelta(seconds=1), apply=True)
        self.assert_original_state()

    def test_running_job_prevents_rotation(self):
        self.connection.execute.side_effect = [
            Mock(fetchone=Mock(return_value={'next_request_at': self.hold, 'now': self.now})),
            Mock(fetchone=Mock(return_value={'n': 1}))]
        with self.assertRaises(RuntimeError):
            archive(self.hold, apply=True)
        self.assert_original_state()

    def test_lingering_agy_prevents_rotation(self):
        self.active.return_value = True
        with self.assertRaises(RuntimeError):
            archive(self.hold, apply=True)
        self.assert_original_state()

    def test_partial_rename_failure_restores_history(self):
        rename = Path.rename
        calls = 0

        def fail_second(path, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('simulated rename failure')
            return rename(path, target)

        with patch.object(Path, 'rename', fail_second), self.assertRaises(OSError):
            archive(self.hold, apply=True)
        self.assert_original_state()


if __name__ == '__main__':
    unittest.main()
