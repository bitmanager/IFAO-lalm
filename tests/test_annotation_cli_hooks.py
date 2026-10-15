"""Verify the executable denial and the deliberately limited rollout."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from lalm.annotation.cli_hooks import install_cli_hooks, POLICY_SHA256


class CliHookTests(unittest.TestCase):
    def test_pilot_installs_executable_deny_and_reproducible_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(install_cli_hooks(temp, 'agy-07'), POLICY_SHA256)
            raw = (Path(temp) / '.agents/hooks.json').read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), POLICY_SHA256)
            entry = json.loads(raw)['audio-transcription-no-web']['PreToolUse'][0]
            import re
            self.assertTrue(re.fullmatch(entry['matcher'], 'search_web'))
            self.assertIsNone(re.fullmatch(entry['matcher'], 'view_file'))
            result = subprocess.run(entry['hooks'][0]['command'], shell=True,
                                    input='{}', capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)['decision'], 'deny')
            self.assertEqual(install_cli_hooks(temp, 'agy-07'), POLICY_SHA256)

    def test_other_accounts_do_not_receive_pilot_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(install_cli_hooks(temp, 'agy-06'))
            self.assertEqual(list(Path(temp).iterdir()), [])
