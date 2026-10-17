"""Verify denial, preserved owner hooks and atomic global installation."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from lalm.annotation.cli_hooks import install_cli_hooks, install_global_cli_hooks, POLICY, POLICY_SHA256


class CliHookTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        self.enterContext(patch('lalm.annotation.cli_hooks.Path.home', return_value=Path(self.home.name)))

    def test_global_hook_is_private_idempotent_and_covers_other_slots(self):
        result = install_global_cli_hooks()
        path = Path(self.home.name) / '.gemini/config/hooks.json'
        self.assertTrue(result['changed'])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        original = path.read_bytes()
        stamp = path.stat().st_mtime_ns
        self.assertFalse(install_global_cli_hooks()['changed'])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(install_cli_hooks(temp, 'agy-06'), POLICY_SHA256)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_global_merge_preserves_existing_hooks_and_exact_backup(self):
        root = Path(self.home.name) / 'config'
        root.mkdir()
        old = b'{"existing-owner-hook":{"enabled":false}}\n'
        (root / 'hooks.json').write_bytes(old)
        result = install_global_cli_hooks(root)
        value = json.loads((root / 'hooks.json').read_bytes())
        self.assertEqual(value['existing-owner-hook'], {'enabled': False})
        self.assertEqual(value['audio-transcription-no-web'], POLICY['audio-transcription-no-web'])
        self.assertEqual(Path(result['backup']).read_bytes(), old)
        self.assertEqual(Path(result['backup']).stat().st_mode & 0o777, 0o600)

    def test_conflicting_and_invalid_global_configuration_is_not_replaced(self):
        root = Path(self.home.name) / 'config'
        root.mkdir()
        for old in (b'{"audio-transcription-no-web":{"enabled":false}}', b'[]', b'not json'):
            (root / 'hooks.json').write_bytes(old)
            with self.assertRaises(ValueError):
                install_global_cli_hooks(root)
            self.assertEqual((root / 'hooks.json').read_bytes(), old)

    def test_failed_atomic_replace_keeps_original_configuration(self):
        root = Path(self.home.name) / 'config'
        root.mkdir()
        old = b'{"owner":{}}'
        (root / 'hooks.json').write_bytes(old)
        with patch('lalm.annotation.cli_hooks.os.replace', side_effect=OSError):
            with self.assertRaises(OSError):
                install_global_cli_hooks(root)
        self.assertEqual((root / 'hooks.json').read_bytes(), old)
        self.assertEqual(list(root.glob('.hooks-*')), [])

    def test_concurrent_global_config_creation_is_preserved(self):
        root = Path(self.home.name) / 'config'
        root.mkdir()
        def concurrent_create(source, destination):
            Path(destination).write_text('{"concurrent-owner":{}}')
            raise FileExistsError()
        with patch('lalm.annotation.cli_hooks.os.link', side_effect=concurrent_create):
            with self.assertRaises(FileExistsError):
                install_global_cli_hooks(root)
        self.assertEqual(json.loads((root / 'hooks.json').read_bytes()), {'concurrent-owner': {}})
        self.assertEqual(list(root.glob('.hooks-*')), [])

    def test_expanded_slot_receives_same_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(install_cli_hooks(temp, 'agy-25'), POLICY_SHA256)
            self.assertEqual(hashlib.sha256(
                (Path(temp) / '.agents/hooks.json').read_bytes()).hexdigest(), POLICY_SHA256)

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
