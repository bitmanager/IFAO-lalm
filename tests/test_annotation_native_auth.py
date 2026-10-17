import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lalm.annotation.native_auth import repair_trailing_auth


class NativeAuthTests(unittest.TestCase):
    def auth(self):
        return {'token': {'access_token': 'local-access', 'refresh_token': 'local-refresh',
                         'token_type': 'Bearer', 'expiry': '2026-10-14T14:00:00Z'},
                'auth_method': 'consumer', 'id_token': 'header.payload.signature'}

    def test_repair_preserves_object_and_private_original(self):
        for suffix in ('}', 'old-signature-fragment'*16+'"}'):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'auth'; original=json.dumps(self.auth())+suffix;path.write_text(original)
                self.assertTrue(repair_trailing_auth(path))
                self.assertEqual(json.loads(path.read_text()),self.auth())
                backup=Path(str(path)+'.trailing-backup')
                self.assertEqual(backup.read_text(), original)
                self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertFalse(repair_trailing_auth(path))

    def test_ambiguous_second_object_is_not_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'auth';original=json.dumps(self.auth())+'{"other":"object"}';path.write_text(original)
            with self.assertRaises(ValueError):repair_trailing_auth(path)
            self.assertEqual(path.read_text(),original)

    def test_incomplete_object_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'auth';path.write_text('{"token":')
            with self.assertRaises(ValueError):repair_trailing_auth(path)
            self.assertEqual(path.read_text(),'{"token":')


if __name__=='__main__':unittest.main()
