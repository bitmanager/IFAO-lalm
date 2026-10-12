import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "infra/mlflow/start.sh"


class StartupTests(unittest.TestCase):
    def test_browser_origin_allowlist_is_passed_without_disabling_security(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exe = root / "venv/bin/mlflow"
            exe.parent.mkdir(parents=True)
            exe.write_text(
                "#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n"
            )
            exe.chmod(0o755)
            env = dict(
                os.environ,
                MLFLOW_ROOT=directory,
                MLFLOW_PORT="5009",
                MLFLOW_ALLOWED_HOSTS="dashboard.example:8445",
                MLFLOW_CORS_ALLOWED_ORIGINS="http://dashboard.example:8445",
            )
            args = json.loads(
                subprocess.check_output(["bash", str(SCRIPT)], env=env, text=True)
            )
            self.assertEqual(
                args[args.index("--allowed-hosts") + 1], "dashboard.example:8445"
            )
            self.assertEqual(
                args[args.index("--cors-allowed-origins") + 1],
                "http://dashboard.example:8445",
            )
            self.assertNotIn("--disable-security-middleware", args)
            env["MLFLOW_EXECUTABLE"] = str(exe)
            self.assertEqual(
                json.loads(
                    subprocess.check_output(["bash", str(SCRIPT)], env=env, text=True)
                ),
                args,
            )

            env.pop("MLFLOW_CORS_ALLOWED_ORIGINS")
            env.pop("MLFLOW_ALLOWED_HOSTS")
            args = json.loads(
                subprocess.check_output(["bash", str(SCRIPT)], env=env, text=True)
            )
            self.assertEqual(
                args[args.index("--cors-allowed-origins") + 1],
                "http://localhost:5009,http://127.0.0.1:5009",
            )


if __name__ == "__main__":
    unittest.main()
