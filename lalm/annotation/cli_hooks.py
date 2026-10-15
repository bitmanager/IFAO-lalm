"""Stock AGY workspace hook; initially limited to one monitored CLI slot."""
import hashlib
import json
from pathlib import Path


PILOT_ACCOUNTS = frozenset({'agy-07', 'agy-25'})
POLICY = {
    'audio-transcription-no-web': {'PreToolUse': [{
        'matcher': '^search_web$',
        'hooks': [{
            'type': 'command',
            'command': "printf '%s\\n' '{\"decision\":\"deny\",\"reason\":\"Transcription must use only the current audio; web search is disabled.\"}'",
            'timeout': 2,
        }],
    }]},
}
POLICY_BYTES = (json.dumps(POLICY, sort_keys=True) + '\n').encode()
POLICY_SHA256 = hashlib.sha256(POLICY_BYTES).hexdigest()


def install_cli_hooks(workspace, account_id):
    if account_id not in PILOT_ACCOUNTS:
        return None
    directory = Path(workspace) / '.agents'
    directory.mkdir(mode=0o700, exist_ok=True)
    (directory / 'hooks.json').write_bytes(POLICY_BYTES)
    return POLICY_SHA256
