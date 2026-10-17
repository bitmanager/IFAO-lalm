"""Stock AGY global hook, with the original workspace pilot as a fallback."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid


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


def install_global_cli_hooks(root=None):
    directory = Path(root) if root is not None else Path.home() / '.gemini/config'
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / 'hooks.json'
    before = path.read_bytes() if path.exists() else None
    current = json.loads(before) if before is not None else {}
    if not isinstance(current, dict):
        raise ValueError('Global hooks must be a JSON object')
    name, policy = next(iter(POLICY.items()))
    if name in current:
        if current[name] != policy:
            raise ValueError('Conflicting global audio hook; preserve and inspect')
        return {'changed': False, 'policy_sha256': POLICY_SHA256,
                'file_sha256': hashlib.sha256(before).hexdigest()}
    payload = (json.dumps(dict(current, **POLICY), sort_keys=True) + '\n').encode()
    backup = None
    if before is not None:
        backup = directory / ('hooks.before-no-web-' + uuid.uuid4().hex + '.json')
        with backup.open('xb') as stream:
            os.chmod(backup, 0o600)
            stream.write(before)
    with tempfile.NamedTemporaryFile(dir=directory, prefix='.hooks-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        if (path.read_bytes() if path.exists() else None) != before:
            raise RuntimeError('Global hooks changed during installation')
        if before is None:
            os.link(temporary, path)  # Never overwrite a concurrently created config.
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'changed': True, 'policy_sha256': POLICY_SHA256,
            'file_sha256': hashlib.sha256(payload).hexdigest(),
            'backup': str(backup) if backup else None}


def install_cli_hooks(workspace, account_id):
    path = Path.home() / '.gemini/config/hooks.json'
    if path.exists():
        value = json.loads(path.read_bytes())
        name, policy = next(iter(POLICY.items()))
        if isinstance(value, dict) and value.get(name) == policy:
            return POLICY_SHA256
    if account_id not in PILOT_ACCOUNTS:
        return None
    directory = Path(workspace) / '.agents'
    directory.mkdir(mode=0o700, exist_ok=True)
    (directory / 'hooks.json').write_bytes(POLICY_BYTES)
    return POLICY_SHA256


if __name__ == '__main__':
    print(json.dumps(install_global_cli_hooks(), sort_keys=True))
