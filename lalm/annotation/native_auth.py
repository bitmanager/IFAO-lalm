"""Repair only trailing garbage after a complete native OAuth JSON object.

Credentials remain in their existing account home. No values are logged, changed,
refreshed or sent anywhere here; ambiguous/partial objects are rejected.
"""
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time


def repair_trailing_auth(path=None):
    path = Path(path) if path else Path.home()/'.gemini/antigravity-cli/antigravity-oauth-token'
    if not path.exists():
        return False
    lock = os.open(path.parent/'.auth-storage-repair.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for attempt in range(3):
            stat = path.stat()
            raw = path.read_text()
            try:
                json.loads(raw)
                return False
            except json.JSONDecodeError:
                try:
                    obj, end = json.JSONDecoder().raw_decode(raw)
                except json.JSONDecodeError:
                    if attempt < 2:
                        time.sleep(.02)
                        continue
                    raise ValueError('auth_storage_incomplete_object') from None
            tail = raw[end:]
            if (not tail.strip() or len(tail) > 4096 or '{' in tail or '[' in tail
                    or not isinstance(obj, dict) or set(obj) != {'token','auth_method','id_token'}
                    or obj['auth_method'] != 'consumer' or not isinstance(obj['token'], dict)
                    or set(obj['token']) != {'access_token','refresh_token','token_type','expiry'}
                    or any(not isinstance(v, str) or not v for v in obj['token'].values())
                    or obj['token']['token_type'] != 'Bearer'
                    or not isinstance(obj['id_token'], str) or obj['id_token'].count('.') != 2):
                raise ValueError('auth_storage_ambiguous_tail_or_shape')
            datetime.fromisoformat(obj['token']['expiry'].replace('Z', '+00:00'))
            current = path.stat()
            if (current.st_ino,current.st_mtime_ns,current.st_size) != (stat.st_ino,stat.st_mtime_ns,stat.st_size):
                continue
            # Preserve the observed original once, privately, before replacement.
            backup = path.with_name(path.name+'.trailing-backup')
            try:
                fd = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, 'w') as f:
                    f.write(raw); f.flush(); os.fsync(f.fileno())
            fd, tmp = tempfile.mkstemp(prefix='.oauth-repair-', dir=path.parent)
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(obj, f); f.flush(); os.fsync(f.fileno())
                current = path.stat()
                if (current.st_ino,current.st_mtime_ns,current.st_size) != (stat.st_ino,stat.st_mtime_ns,stat.st_size):
                    continue
                os.replace(tmp, path)
                return True
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
        raise ValueError('auth_storage_changed_during_repair')
    finally:
        os.close(lock)
