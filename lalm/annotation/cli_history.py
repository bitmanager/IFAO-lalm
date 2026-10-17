"""Archive CLI conversation state after pausing and draining one worker.

Run inside that worker container. No files are deleted and no model is called.
OAuth, settings, installation identity and model caches remain in place.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

from lalm.annotation.store import connect


HISTORY = {'annotations', 'brain', 'conversations', 'implicit', 'log', 'presence',
           'cli.log', 'history.jsonl', 'conversation_summaries.db',
           'conversation_summaries.db-wal', 'conversation_summaries.db-shm',
           'jetbox_summaries_proto.pb'}
PROFILE = ('antigravity-oauth-token', 'settings.json', 'installation_id', 'jetski_state.pbtxt')


def agy_active():
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / 'comm').read_text().strip() == 'agy':
                return True
        except (FileNotFoundError, ProcessLookupError):
            pass
    return False


def archive(expected_hold, apply=False):
    account = os.environ['ACCOUNT_ID']
    root = Path.home() / '.gemini' / 'antigravity-cli'
    with connect() as conn:
        row = conn.execute('SELECT next_request_at,now() AS now FROM accounts '
                           'WHERE id=%s FOR UPDATE', (account,)).fetchone()
        if (not row or row['next_request_at'] != expected_hold
                or (row['next_request_at'] - row['now']).total_seconds() < 30):
            raise RuntimeError('Worker must have the exact unexpired maintenance hold')
        if conn.execute("SELECT count(*) AS n FROM jobs WHERE account_id=%s AND state='RUNNING'",
                        (account,)).fetchone()['n']:
            raise RuntimeError('Wait for running jobs; never interrupt inference')
        if agy_active():
            raise RuntimeError('AGY process is still alive; do not rotate its databases')
        paths = [p for p in root.iterdir() if p.name in HISTORY
                 or p.match('jetbox_summaries_proto.pb.*.tmp')]
        report = {'account': account, 'entries': len(paths), 'applied': apply}
        if not apply:
            return report
        before = {name: hashlib.sha256((root/name).read_bytes()).digest()
                  for name in PROFILE if (root/name).exists()}
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        backup = root.parent / 'annotation-history-backups' / stamp
        backup.parent.mkdir(exist_ok=True, mode=0o700)
        backup.parent.chmod(0o700)
        backup.mkdir(mode=0o700)
        moved = []
        try:
            for path in paths:
                path.rename(backup/path.name)
                moved.append(path)
            after = {name: hashlib.sha256((root/name).read_bytes()).digest() for name in before}
            if after != before:
                raise RuntimeError('Profile changed during history archive')
        except Exception:
            for path in reversed(moved):
                (backup/path.name).rename(path)
            raise
        report.update(backup=str(backup), profile_unchanged=True)
        (backup/'archive-manifest.json').write_text(json.dumps(report, sort_keys=True))
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hold-until', required=True, type=datetime.fromisoformat)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(archive(args.hold_until, args.apply), sort_keys=True))
