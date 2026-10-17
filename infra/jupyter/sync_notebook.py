"""Explicit two-way sync of the single notebook; no outputs or credentials in Git."""
import argparse
import json
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
LOCAL = REPO / 'notebooks/speech_data_progress.ipynb'
REMOTE = '/home/analyst/speech_data_progress.ipynb'
SSH = str(Path.home() / '.local/bin/ssh-gpu')

def clean(n):
    n = json.loads(json.dumps(n))
    for cell in n['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        cell.get('metadata', {}).pop('execution', None)
    return n

def sources(n):
    return [(c['cell_type'], ''.join(c['source']), c.get('attachments', {})) for c in n['cells']]

def remote(code):
    return subprocess.check_output([SSH, 'dev-1',
        'docker exec -i speech-data-jupyterhub-1 python -'], input=code.encode()).decode()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ('check', 'pull', 'push'):
        group.add_argument('--'+mode, action='store_true')
    args = parser.parse_args()
    raw = remote(f'from pathlib import Path\nprint(Path({REMOTE!r}).read_text())\n')
    live = json.loads(raw)
    local = json.loads(LOCAL.read_text())
    has_outputs = any(c.get('outputs') or c.get('execution_count') is not None for c in local['cells'])
    equal = sources(local) == sources(live)
    if args.check:
        print('Sources: '+('in sync' if equal else 'DIFFERENT — inspect browser edits before deployment'))
        print('Git outputs: '+('MUST CLEAR' if has_outputs else 'clean'))
        return 0 if equal and not has_outputs else 1
    stamp = time.time_ns()
    if args.pull:
        backup = Path.home()/'.local/share/speech-data/notebook-backups'
        backup.mkdir(parents=True, exist_ok=True, mode=0o700)
        dest = backup/f'{stamp}.ipynb'
        dest.write_bytes(LOCAL.read_bytes()); dest.chmod(0o600)
        LOCAL.write_text(json.dumps(clean(live), ensure_ascii=False, indent=1)+'\n')
        print('Pulled browser sources without outputs. Review git diff before committing.')
    else:
        content = json.dumps(clean(local), ensure_ascii=False, indent=1)+'\n'
        # Detect browser edits made after the read; preserve all existing outputs in backup.
        expected = sources(live)
        code = ('import json,os,shutil\nfrom pathlib import Path\n'
                f'p=Path({REMOTE!r})\nn=json.loads(p.read_text())\n'
                f'assert [(c["cell_type"], "".join(c["source"]), c.get("attachments", {{}})) for c in n["cells"]] == {expected!r}, "Concurrent browser edit; retry sync"\n'
                'backup=p.parent/".notebook-archive"\nbackup.mkdir(exist_ok=True)\n'
                f'shutil.copy2(p,backup/"progress-{stamp}.ipynb")\n'
                f'tmp=p.with_suffix(".sync-tmp")\ntmp.write_text({content!r})\nos.chown(tmp,10002,10002)\ntmp.replace(p)\n')
        remote(code)
        print('Deployed notebook sources; previous server copy backed up in .notebook-archive.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
