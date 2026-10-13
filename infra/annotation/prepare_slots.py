"""Add empty AGY containers to an existing Compose config, sharing VPN egress.

Does not start services, enable accounts, or copy authentication volumes.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone


def add_slots(compose, mapping):
    result = copy.deepcopy(compose)
    for target, source in mapping.items():
        if not all(re.fullmatch(r'agy-[0-9]{2}', x) for x in (target, source)):
            raise ValueError('Expected agy-NN IDs')
        if target in result['services']:
            raise ValueError(f'{target} already exists; refusing replacement')
        worker = copy.deepcopy(compose['services'][source])
        old_home = source + '-home:/home/worker'
        if old_home not in worker['volumes']:
            raise ValueError('Unexpected home mount')
        worker['volumes'] = [target+'-home:/home/worker' if v == old_home else v for v in worker['volumes']]
        worker['profiles'] = [target]
        worker['environment'].update(ACCOUNT_ID=target, WORKER_CONCURRENCY='20')
        # Retain donor private network/proxy, but never donor account home.
        proxy = source + '-proxy'
        profiles = result['services'][proxy].setdefault('profiles', [])
        if target not in profiles:
            profiles.append(target)
        result['services'][target] = worker
        result['volumes'][target+'-home'] = {}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compose', type=Path, required=True)
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    mapping = json.loads(args.mapping.read_text())
    result = add_slots(json.loads(args.compose.read_text()), mapping)
    if args.apply:
        os.umask(0o077)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        shutil.copy2(args.compose, str(args.compose)+'.backup-'+stamp)
        temp = args.compose.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2)+'\n')
        temp.chmod(0o600)
        temp.replace(args.compose)
    print(json.dumps({'applied':args.apply,'slot_to_vpn_owner':mapping}))
