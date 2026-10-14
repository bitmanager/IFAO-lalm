"""Move one annotation account to an existing, reachable VPN proxy service.

Run on dev-1 after the target proxy has passed an account-local audio probe.
The generated Compose file is private runtime state; this command keeps a
timestamped backup before changing only the selected account's route.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def routed_compose(compose, account, proxy, network):
    services = compose['services']
    if account not in services or proxy not in services or network not in compose['networks']:
        raise ValueError('account_proxy_or_network_missing')
    if network not in services[proxy].get('networks', []):
        raise ValueError('proxy_cannot_reach_target_network')
    for dependency in ('postgres', 's3-relay'):
        if network not in services[dependency].get('networks', []):
            raise ValueError(f'{dependency}_cannot_reach_target_network')
    target = services[account]
    if not target.get('environment', {}).get('ACCOUNT_ID') == account:
        raise ValueError('not_an_annotation_account')
    route = f'http://{proxy}:1080'
    target['environment']['HTTPS_PROXY'] = route
    target['environment']['HTTP_PROXY'] = route
    target['networks'] = [network]
    target['depends_on'] = {name: value for name, value in target['depends_on'].items()
                            if not name.endswith('-proxy')}
    target['depends_on'][proxy] = {'condition': 'service_started'}
    profiles = services[proxy].setdefault('profiles', [])
    if account not in profiles:
        profiles.append(account)
    return compose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compose', type=Path, default=Path(__file__).parent / 'compose.generated.json')
    parser.add_argument('--account', required=True)
    parser.add_argument('--proxy', required=True)
    parser.add_argument('--network', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    before = args.compose.read_bytes()
    compose = routed_compose(json.loads(before), args.account, args.proxy, args.network)
    after = (json.dumps(compose, indent=2) + '\n').encode()
    if before == after:
        print('route_already_set', args.account, args.proxy, args.network)
        return
    if not args.apply:
        print('route_ready', args.account, args.proxy, args.network)
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = args.compose.with_name(args.compose.name + '.route-backup-' + stamp)
    with backup.open('xb') as handle:
        handle.write(before)
    backup.chmod(0o600)
    fd, name = tempfile.mkstemp(prefix='.compose-route-', dir=args.compose.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(after)
        os.chmod(name, 0o600)
        os.replace(name, args.compose)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    print('route_applied', args.account, args.proxy, args.network, 'backup', backup.name)


if __name__ == '__main__':
    main()
