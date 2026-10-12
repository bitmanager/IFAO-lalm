import argparse
import json

from .pipeline import ingest, publish, status
from .store import connect, initialize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['init', 'ingest', 'publish', 'status', 'account'])
    parser.add_argument('--account-id')
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--min-interval', type=float, default=1)
    parser.add_argument('--enable', action='store_true')
    args = parser.parse_args()
    if args.action == 'init':
        initialize()
    elif args.action == 'account':
        if not args.account_id or not 1 <= args.concurrency <= 64 or args.min_interval < 0:
            parser.error('valid account-id, concurrency and min-interval required')
        with connect() as conn:
            conn.execute('INSERT INTO accounts(id,enabled,concurrency,min_interval_seconds) VALUES (%s,%s,%s,%s) '
                         'ON CONFLICT(id) DO UPDATE SET enabled=excluded.enabled,concurrency=excluded.concurrency,'
                         'min_interval_seconds=excluded.min_interval_seconds,blocked_reason=NULL',
                         (args.account_id, args.enable, args.concurrency, args.min_interval))
    else:
        print(json.dumps({'ingest': ingest, 'publish': publish, 'status': status}[args.action](), default=str))


if __name__ == '__main__':
    main()
