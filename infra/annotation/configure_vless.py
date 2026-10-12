"""Read a VLESS link from a secret file; never place it in argv/logs."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
import uuid


def convert(link):
    url = urlsplit(link.strip())
    if url.scheme != 'vless' or not url.hostname or not url.port or not url.username:
        raise ValueError('Expected a complete VLESS URI')
    user = str(uuid.UUID(unquote(url.username)))
    params = parse_qs(url.query, keep_blank_values=True)
    if any(len(v) != 1 for v in params.values()):
        raise ValueError('Duplicate VLESS parameters')
    p = {k: v[0] for k, v in params.items()}
    supported = {'type', 'security', 'encryption', 'flow', 'sni', 'fp', 'pbk', 'sid',
                 'spx', 'alpn', 'host', 'path', 'serviceName', 'headerType'}
    if set(p) - supported:
        raise ValueError('Unsupported VLESS parameters; supply a validated Xray config instead')
    transport, security = p.get('type', 'tcp'), p.get('security', 'tls')
    if transport not in ('tcp', 'ws', 'grpc') or security not in ('tls', 'reality'):
        raise ValueError('Supported: TCP/WS/gRPC with TLS, TCP with REALITY')
    if p.get('encryption', 'none') != 'none' or p.get('headerType', 'none') != 'none':
        raise ValueError('Unsupported encryption/header mode')
    if security == 'reality' and transport != 'tcp':
        raise ValueError('REALITY requires TCP in this converter')
    flow = p.get('flow', '')
    if flow not in ('', 'xtls-rprx-vision') or flow and transport != 'tcp':
        raise ValueError('Unsupported flow')
    settings = {'serverName': p.get('sni', url.hostname)}
    if security == 'reality':
        if not p.get('pbk') or not p.get('sni'):
            raise ValueError('REALITY public key and server name required')
        settings.update(publicKey=p['pbk'], shortId=p.get('sid', ''),
                        fingerprint=p.get('fp', 'chrome'), spiderX=p.get('spx', '/'))
    else:
        settings['allowInsecure'] = False
        if p.get('fp'):
            settings['fingerprint'] = p['fp']
        if p.get('alpn'):
            settings['alpn'] = p['alpn'].split(',')
    stream = {'network': transport, 'security': security, security + 'Settings': settings}
    if transport == 'ws':
        stream['wsSettings'] = {'path': p.get('path', '/'), 'headers': {'Host': p.get('host', url.hostname)}}
    if transport == 'grpc':
        stream['grpcSettings'] = {'serviceName': p.get('serviceName', '')}
    return {'log': {'loglevel': 'warning'},
            'inbounds': [{'listen': '0.0.0.0', 'port': 1080, 'protocol': 'http'}],
            'outbounds': [{'tag': 'vless', 'protocol': 'vless', 'settings': {'vnext': [{
                'address': url.hostname, 'port': url.port,
                'users': [{'id': user, 'encryption': 'none', 'flow': flow}]}]},
                'streamSettings': stream}]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link-file', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    value = convert(args.link_file.read_text())
    os.umask(0o077)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2))
    args.output.chmod(0o600)
    print('VLESS configuration written; validate with the pinned Xray image before enabling the account.')


if __name__ == '__main__':
    main()
