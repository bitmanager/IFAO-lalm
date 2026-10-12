"""Generate Compose account slots, isolated egress and local secrets; no login copying."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def build(accounts, agy_path):
    ids = [a['id'] for a in accounts]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'agy-[a-z0-9-]+', x) for x in ids):
        raise ValueError('unique agy-* account IDs required')
    env = {'ANNOTATION_DATABASE_URL': '${ANNOTATION_DATABASE_URL}', 'S3_ENDPOINT': 'http://minio:9000',
           'S3_ACCESS_KEY': '${S3_ACCESS_KEY}', 'S3_SECRET_KEY': '${S3_SECRET_KEY}', 'S3_BUCKET': 'speech-data'}
    deps = {'postgres': {'condition': 'service_healthy'}, 'minio': {'condition': 'service_started'}}
    image = {'image': 'speech-annotation:local', 'build': {'context': str(REPO),
             'dockerfile': 'infra/annotation/Dockerfile'}}
    common = {**image, 'environment': env, 'networks': ['backend'], 'depends_on': deps}
    afenv = {**env, 'AIRFLOW__CORE__EXECUTOR': 'LocalExecutor',
             'AIRFLOW__DATABASE__SQL_ALCHEMY_CONN': '${AIRFLOW_DATABASE_URL}',
             'AIRFLOW__CORE__LOAD_EXAMPLES': 'false',
             'AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION': 'true',
             'AIRFLOW__CORE__EXECUTION_API_SERVER_URL': 'http://airflow-api:8080/execution/',
             'AIRFLOW__API_AUTH__JWT_SECRET': '${AIRFLOW_JWT_SECRET}',
             'AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS': 'admin:admin',
             'AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE': '/opt/airflow/auth/passwords.json',
             'AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK': 'true'}
    af = {'image': 'speech-airflow:local', 'build': {'context': str(REPO),
          'dockerfile': 'infra/annotation/Dockerfile.airflow'}, 'environment': afenv,
          'networks': ['backend'], 'volumes': ['airflow-logs:/opt/airflow/logs',
           './runtime/airflow-auth:/opt/airflow/auth'], 'restart': 'unless-stopped'}
    services = {
      'postgres': {'image': 'postgres:16-bookworm', 'environment': {'POSTGRES_USER': 'speech',
        'POSTGRES_PASSWORD': '${POSTGRES_PASSWORD}', 'POSTGRES_DB': 'speech'},
        'networks': ['backend'], 'volumes': ['postgres-data:/var/lib/postgresql/data',
        './init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro'],
        'healthcheck': {'test': ['CMD-SHELL', 'pg_isready -U speech -d speech'], 'interval': '5s', 'retries': 20},
        'restart': 'unless-stopped'},
      'minio': {'image': 'speech-minio:2025-09-07', 'build': {'context': str(REPO),
        'dockerfile': 'infra/annotation/Dockerfile.minio'},
        'command': ['server', '/data', '--console-address', ':9001'],
        'environment': {'MINIO_ROOT_USER': '${S3_ACCESS_KEY}', 'MINIO_ROOT_PASSWORD': '${S3_SECRET_KEY}'},
        'networks': ['backend'], 'volumes': ['s3-data:/data'], 'restart': 'unless-stopped'},
      'init': {**common, 'command': ['python', '-m', 'lalm.annotation.cli', 'init'], 'restart': 'on-failure'},
      'control': {**common, 'profiles': ['tools'], 'entrypoint': ['python', '-m', 'lalm.annotation.cli']},
      'airflow-init': {**af, 'command': ['db', 'migrate'], 'restart': 'no', 'depends_on': deps},
    }
    afdeps = {**deps, 'airflow-init': {'condition': 'service_completed_successfully'},
              'init': {'condition': 'service_completed_successfully'}}
    for name, command in [('api', 'api-server'), ('scheduler', 'scheduler'), ('dag-processor', 'dag-processor')]:
        services['airflow-' + name] = {**af, 'command': command, 'depends_on': afdeps}
    services['airflow-api']['ports'] = ['127.0.0.1:18080:8080']
    services['airflow-api']['networks'] = ['backend', 'ui']
    networks = {'backend': {'internal': True}, 'egress': {}, 'ui': {}}
    volumes = {x: {} for x in ['postgres-data', 's3-data', 'airflow-logs']}
    for account in accounts:
        aid = account['id']
        if not isinstance(account['concurrency'], int) or not 1 <= account['concurrency'] <= 64:
            raise ValueError('invalid concurrency')
        private = aid + '-private'
        networks[private] = {'internal': True}
        services['postgres']['networks'].append(private)
        services['minio']['networks'].append(private)
        services[aid + '-proxy'] = {'image': '${XRAY_IMAGE:-ghcr.io/xtls/xray-core:25.3.6}',
            'profiles': [aid], 'command': ['run', '-config', '/etc/xray/config.json'],
            'volumes': [f'./runtime/{aid}/xray.json:/etc/xray/config.json:ro'],
            'networks': [private, 'egress'], 'restart': 'unless-stopped',
            'cap_drop': ['ALL'], 'security_opt': ['no-new-privileges:true'], 'read_only': True}
        volumes[aid + '-home'] = {}
        services[aid] = {**image, 'profiles': [aid], 'restart': 'unless-stopped',
            'environment': {**env, 'ACCOUNT_ID': aid, 'WORKER_CONCURRENCY': str(account['concurrency']),
                'HTTPS_PROXY': f'http://{aid}-proxy:1080', 'HTTP_PROXY': f'http://{aid}-proxy:1080',
                'NO_PROXY': 'postgres,minio,127.0.0.1,localhost'},
            'depends_on': {**deps, aid + '-proxy': {'condition': 'service_started'}},
            'networks': [private], 'volumes': [f'{agy_path}:/opt/bin/agy:ro',
                                              f'{aid}-home:/home/worker'],
            'tmpfs': ['/work:uid=10001,gid=10001,mode=0700,size=1g', '/tmp:size=256m'],
            'read_only': True, 'cap_drop': ['ALL'], 'security_opt': ['no-new-privileges:true'],
            'pids_limit': 512, 'mem_limit': '4g', 'cpus': 2, 'stop_grace_period': '360s'}
    return {'name': 'speech-data', 'services': services, 'networks': networks, 'volumes': volumes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accounts', type=Path, default=ROOT / 'accounts.example.json')
    parser.add_argument('--agy', type=Path, default=Path('/home/alexesn/.local/bin/agy'))
    args = parser.parse_args()
    accounts = json.loads(args.accounts.read_text())['accounts']
    compose = build(accounts, str(args.agy.resolve()))
    binary_hash = hashlib.sha256(args.agy.read_bytes()).hexdigest()
    for account in accounts:
        compose['services'][account['id']]['environment']['AGY_EXPECTED_SHA256'] = binary_hash
    os.umask(0o077)
    (ROOT / 'runtime').mkdir(exist_ok=True)
    for account in accounts:
        directory = ROOT / 'runtime' / account['id']
        directory.mkdir(exist_ok=True)
        config = directory / 'xray.json'
        if not config.exists():
            config.write_text(json.dumps({'log': {'loglevel': 'warning'},
              'inbounds': [{'listen': '0.0.0.0', 'port': 1080, 'protocol': 'http'}],
              'outbounds': [{'protocol': 'blackhole', 'tag': 'not-configured'}]}, indent=2))
    envpath = ROOT / '.env'
    if not envpath.exists():
        password = secrets.token_hex(24)
        envpath.write_text(f'POSTGRES_PASSWORD={password}\n'
          f'ANNOTATION_DATABASE_URL=postgresql://speech:{password}@postgres/speech\n'
          f'AIRFLOW_DATABASE_URL=postgresql+psycopg2://speech:{password}@postgres/airflow\n'
          f'S3_ACCESS_KEY=speech-admin\nS3_SECRET_KEY={secrets.token_hex(24)}\n'
          f'AIRFLOW_JWT_SECRET={secrets.token_hex(32)}\n')
    auth = ROOT / 'runtime/airflow-auth'
    auth.mkdir(exist_ok=True)
    # Airflow SimpleAuthManager opens the password file a+ even when preseeded.
    # Outer runtime directory is 0700; only this subdirectory is mounted in Airflow.
    auth.chmod(0o777)
    if not (auth / 'passwords.json').exists():
        (auth / 'passwords.json').write_text(json.dumps({'admin': secrets.token_urlsafe(24)}))
    (auth / 'passwords.json').chmod(0o666)
    (ROOT / 'compose.generated.json').write_text(json.dumps(compose, indent=2))
    print('Rendered account slots:', ', '.join(a['id'] for a in accounts))
    print('Account profiles are opt-in. No host login/session was copied.')


if __name__ == '__main__':
    main()
