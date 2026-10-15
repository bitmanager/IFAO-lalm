"""Generate Compose account slots, isolated egress and local secrets; no login copying."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil

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
             'ANNOTATION_MODEL': '${ANNOTATION_MODEL:-gemini-3.8-flash-low}',
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
           './runtime/airflow-auth:/opt/airflow/auth',
           f'{REPO}/airflow/dags:/opt/airflow/dags:ro'], 'restart': 'unless-stopped'}
    services = {
      'postgres': {'image': 'postgres:17.10-alpine', 'environment': {'POSTGRES_USER': 'speech',
        'POSTGRES_PASSWORD': '${POSTGRES_PASSWORD}', 'POSTGRES_DB': 'speech'},
        'command': ['postgres', '-c', 'max_connections=512'],
        'networks': ['backend'], 'volumes': ['postgres-data:/var/lib/postgresql/data',
        './init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro'],
        'healthcheck': {'test': ['CMD-SHELL', 'pg_isready -U speech -d speech'], 'interval': '5s', 'retries': 20},
        'restart': 'unless-stopped'},
      'minio': {'image': 'speech-minio:2025-09-07', 'build': {'context': str(REPO),
        'dockerfile': 'infra/annotation/Dockerfile.minio'},
        'command': ['server', '/data', '--console-address', ':9001'],
        'environment': {'MINIO_ROOT_USER': '${S3_ACCESS_KEY}', 'MINIO_ROOT_PASSWORD': '${S3_SECRET_KEY}'},
        'networks': ['backend'], 'volumes': ['s3-data:/data'], 'restart': 'unless-stopped'},
      'init': {**common, 'image': 'speech-airflow:local',
          'build': {'context': str(REPO), 'dockerfile': 'infra/annotation/Dockerfile.airflow'},
          'entrypoint': ['python', '-m', 'lalm.annotation.cli'], 'command': ['init'], 'restart': 'on-failure'},
      'source-preparer': {**common, 'profiles': ['sources'],
          'environment': {**env, 'PREPARE_MAX_PENDING': '${PREPARE_MAX_PENDING:-6000}'},
          'command': ['python', '-m', 'lalm.annotation.prepare'],
          'restart': 'unless-stopped', 'mem_limit': '4g', 'cpus': 2,
          'stop_grace_period': '60s'},
      'control': {**common, 'image': 'speech-airflow:local',
          'build': {'context': str(REPO), 'dockerfile': 'infra/annotation/Dockerfile.airflow'},
          'profiles': ['tools'], 'entrypoint': ['python', '-m', 'lalm.annotation.cli']},
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
        models = account.get('models', ['gemini-3.8-flash-low'])
        if (not isinstance(models, list) or not models or
                any(m not in ('gemini-3.8-flash-low', 'gemini-4-argon-low') for m in models)):
            raise ValueError('invalid annotation models')
        if not isinstance(account['concurrency'], int) or not 1 <= account['concurrency'] <= 64:
            raise ValueError('invalid concurrency')
        private = aid + '-private'
        networks[private] = {'internal': True}
        services['postgres']['networks'].append(private)
        services['minio']['networks'].append(private)
        services[aid + '-proxy'] = {'image': '${XRAY_IMAGE:-ghcr.io/xtls/xray-core:25.3.6}',
            'profiles': [aid], 'user': f'{os.getuid()}:{os.getgid()}',
            'command': ['run', '-config', '/etc/xray/config.json'],
            'volumes': [f'./runtime/{aid}/xray.json:/etc/xray/config.json:ro'],
            'networks': [private, 'egress'], 'restart': 'unless-stopped',
            'cap_drop': ['ALL'], 'security_opt': ['no-new-privileges:true'], 'read_only': True}
        volumes[aid + '-home'] = {}
        services[aid] = {**image, 'profiles': [aid], 'restart': 'unless-stopped',
            'environment': {**env, 'ACCOUNT_ID': aid, 'WORKER_CONCURRENCY': str(account['concurrency']),
                'NATIVE_SHADOW_ENABLED': 'true' if aid == 'agy-03' else 'false',
                'NATIVE_API_PERCENT': {'agy-03': '50', 'agy-06': '50',
                    'agy-08': '70', 'agy-11': '100', 'agy-12': '100', 'agy-13': '100',
                    'agy-16': '70',
                    'agy-18': '50', 'agy-19': '100', 'agy-21': '100',
                    'agy-25': '70'}.get(aid, '0'),
                'NATIVE_API_MAX_PERCENT': {'agy-03': '50', 'agy-06': '70',
                    'agy-08': '70', 'agy-11': '100', 'agy-12': '100', 'agy-13': '100',
                    'agy-16': '70',
                    'agy-18': '70', 'agy-19': '100', 'agy-21': '100',
                    'agy-25': '70'}.get(aid, '50'),
                'NATIVE_API_CONCURRENCY': '8' if aid in ('agy-13', 'agy-21') else '5',
                'AGY_CLI_CONCURRENCY': '5' if aid in ('agy-11', 'agy-13', 'agy-19', 'agy-21') else '64',
                'ANNOTATION_ALLOWED_MODELS': ','.join(models),
                'HTTPS_PROXY': f'http://{aid}-proxy:1080', 'HTTP_PROXY': f'http://{aid}-proxy:1080',
                'NO_PROXY': 'postgres,minio,127.0.0.1,localhost'},
            'depends_on': {**deps, aid + '-proxy': {'condition': 'service_started'}},
            'networks': [private], 'volumes': [f'{agy_path}:/opt/bin/agy:ro',
                                              f'{aid}-home:/home/worker',
                                              f'{REPO}/lalm/benchmark_langfuse_asr.py:/opt/speech/lalm/benchmark_langfuse_asr.py:ro'],
            'tmpfs': ['/work:uid=10001,gid=10001,mode=0700,size=1g', '/tmp:size=256m'],
            'read_only': True, 'cap_drop': ['ALL'], 'security_opt': ['no-new-privileges:true'],
            'pids_limit': 1024, 'mem_limit': '16g' if account['concurrency'] > 10 else '8g',
            'cpus': 4, 'stop_grace_period': '720s' if aid == 'agy-25' else '360s'}
    return {'name': 'speech-data', 'services': services, 'networks': networks, 'volumes': volumes}


def use_external_storage(compose, accounts):
    """Use existing dev-1 S3; TLS passes through a TCP relay unchanged."""
    services = compose['services']
    del services['minio']
    compose['volumes'].pop('s3-data', None)
    nets = ['backend'] + [a['id'] + '-private' for a in accounts]
    alias = 'exp-srv-rtx6000x4-02'
    compose['networks']['storage-egress'] = {}
    services['s3-relay'] = {
        'image': 'alpine/socat:1.8.0.3',
        'command': ['TCP-LISTEN:19000,fork,reuseaddr', 'TCP:100.119.47.17:19000'],
        'networks': {**{n: {'aliases': [alias]} for n in nets}, 'storage-egress': {}},
        'read_only': True, 'user': '10001:10001', 'cap_drop': ['ALL'],
        'security_opt': ['no-new-privileges:true'], 'restart': 'unless-stopped'}
    for svc in services.values():
        deps = svc.get('depends_on', {})
        if 'minio' in deps:
            del deps['minio']; deps['s3-relay'] = {'condition': 'service_started'}
        env = svc.get('environment', {})
        if 'S3_ENDPOINT' in env:
            env.update(S3_ENDPOINT=f'https://{alias}:19000', S3_BUCKET='speech-annotations',
                       S3_DATASET_BUCKET='speech-datasets', S3_CA_BUNDLE='/etc/speech/ca.pem')
            binding = './runtime/storage-ca.pem:/etc/speech/ca.pem:ro'
            if binding not in svc.setdefault('volumes', []): svc['volumes'].append(binding)
            if 'NO_PROXY' in env: env['NO_PROXY'] += ',' + alias
    return compose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accounts', type=Path, default=ROOT / 'accounts.example.json')
    parser.add_argument('--agy', type=Path, required=True)
    parser.add_argument('--storage-env', type=Path, required=True)
    parser.add_argument('--storage-ca', type=Path, required=True)
    args = parser.parse_args()
    accounts = json.loads(args.accounts.read_text())['accounts']
    compose = use_external_storage(build(accounts, str(args.agy.resolve())), accounts)
    binary_hash = hashlib.sha256(args.agy.read_bytes()).hexdigest()
    for account in accounts:
        compose['services'][account['id']]['environment']['AGY_EXPECTED_SHA256'] = binary_hash
    (ROOT / 'init-db.sql').chmod(0o644)
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
    storage = {}
    for line in args.storage_env.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1); storage[key] = shlex.split(value)[0]
    values = dict(line.split('=', 1) for line in envpath.read_text().splitlines() if '=' in line)
    values['S3_ACCESS_KEY'] = storage['AWS_ACCESS_KEY_ID']
    values['S3_SECRET_KEY'] = storage['AWS_SECRET_ACCESS_KEY']
    envpath.write_text(''.join(k+'='+v+'\n' for k,v in values.items()))
    envpath.chmod(0o600)
    shutil.copyfile(args.storage_ca, ROOT / 'runtime/storage-ca.pem')
    (ROOT / 'runtime/storage-ca.pem').chmod(0o644)
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
