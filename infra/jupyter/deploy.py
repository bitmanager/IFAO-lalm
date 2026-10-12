"""Dev-1-only deployment; existing S3 training credentials remain read-only."""
import json
from pathlib import Path
import secrets
import socket
if socket.gethostname() != 'exp-srv-rtx6000x4-02':
    raise SystemExit('Run on dev-1 only')
repo = Path(__file__).resolve().parents[2]
# Package contains source only; container users must be able to read bind mounts.
for path in (repo / 'lalm/annotation').rglob('*'):
    path.chmod(0o755 if path.is_dir() else 0o644)
(repo / 'lalm/annotation').chmod(0o755)
for path in (repo / 'notebooks').glob('*.ipynb'):
    path.chmod(0o644)
(repo / 'notebooks').chmod(0o755)
root = repo / 'infra/annotation'
runtime = root / 'runtime/jupyter'
runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
password = runtime / 'password'
if not password.exists():
    password.write_text(secrets.token_urlsafe(24)+'\n')
    password.chmod(0o600)
compose = json.loads((root/'compose.generated.json').read_text())
compose['services']['jupyterhub'] = {
    'image': 'speech-jupyter:local', 'restart': 'unless-stopped',
    'ports': ['100.119.47.17:18888:8000', '127.0.0.1:18888:8000'],
    'networks': ['ui'], 'mem_limit': '6g', 'cpus': 4,
    'volumes': ['jupyter-state:/srv/jupyterhub', 'jupyter-home:/home/analyst',
        str(repo/'notebooks')+':/opt/notebooks:ro',
        str(password)+':/run/secrets/notebook-password:ro',
        '/mnt/local/drive2/speech-data-storage-20261012/config/training.env:/run/secrets/s3-readonly.env:ro',
        '/mnt/local/drive2/speech-data-storage-20261012/config/ca.pem:/etc/speech/ca.pem:ro',
        str(repo/'lalm/annotation')+':/opt/speech/lalm/annotation:ro']}
compose['volumes'].update({'jupyter-state': {}, 'jupyter-home': {}})
# Same read-only source tree is imported by notebooks and all Airflow components.
for name in ('airflow-api','airflow-scheduler','airflow-dag-processor'):
    volumes = compose['services'][name]['volumes']
    mount = str(repo/'lalm/annotation')+':/opt/speech/lalm/annotation:ro'
    if mount not in volumes:
        volumes.append(mount)
ports=compose['services']['airflow-api']['ports']
if '100.119.47.17:18080:8080' not in ports:
    ports.append('100.119.47.17:18080:8080')
for aid in ('agy-01','agy-02','agy-03','agy-07'):
    compose['services'][aid]['environment']['WORKER_CONCURRENCY']='20'
(root/'compose.generated.json').write_text(json.dumps(compose, indent=2)+'\n')
