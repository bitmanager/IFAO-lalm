import shlex
from pathlib import Path
c = get_config()
c.JupyterHub.bind_url = 'http://:8000'
c.Authenticator.allowed_users = {'analyst'}
c.JupyterHub.authenticator_class = 'jupyterhub.auth.PAMAuthenticator'
c.Spawner.default_url = '/lab/tree/speech_data_progress.ipynb'
c.Spawner.notebook_dir = '/home/analyst'
c.Spawner.environment = {'PYTHONPATH': '/opt/speech', 'S3_DATASET_BUCKET': 'speech-datasets'}
for line in Path('/run/secrets/s3-readonly.env').read_text().splitlines():
    if '=' not in line or line.lstrip().startswith('#'):
        continue
    name, value = line.removeprefix('export ').split('=', 1)
    if name in {'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION', 'AWS_ENDPOINT_URL'}:
        c.Spawner.environment[name] = shlex.split(value)[0]
c.Spawner.environment['S3_ENDPOINT'] = c.Spawner.environment['AWS_ENDPOINT_URL']
c.Spawner.environment['AWS_CA_BUNDLE'] = '/etc/speech/ca.pem'
c.JupyterHub.db_url = 'sqlite:////srv/jupyterhub/jupyterhub.sqlite'
c.JupyterHub.cookie_secret_file = '/srv/jupyterhub/cookie_secret'
