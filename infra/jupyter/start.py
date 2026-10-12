"""Initialize private PAM login and persistent workspace, without logging credentials."""
import os
from pathlib import Path
import shutil
import subprocess
password = Path('/run/secrets/notebook-password').read_text().strip()
subprocess.run(['chpasswd'], input='analyst:'+password+'\n', text=True, check=True)
for notebook in Path('/opt/notebooks').glob('*.ipynb'):
    target = Path('/home/analyst') / notebook.name
    if not target.exists():
        shutil.copyfile(notebook, target)
        os.chown(target, 10002, 10002)
os.chown('/home/analyst', 10002, 10002)
os.chdir('/srv/jupyterhub')
os.execvp('jupyterhub', ['jupyterhub', '-f', '/etc/jupyterhub/config.py'])
