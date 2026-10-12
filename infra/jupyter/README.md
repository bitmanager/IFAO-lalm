# Speech data notebook (dev-1 only)

This is part of **bitmanager/IFAO-lalm**, alongside `airflow/dags` and
`lalm/annotation`. Airflow and JupyterHub import the same read-only bind-mounted
`lalm.annotation` package. Notebooks contain queries and presentation, not a
second implementation of the processing pipeline.

Data stays in existing MinIO. The notebook reads immutable, checksum-verified
Parquet snapshots through `lalm.annotation.inspection.open_snapshot` and existing
read-only training credentials. It has no annotation database or AGY credentials.
The operational PostgreSQL is not the analytical store.

On dev-1, from `/mnt/local/drive2/speech-annotation-20261012/code`:

```sh
docker build --network host -t speech-jupyter:local -f infra/jupyter/Dockerfile .
python3 infra/jupyter/deploy.py
infra/annotation/dc up -d --no-deps jupyterhub
```

Run `deploy.py` again after regenerating the annotation Compose file. It adds
JupyterHub, shared source mounts, tailnet-only Airflow access and the configured
30-request limits for the four active accounts. Apply Airflow/worker changes
with a graceful restart, allowing in-flight requests to finish.

JupyterHub: `http://exp-srv-rtx6000x4-02.tail8f0274.ts.net:18888`
Airflow: `http://exp-srv-rtx6000x4-02.tail8f0274.ts.net:18080`
These addresses require the corporate tailnet. Host ports bind only loopback
and the dev-1 Tailscale address, not the public interface.

Login is `analyst`; the generated password is in
`infra/annotation/runtime/jupyter/password` on dev-1 (mode 0600).
PAM authentication and LocalProcessSpawner run notebooks as unprivileged
`analyst`. Hub state and analyst notebooks persist in separate Docker volumes.
A source notebook is copied on first boot only; subsequent user edits survive.
To refresh it, copy the repository version explicitly after preserving edits.

Never commit credentials, notebook outputs, or production examples.
Snapshots and Gemini responses remain inspection/review data, not human gold.

Existing MinIO console is exposed through Tailscale Serve (no new MinIO):

```sh
sudo tailscale serve --bg --http=19001 https+insecure://127.0.0.1:19001
```

Console: `http://exp-srv-rtx6000x4-02.tail8f0274.ts.net:19001/`.
The HTTPS upstream is loopback with the existing private certificate.
