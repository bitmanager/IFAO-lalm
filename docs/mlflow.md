# MLflow monitoring

Native training and QA evaluation support opt-in MLflow tracking. Install
`pip install -r lalm/requirements-monitoring.txt`, then use the normal entrypoint:

```sh
MLFLOW_ENABLED=true MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python lalm/train.py ...
```

Set the actual tracking address reachable from that process. Hydra's
`monitoring.enabled=true` also enables logging; `MLFLOW_ENABLED` takes precedence.
Default configs leave monitoring disabled so environments without MLflow retain
existing behavior. Only DDP rank 0 creates a run. Missing dependencies or an
unreachable server during initialization raise an explicit error; transient
metric-delivery failures are logged without aborting optimization.

Training records current/average loss, token accuracy, learning rate, gradient
norm/scale, examples and target tokens per second, audio frames, PyTorch peak
allocated/reserved memory, validation splits, model counts, config hashes and
Git revision. System monitoring records CPU/RAM and GPU usage/memory; a sampler
adds per-device power draw/limit, temperature and clocks for NVIDIA/AMD devices.
Unsupported hardware readings are omitted. QA records accuracy, sample counts,
generation time and mean/p95 batch latency. CUDA synchronization brackets timing.
Predictions, audio, transcripts and model weights are not uploaded as artifacts.

## Tracking server and private dashboard

`infra/mlflow/start.sh` runs MLflow on loopback using persistent SQLite and
artifact directories under `MLFLOW_ROOT`. Install the monitoring dependencies
in `${MLFLOW_ROOT}/venv`, then run the script in a persistent service/session.
Set `MLFLOW_ALLOWED_HOSTS` explicitly for the intended dashboard hostnames and
ports, and `MLFLOW_CORS_ALLOWED_ORIGINS` to the browser origins including scheme
and port (for example `http://my-tailnet-host:8445`). Host validation alone can
let the page load while browser POST requests for charts receive HTTP 403.
The defaults allow loopback only. Storage and dependencies are operator
managed; the script is not a production database provisioning tool.

An SSH local forward can expose the tracking port on a workstation. To publish
through a server's private network, forward to a loopback port on that server
and use Tailscale Serve or `infra/mlflow/tailnet_proxy.py`. The TCP proxy binds
only to the address explicitly supplied by `--listen-host`. For the provided
systemd template, install it under `/usr/local/lib/speech-mlflow-tailnet`, copy
`tailnet-proxy.env.example` to `/etc/speech-mlflow/tailnet-proxy.env`, and replace
its address with that server's Tailscale IP. The direct TCP proxy uses HTTP;
HTTPS requires a TLS terminating proxy. SSH tunnels must remain running.

## Existing training and serving processes

`infra/mlflow/collect_existing_runs.py` polls TensorBoard scalar events, numeric
JSONL logs, hardware telemetry and Prometheus serving endpoints without changing
or restarting existing jobs. TensorBoard points retain source steps and wall
times. JSONL without a wall clock uses import time while retaining training step
and elapsed-time values. Each configured source has its own run and persisted
cursor, advanced after successful delivery. Transport is at-least-once: a crash
partway through a batch may replay already sent points. Repeating a completed
import with its cursor sends no duplicate points.

Hardware/serving metrics use a separate host run. Whole-host GPU usage does not
identify a particular job's load. Prometheus counters remain counters; they are
not treated as losses. Label combinations have distinct metric keys. `--once`
ends runs when the import finishes; that status describes the import, not a
verified training outcome. A continuous collector remains RUNNING while its
service is active. A source's optional PID adds `monitor/source_process_running`
and a collector heartbeat. These use persistent monitoring steps so process
status stays current even after the source stops writing training events.

```json
{
  "tracking_uri": "http://127.0.0.1:5000",
  "experiment": "SpeechLLM R&D",
  "interval_seconds": 30,
  "hardware": true,
  "host_run_name": "hardware/my-host",
  "sources": [{"name": "training/my-run", "format": "tensorboard", "path": "/runs/my-run/tensorboard"}],
  "prometheus": ["http://127.0.0.1:30000/metrics"]
}
```

Run `python infra/mlflow/collect_existing_runs.py --config config.json --state state.json`
(or add `--once` for history). Persist `state.json` across restarts and run with
an account that can read the logs. JSONL sources use `"format": "jsonl"` and can
restrict numeric fields using a `metrics` array. The collector is independent
of the trainer's opt-in setting and needs TensorBoard only for that source type.

## Validation

`python -m unittest discover -s tests -p 'test_mlflow*.py'` tests incomplete
records, retries/cursors, metric filtering, label separation, rank suppression,
boolean settings and SDK delivery failures without a GPU.
`MLFLOW_TRACKING_URI=... python tests/mlflow_native_smoke.py` verifies native
metric delivery and run finalization against a real tracking server without a
GPU training job. Full model training remains a separate integration check.

For training loss and GPU metrics, open **Model training → Runs**. A direct link
is `/#/experiments/<experiment_id>/runs?workflowType=machine_learning`.
The GenAI overview queries traces, latency and token usage; training scalars do
not populate those charts. Browser requests must use an origin configured in
`MLFLOW_CORS_ALLOWED_ORIGINS`, even when serving the UI through an SSH/TCP proxy.

## Server-only deployment

Run tracking/storage on an always-on server, independently of the agent
workstation. The native `speech-mlflow-tracking.service` template uses a dedicated
`speech-mlflow` system account, `/var/lib/speech-mlflow/tracking` for durable data,
and `/etc/speech-mlflow/tracking.env` for configuration. `MLFLOW_EXECUTABLE` can
point to an operator-installed virtualenv separately from `MLFLOW_ROOT`.
The matching `tracking.env.example`, `forward.env.example`, and
`tailnet-proxy.env.example` show the operator configuration. Create the system
account and data directory before enabling tracking. Install the startup and
collector modules in `/usr/local/lib/speech-mlflow-monitor`, with
`mlflow_monitoring.py` alongside the collector, and persist each collector's
state directory. Enable the required systemd services so they survive logout
and restart after a server reboot.
The Compose example is an alternative deployment; choose one tracking backend.
Both bind the tracking port on loopback. The private-interface proxy runs on
the same server and forwards directly to that local port.

Collectors also run on servers. If a source cannot reach tracking directly,
`speech-mlflow-forward@.service` can keep a server-to-server SSH link alive; its
instance environment supplies the key, verified known-hosts file and forwarding
specification. Use dedicated keys restricted to the required forwarding ports,
not copied workstation credentials. No runtime forwarding through the agent
workstation is required.

For a remote GPU, `gpu_command` can be an operator-configured argv list that
returns the same NVIDIA CSV columns as the local probe. For example, use SSH
with a key forced to execute only the fixed `nvidia-smi` query. The collector
runs this argv without a shell and parses numeric GPU fields; it does not mix
its own host RAM with remote GPU statistics. Prometheus can be scraped through
a server-side local forward. This avoids installing a persistent monitoring
process on the remote GPU workload host.

During migration, pause collectors, stop the old tracking process, checkpoint
SQLite, preserve the database/artifact paths and collector cursors, then start
the new backend before resuming collectors. Retain a snapshot and verify run IDs,
metric histories and fresh heartbeats before removing the old connections.
