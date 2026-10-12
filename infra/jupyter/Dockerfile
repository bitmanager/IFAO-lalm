FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && rm -rf /var/lib/apt/lists/* \
 && npm install -g configurable-http-proxy@5.0.1
RUN pip install --no-cache-dir jupyterhub==5.3.0 jupyterlab==4.4.10 duckdb==1.4.1 pandas==2.3.3 boto3==1.42.30 pyarrow==21.0.0
COPY infra/annotation/requirements.txt /tmp/speech-requirements.txt
RUN pip install --no-cache-dir -r /tmp/speech-requirements.txt
RUN useradd --create-home --uid 10002 analyst
COPY --chmod=644 lalm/benchmark_langfuse_asr.py /opt/speech/lalm/benchmark_langfuse_asr.py
COPY lalm/annotation /opt/speech/lalm/annotation
COPY notebooks /opt/notebooks
COPY infra/jupyter/config.py /etc/jupyterhub/config.py
COPY infra/jupyter/start.py /opt/start.py
ENV PYTHONPATH=/opt/speech
CMD ["python", "/opt/start.py"]
