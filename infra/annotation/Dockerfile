FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libsecret-1-0 \
    && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 10001 worker \
    && mkdir /work && chown worker:worker /work
COPY infra/annotation/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /opt/speech
COPY lalm/benchmark_langfuse_asr.py lalm/benchmark_langfuse_asr.py
COPY lalm/annotation lalm/annotation
ENV PYTHONPATH=/opt/speech PYTHONUNBUFFERED=1 PATH=/opt/bin:$PATH
USER worker
CMD ["python", "-m", "lalm.annotation.worker"]
