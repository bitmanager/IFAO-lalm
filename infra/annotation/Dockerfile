FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 worker \
    && mkdir /work && chown worker:worker /work
COPY infra/annotation/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /opt/speech
COPY --chown=worker:worker lalm/benchmark_langfuse_asr.py lalm/benchmark_langfuse_asr.py
COPY --chown=worker:worker lalm/annotation lalm/annotation
ENV PYTHONPATH=/opt/speech PYTHONUNBUFFERED=1 PATH=/opt/bin:$PATH
USER worker
CMD ["python", "-m", "lalm.annotation.worker"]
