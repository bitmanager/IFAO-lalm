"""Replenish the worker queue independently of long review comparisons."""
from datetime import datetime, timezone

from airflow.sdk import dag, task


@dag(schedule='*/5 * * * *', start_date=datetime(2026, 10, 12, tzinfo=timezone.utc),
     catchup=False, max_active_runs=1, is_paused_upon_creation=True,
     tags=['speech', 'annotation', 'ingest'])
def annotation_ingest_pipeline():
    @task(retries=2)
    def ingest_manifests():
        from lalm.annotation.pipeline import ingest
        return ingest()

    ingest_manifests()


annotation_ingest_pipeline()
