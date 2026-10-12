from datetime import datetime, timezone

from airflow.sdk import dag, task


@dag(schedule='*/5 * * * *', start_date=datetime(2026, 10, 12, tzinfo=timezone.utc),
     catchup=False, max_active_runs=1, tags=['speech', 'annotation'])
def audio_labeling_pipeline():
    @task(retries=2)
    def ingest_manifests():
        from lalm.annotation.pipeline import ingest
        return ingest()

    @task(retries=2)
    def publish_validated():
        from lalm.annotation.pipeline import publish
        return publish()

    ingest_manifests() >> publish_validated()


audio_labeling_pipeline()
