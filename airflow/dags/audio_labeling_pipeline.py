from datetime import datetime, timezone

from airflow.sdk import dag, task


@dag(schedule='*/5 * * * *', start_date=datetime(2026, 10, 12, tzinfo=timezone.utc),
     catchup=False, max_active_runs=1, tags=['speech', 'annotation'])
def audio_labeling_pipeline():
    @task(retries=2)
    def discover_source_objects():
        from lalm.annotation.discovery import discover
        return discover()

    @task(retries=2)
    def queue_source_preparation():
        from lalm.annotation.discovery import queue_sources
        return queue_sources()

    @task(retries=2)
    def publish_validated():
        from lalm.annotation.pipeline import publish
        return publish()

    @task(retries=2)
    def compare_independent_passes():
        from lalm.annotation.review import reconcile
        return reconcile()

    compared = compare_independent_passes()
    # Prepared manifests already exist independently of this inventory scan.
    discover_source_objects() >> queue_source_preparation()
    compared >> publish_validated()


audio_labeling_pipeline()
