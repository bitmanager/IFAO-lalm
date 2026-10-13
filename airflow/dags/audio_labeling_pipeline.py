from datetime import datetime, timezone

from airflow.sdk import Asset, dag, task


inspection_snapshot = Asset('s3://speech-datasets/analytics/latest.json')


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
    def ingest_manifests():
        from lalm.annotation.pipeline import ingest
        return ingest()

    @task(retries=2)
    def publish_validated():
        from lalm.annotation.pipeline import publish
        return publish()

    @task(retries=2)
    def compare_independent_passes():
        from lalm.annotation.review import reconcile
        return reconcile()

    @task(retries=2, outlets=[inspection_snapshot])
    def publish_inspection_parquet(*, outlet_events):
        from lalm.annotation.snapshots import publish_snapshots
        result = publish_snapshots()
        outlet_events[inspection_snapshot].extra = {
            **result, 'purpose': 'inspection_only_not_training_release',
        }
        return result

    ingested = ingest_manifests()
    compared = compare_independent_passes()
    inspection = publish_inspection_parquet()
    discover_source_objects() >> queue_source_preparation() >> ingested
    # Inspection reads its own consistent, read-only database snapshot.
    # Reconciliation reports are not an input to these Parquet tables.
    ingested >> [compared, inspection]
    compared >> publish_validated()


audio_labeling_pipeline()
