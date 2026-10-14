from datetime import datetime, timezone

from airflow.sdk import Asset, dag, task


inspection_snapshot = Asset('s3://speech-datasets/analytics/latest.json')


@dag(schedule='*/10 * * * *', start_date=datetime(2026, 10, 12, tzinfo=timezone.utc),
     catchup=False, max_active_runs=1, tags=['speech', 'annotation', 'inspection'])
def inspection_snapshot_pipeline():
    @task(retries=2, outlets=[inspection_snapshot])
    def publish_inspection_parquet(*, outlet_events):
        from lalm.annotation.snapshots import publish_snapshots
        result = publish_snapshots()
        outlet_events[inspection_snapshot].extra = {
            **result, 'purpose': 'inspection_only_not_training_release',
        }
        return result

    publish_inspection_parquet()


inspection_snapshot_pipeline()
