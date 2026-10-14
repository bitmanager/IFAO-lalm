from datetime import datetime, timedelta, timezone

from airflow.sdk import Asset, dag, task


ops_snapshot = Asset('s3://speech-datasets/analytics/ops/latest.json')


@dag(schedule='*/5 * * * *', start_date=datetime(2026, 10, 14, tzinfo=timezone.utc),
     catchup=False, max_active_runs=1, dagrun_timeout=timedelta(minutes=3),
     tags=['speech', 'annotation', 'monitoring'])
def annotation_ops_monitor():
    @task(retries=1, retry_delay=timedelta(seconds=30), outlets=[ops_snapshot])
    def publish_operational_metrics(*, outlet_events):
        from lalm.annotation.ops_monitor import publish_metrics
        result = publish_metrics()
        outlet_events[ops_snapshot].extra = {
            **result, 'purpose': 'annotation_operations_monitoring_no_transcripts',
        }
        return result

    publish_operational_metrics()


annotation_ops_monitor()
