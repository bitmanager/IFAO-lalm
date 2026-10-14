"""Conservative native API shadow work. Results are review-only."""
from datetime import datetime, timezone
from airflow.sdk import dag, task


@dag(schedule='0 */6 * * *', start_date=datetime(2026, 10, 14, tzinfo=timezone.utc),
     catchup=False, max_active_runs=1, tags=['speech', 'annotation', 'shadow'])
def native_api_shadow_pipeline():
    @task(retries=1)
    def enqueue():
        from lalm.annotation.native_shadow_queue import schedule
        return schedule()

    enqueue()


native_api_shadow_pipeline()
