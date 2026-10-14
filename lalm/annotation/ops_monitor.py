"""Publish compact, PII-free operational metrics for the annotation fleet."""
from datetime import datetime, timezone
import hashlib
import json
from decimal import Decimal

from .store import connect, dataset_bucket, s3


def collect_metrics():
    with connect() as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        captured_at = conn.execute('SELECT transaction_timestamp() AS ts').fetchone()['ts']
        states = conn.execute(
            'SELECT state, count(*)::bigint AS count FROM jobs GROUP BY state ORDER BY state'
        ).fetchall()
        global_metrics = conn.execute("""
            SELECT
              count(*) FILTER (WHERE state='RUNNING')::bigint AS running,
              count(*) FILTER (WHERE state='PENDING')::bigint AS queued,
              count(*) FILTER (WHERE state='RETRYABLE_FAILED')::bigint AS retryable,
              count(*) FILTER (WHERE state='NEEDS_REVIEW')::bigint AS needs_review,
              count(*) FILTER (WHERE state='PERMANENT_FAILED')::bigint AS permanent_failed,
              count(*) FILTER (WHERE result IS NOT NULL AND updated_at >= now()-interval '5 minutes')::bigint AS saved_5m,
              count(*) FILTER (WHERE result IS NOT NULL AND updated_at >= now()-interval '15 minutes')::bigint AS saved_15m,
              count(*) FILTER (WHERE result IS NOT NULL AND updated_at >= now()-interval '60 minutes')::bigint AS saved_60m,
              count(*) FILTER (WHERE state='RUNNING' AND lease_until < now())::bigint AS expired_leases,
              min(lease_until) FILTER (WHERE state='RUNNING') AS earliest_lease_expiry
            FROM jobs
        """).fetchone()
        accounts = conn.execute("""
            WITH job_metrics AS (
              SELECT account_id,
                count(*) FILTER (WHERE state='RUNNING')::bigint AS running,
                count(*) FILTER (WHERE result IS NOT NULL AND updated_at >= now()-interval '5 minutes')::bigint AS saved_5m,
                count(*) FILTER (WHERE result IS NOT NULL AND updated_at >= now()-interval '15 minutes')::bigint AS saved_15m,
                count(*) FILTER (WHERE state='RUNNING')::bigint AS active_jobs,
                min(updated_at) FILTER (WHERE state='RUNNING') AS oldest_running_update
              FROM jobs GROUP BY account_id
            ), attempt_metrics AS (
              SELECT account_id,
                count(*) FILTER (WHERE finished_at >= now()-interval '5 minutes'
                  AND error_code IS NOT NULL AND error_code <> 'pilot_review_only')::bigint AS errors_5m,
                count(*) FILTER (WHERE finished_at >= now()-interval '15 minutes'
                  AND error_code IS NOT NULL AND error_code <> 'pilot_review_only')::bigint AS errors_15m,
                count(*) FILTER (WHERE finished_at >= now()-interval '15 minutes'
                  AND error_code='pilot_review_only')::bigint AS review_outputs_15m,
                count(*) FILTER (WHERE finished_at >= now()-interval '60 minutes' AND error_code='ambiguous_timeout')::bigint AS timeouts_60m,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch FROM (finished_at-started_at)))
                  FILTER (WHERE finished_at >= now()-interval '15 minutes' AND error_code IS NULL) AS median_success_seconds_15m
              FROM attempts WHERE finished_at IS NOT NULL GROUP BY account_id
            ), error_codes AS (
              SELECT account_id, jsonb_object_agg(error_code, n) AS codes_15m
              FROM (
                SELECT account_id,error_code,count(*)::bigint AS n FROM attempts
                WHERE finished_at >= now()-interval '15 minutes' AND error_code IS NOT NULL
                  AND error_code <> 'pilot_review_only'
                GROUP BY account_id,error_code
              ) e GROUP BY account_id
            )
            SELECT a.id,a.enabled,a.concurrency,a.min_interval_seconds,a.blocked_reason,
              greatest(0,extract(epoch FROM (a.next_request_at-now()))) AS cooldown_seconds,
              coalesce(j.running,0)::bigint AS running,
              coalesce(j.saved_5m,0)::bigint AS saved_5m,
              coalesce(j.saved_15m,0)::bigint AS saved_15m,
              CASE WHEN coalesce(j.saved_15m,0)>0 THEN round((j.saved_15m/15.0)::numeric,2) ELSE 0 END AS saved_per_minute_15m,
              extract(epoch FROM (now()-j.oldest_running_update)) AS oldest_running_update_seconds,
              coalesce(am.errors_5m,0)::bigint AS errors_5m,
              coalesce(am.errors_15m,0)::bigint AS errors_15m,
              coalesce(am.review_outputs_15m,0)::bigint AS review_outputs_15m,
              coalesce(am.timeouts_60m,0)::bigint AS ambiguous_timeouts_60m,
              am.median_success_seconds_15m,
              coalesce(ec.codes_15m,'{}'::jsonb) AS error_codes_15m
            FROM accounts a
            LEFT JOIN job_metrics j ON j.account_id=a.id
            LEFT JOIN attempt_metrics am ON am.account_id=a.id
            LEFT JOIN error_codes ec ON ec.account_id=a.id
            ORDER BY a.id
        """).fetchall()
        recent_errors = conn.execute("""
            SELECT a.job_id,a.attempt,a.account_id,a.error_code,a.started_at,a.finished_at,
              round(extract(epoch FROM (a.finished_at-a.started_at))::numeric,1) AS duration_seconds
            FROM attempts a WHERE a.finished_at >= now()-interval '15 minutes'
              AND a.error_code IS NOT NULL AND a.error_code <> 'pilot_review_only'
            ORDER BY a.finished_at DESC LIMIT 100
        """).fetchall()
        source_states = conn.execute(
            'SELECT state,count(*)::bigint AS count FROM source_assets GROUP BY state ORDER BY state'
        ).fetchall()

    metrics = {
        'schema_version': 1,
        'captured_at': captured_at.astimezone(timezone.utc).isoformat(),
        'purpose': 'annotation_operations_monitoring_no_transcripts',
        'jobs_by_state': {row['state']: row['count'] for row in states},
        'global': dict(global_metrics),
        'accounts': [dict(row) for row in accounts],
        'recent_errors': [dict(row) for row in recent_errors],
        'source_assets_by_state': {row['state']: row['count'] for row in source_states},
    }
    return json.loads(json.dumps(metrics, default=_json_value))


def _json_value(value):
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f'Unsupported metric value: {type(value).__name__}')


def publish_metrics():
    payload = json.dumps(collect_metrics(), ensure_ascii=False, sort_keys=True).encode()
    client = s3()
    bucket = dataset_bucket()
    digest = hashlib.sha256(payload).hexdigest()
    stamp = datetime.now(timezone.utc).strftime('%Y/%m/%d/%H%M%S')
    history_key = f'analytics/ops/history/{stamp}-{digest[:12]}.json'
    client.put_object(Bucket=bucket, Key=history_key, Body=payload,
                      ContentType='application/json')
    client.put_object(Bucket=bucket, Key='analytics/ops/latest.json', Body=payload,
                      ContentType='application/json')
    return {'key': history_key, 'sha256': digest, 'bytes': len(payload)}
