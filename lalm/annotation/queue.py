"""Atomic claims and persistent per-account limits across worker restarts."""
import uuid

from psycopg.types.json import Jsonb

from .contract import MODEL, allowed_models, job_id
from .store import connect


def protected(conn, sample):
    return conn.execute('SELECT 1 FROM holdouts WHERE (kind=%s AND value=%s) '
                        'OR (kind=%s AND value=ANY(%s))',
                        ('conversation_id', sample['conversation_id'],
                         'audio_sha256', [sample['audio_sha256'], sample.get('source_audio_sha256', '')])).fetchone() is not None


def enqueue(conn, sample):
    ident = job_id(sample)
    conn.execute('SELECT pg_advisory_xact_lock(173921)')
    if protected(conn, sample):
        raise ValueError('holdout_contamination')
    # All segments of a conversation and identical audio stay in one split.
    conflict = conn.execute("SELECT 1 FROM jobs WHERE (sample->>'conversation_id'=%s "
                            "OR sample->>'audio_sha256'=%s) AND sample->>'split'<>%s",
                            (sample['conversation_id'], sample['audio_sha256'], sample['split'])).fetchone()
    if conflict:
        raise ValueError('split_contamination')
    conn.execute('INSERT INTO jobs(id,sample) VALUES (%s,%s) ON CONFLICT DO NOTHING',
                 (ident, Jsonb(sample)))
    return ident


def claim(account):
    with connect() as conn:
        row = conn.execute('SELECT *, next_request_at<=now() AS ready FROM accounts '
                           'WHERE id=%s FOR UPDATE', (account,)).fetchone()
        if not row or not row['enabled'] or row['blocked_reason'] or not row['ready']:
            return None
        active = conn.execute("SELECT count(*) AS n FROM jobs WHERE account_id=%s AND state='RUNNING'",
                              (account,)).fetchone()['n']
        if active >= row['concurrency']:
            return None
        job = conn.execute("SELECT * FROM jobs WHERE state IN ('PENDING','RETRYABLE_FAILED') "
                           "AND next_attempt_at<=now() AND attempt<3 "
                           "AND COALESCE(sample->>'annotation_model',%s)=ANY(%s) "
                           "ORDER BY (sample->>'prompt_variant'='verbatim'), created_at "
                           'FOR UPDATE SKIP LOCKED LIMIT 1', (MODEL, list(allowed_models()))).fetchone()
        if not job:
            return None
        if protected(conn, job['sample']):
            conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',error_code='holdout_contamination' WHERE id=%s",
                         (job['id'],))
            return None
        token = str(uuid.uuid4())
        job = conn.execute("UPDATE jobs SET state='RUNNING',account_id=%s,attempt=attempt+1,lease_token=%s,"
                           "lease_until=now()+interval '10 minutes',updated_at=now() WHERE id=%s RETURNING *",
                           (account, token, job['id'])).fetchone()
        conn.execute('INSERT INTO attempts(job_id,attempt,account_id,lease_token) VALUES (%s,%s,%s,%s)',
                     (job['id'], job['attempt'], account, token))
        conn.execute("UPDATE accounts SET next_request_at=now()+min_interval_seconds*interval '1 second' WHERE id=%s",
                     (account,))
        return job


def finish(job, state, result=None, error_code=None, retry_seconds=60, block=None):
    with connect() as conn:
        # Use the same lock order as claim to avoid deadlocks.
        conn.execute('SELECT id FROM accounts WHERE id=%s FOR UPDATE', (job['account_id'],))
        if state == 'RETRYABLE_FAILED' and job['attempt'] >= 3:
            state = 'PERMANENT_FAILED'
        updated = conn.execute("UPDATE jobs SET state=%s,result=%s,error_code=%s,lease_until=NULL,"
                               "next_attempt_at=now()+%s*interval '1 second',updated_at=now() "
                               "WHERE id=%s AND lease_token=%s AND state='RUNNING' RETURNING id",
                               (state, Jsonb(result) if result is not None else None, error_code,
                                retry_seconds, job['id'], job['lease_token'])).fetchone()
        if not updated:
            raise RuntimeError('stale_lease')
        conn.execute('UPDATE attempts SET state=%s,error_code=%s,finished_at=now(),artifact_prefix=%s '
                     'WHERE job_id=%s AND attempt=%s',
                     (state, error_code, f"annotations/raw/{job['id']}/{job['attempt']}", job['id'], job['attempt']))
        if block == 'quota_cooldown':
            conn.execute("UPDATE accounts SET next_request_at=greatest(next_request_at,now()+%s*interval '1 second') WHERE id=%s",
                         (retry_seconds, job['account_id']))
        elif block:
            conn.execute('UPDATE accounts SET enabled=false,blocked_reason=%s WHERE id=%s',
                         (block, job['account_id']))
        elif state == 'RETRYABLE_FAILED':
            conn.execute("UPDATE accounts SET next_request_at=greatest(next_request_at,now()+%s*interval '1 second') "
                         'WHERE id=%s', (retry_seconds, job['account_id']))


def recover_expired():
    # A killed process may already have consumed quota. Never silently resubmit it.
    with connect() as conn:
        rows = conn.execute("UPDATE jobs SET state='NEEDS_REVIEW',error_code='ambiguous_worker_loss',updated_at=now() "
                            "WHERE state='RUNNING' AND lease_until<now() RETURNING id,attempt").fetchall()
        for row in rows:
            conn.execute("UPDATE attempts SET state='NEEDS_REVIEW',error_code='ambiguous_worker_loss',finished_at=now() "
                         'WHERE job_id=%s AND attempt=%s', (row['id'], row['attempt']))
        return len(rows)
