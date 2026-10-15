CREATE TABLE IF NOT EXISTS accounts (
  id text PRIMARY KEY,
  enabled boolean NOT NULL DEFAULT false,
  concurrency integer NOT NULL CHECK (concurrency BETWEEN 1 AND 64),
  min_interval_seconds double precision NOT NULL CHECK (min_interval_seconds >= 0),
  next_request_at timestamptz NOT NULL DEFAULT now(),
  blocked_reason text
);
CREATE TABLE IF NOT EXISTS holdouts (
  kind text NOT NULL CHECK (kind IN ('conversation_id','audio_sha256')),
  value text NOT NULL,
  PRIMARY KEY (kind,value)
);
CREATE TABLE IF NOT EXISTS jobs (
  id text PRIMARY KEY,
  sample jsonb NOT NULL,
  state text NOT NULL DEFAULT 'PENDING' CHECK (state IN
    ('PENDING','RUNNING','RETRYABLE_FAILED','NEEDS_REVIEW','PERMANENT_FAILED','VALIDATED','PUBLISHED')),
  account_id text REFERENCES accounts(id),
  attempt integer NOT NULL DEFAULT 0,
  lease_token text,
  lease_until timestamptz,
  next_attempt_at timestamptz NOT NULL DEFAULT now(),
  result jsonb,
  error_code text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(state,next_attempt_at,created_at);
-- Keep clean installs aligned with the existing production identity guards.
CREATE INDEX IF NOT EXISTS jobs_conversation_identity
  ON jobs ((sample->>'conversation_id'));
CREATE INDEX IF NOT EXISTS jobs_audio_identity
  ON jobs ((sample->>'audio_sha256'));
CREATE UNIQUE INDEX IF NOT EXISTS precision_v1_audio_pass_model_uq
  ON jobs ((sample->>'audio_sha256'), (sample->>'prompt_variant'),
           COALESCE(sample->>'annotation_model', 'gemini-3.8-flash-low'))
  WHERE sample->>'pilot_id' LIKE 'precision-ru-triad-v1-%';
-- Frequent state changes need cleanup before the global 20% threshold is met.
ALTER TABLE jobs SET (autovacuum_vacuum_scale_factor=0.02,
                     autovacuum_vacuum_threshold=1000);
CREATE INDEX IF NOT EXISTS jobs_claim_fifo ON jobs(created_at,id)
  WHERE state IN ('PENDING','RETRYABLE_FAILED') AND attempt<3;
CREATE TABLE IF NOT EXISTS attempts (
  job_id text REFERENCES jobs(id),
  attempt integer NOT NULL,
  account_id text NOT NULL,
  lease_token text NOT NULL,
  started_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,
  state text NOT NULL DEFAULT 'RUNNING',
  artifact_prefix text,
  error_code text,
  PRIMARY KEY(job_id,attempt)
);
-- Recent fleet/transport diagnostics should not scan the entire attempt history.
CREATE INDEX IF NOT EXISTS attempts_started_at ON attempts(started_at);
CREATE TABLE IF NOT EXISTS publications (
  id text PRIMARY KEY,
  manifest_key text NOT NULL,
  count integer NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS source_manifests (
  object_key text PRIMARY KEY,
  sha256 text NOT NULL,
  row_count integer NOT NULL,
  imported_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS source_assets (
  id text PRIMARY KEY,
  bucket text NOT NULL,
  object_key text NOT NULL,
  etag text NOT NULL,
  bytes bigint NOT NULL,
  codec text NOT NULL,
  state text NOT NULL DEFAULT 'DISCOVERED',
  cursor_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(bucket,object_key,etag)
);
CREATE INDEX IF NOT EXISTS source_assets_queue ON source_assets(state,codec,created_at);

CREATE TABLE IF NOT EXISTS annotation_review_cursors (
  pilot_id text PRIMARY KEY,
  checked_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_review_group ON jobs ((sample->>'pilot_id'),updated_at);

CREATE TABLE IF NOT EXISTS prepared_manifests (
  object_key text PRIMARY KEY,
  row_count integer NOT NULL CHECK (row_count >= 0),
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Native API shadow work is isolated from training jobs. A claimed batch is
-- never retried automatically because a lost response may have used quota.
CREATE TABLE IF NOT EXISTS native_shadow_batches (
  id text PRIMARY KEY,
  account_id text NOT NULL REFERENCES accounts(id),
  manifest_key text NOT NULL,
  result_key text,
  state text NOT NULL CHECK (state IN ('PENDING','RUNNING','COMPLETE','NEEDS_REVIEW')),
  error_code text,
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz
);
CREATE TABLE IF NOT EXISTS native_shadow_samples (
  audio_sha256 text PRIMARY KEY,
  batch_id text NOT NULL REFERENCES native_shadow_batches(id)
);

-- Keep split-contamination checks efficient as annotation history grows.
CREATE INDEX IF NOT EXISTS jobs_conversation_identity ON jobs ((sample->>'conversation_id'));
CREATE INDEX IF NOT EXISTS jobs_audio_identity ON jobs ((sample->>'audio_sha256'));
-- One inference per exact prepared audio and pass in this prompt version,
-- even when two source manifests reference the same audio under different calls.
CREATE UNIQUE INDEX IF NOT EXISTS precision_v1_audio_pass_model_uq
ON jobs ((sample->>'audio_sha256'), (sample->>'prompt_variant'),
         (COALESCE(sample->>'annotation_model','gemini-3.8-flash-low')))
WHERE sample->>'pilot_id' LIKE 'precision-ru-triad-v1-%';

-- Older workers may classify a 429 after generation as retryable. Enforce
-- quarantine at the durable boundary too, without restarting live workers.
CREATE OR REPLACE FUNCTION quarantine_ambiguous_rate_limit() RETURNS trigger
LANGUAGE plpgsql AS '
BEGIN
  NEW.state := ''NEEDS_REVIEW'';
  NEW.error_code := ''ambiguous_rate_limit_after_dispatch'';
  RETURN NEW;
END;';
CREATE OR REPLACE TRIGGER jobs_quarantine_rate_limit
BEFORE INSERT OR UPDATE OF state,error_code ON jobs
FOR EACH ROW
WHEN (NEW.state = 'RETRYABLE_FAILED' AND NEW.error_code = 'rate_limited')
EXECUTE FUNCTION quarantine_ambiguous_rate_limit();
CREATE OR REPLACE TRIGGER attempts_quarantine_rate_limit
BEFORE INSERT OR UPDATE OF state,error_code ON attempts
FOR EACH ROW
WHEN (NEW.state = 'RETRYABLE_FAILED' AND NEW.error_code = 'rate_limited')
EXECUTE FUNCTION quarantine_ambiguous_rate_limit();
