-- Read-only views: source labels, machine output and human approval stay distinct.
CREATE OR REPLACE VIEW review_sources AS
SELECT id AS source_id, split_part(object_key,'/',2) AS corpus,
       bucket, object_key, etag, bytes, codec, state AS preparation_state,
       (cursor_json->>'rows')::bigint AS rows_read,
       (cursor_json->>'samples')::bigint AS samples_prepared,
       (cursor_json->>'errors')::bigint AS preparation_errors,
       cursor_json->>'error_type' AS error_type, created_at, updated_at,
       cursor_json->>'source_kind' AS source_kind,
       (cursor_json->>'metadata_rows')::bigint AS metadata_rows,
       cursor_json->>'audio_link_status' AS audio_link_status
FROM source_assets;

CREATE OR REPLACE VIEW review_passes AS
SELECT id AS job_id, sample->>'pilot_id' AS batch_id,
       sample->>'sample_id' AS sample_id, sample->>'conversation_id' AS conversation_id,
       split_part(COALESCE(sample#>>'{source_provenance,key}',
                           sample#>>'{source_provenance,source_key}'), '/', 2) AS corpus,
       COALESCE(sample->>'prompt_variant','verbatim') AS prompt_variant,
       sample->>'audio_bucket' AS audio_bucket, sample->>'audio_key' AS audio_key,
       sample->>'audio_sha256' AS audio_sha256,
       COALESCE(sample#>'{source_provenance,original_label}',
                sample#>'{source_provenance,source_label}') AS original_annotation,
       CASE WHEN COALESCE(sample#>'{source_provenance,original_label}',
                          sample#>'{source_provenance,source_label}') IS NULL
                  OR COALESCE(sample#>'{source_provenance,original_label}',
                              sample#>'{source_provenance,source_label}')='null'::jsonb
            THEN 'not_linked_or_not_available' ELSE 'attached' END AS original_annotation_status,
       state AS processing_state, error_code, attempt, account_id,
       result->>'verbatim_text' AS gemini_text, result->>'language' AS language,
       (result->>'inference_seconds')::double precision AS inference_seconds,
       (result->>'audio_seconds')::double precision AS audio_seconds,
       result->>'model' AS model, result->>'contract_sha256' AS contract_sha256,
       result->>'raw_prefix' AS raw_result_prefix,
       COALESCE((sample->>'review_only')::boolean,false) AS review_required,
       false AS human_verified,
       sample->'source_provenance' AS source_provenance,
       created_at, updated_at
FROM jobs;

CREATE OR REPLACE VIEW review_examples AS
SELECT batch_id, sample_id, min(corpus) AS corpus,
       min(audio_bucket) AS audio_bucket, min(audio_key) AS audio_key,
       min(audio_sha256) AS audio_sha256,
       max(original_annotation::text) AS original_annotation_json,
       min(original_annotation_status) AS original_annotation_status,
       max(gemini_text) FILTER (WHERE prompt_variant='verbatim') AS gemini_verbatim,
       max(gemini_text) FILTER (WHERE prompt_variant='coverage') AS gemini_coverage,
       max(gemini_text) FILTER (WHERE prompt_variant='literal_audit') AS gemini_literal_audit,
       max(gemini_text) FILTER (WHERE prompt_variant='source_correction') AS gemini_source_correction,
       count(*) AS passes_requested, count(gemini_text) AS passes_with_text,
       count(*) FILTER (WHERE processing_state IN ('PENDING','RUNNING','RETRYABLE_FAILED')) AS unfinished_passes,
       count(*) FILTER (WHERE error_code IS NOT NULL AND error_code<>'pilot_review_only') AS flagged_passes,
       bool_or(review_required) AS review_required, false AS human_verified,
       max(updated_at) AS updated_at
FROM review_passes GROUP BY batch_id,sample_id;

CREATE OR REPLACE VIEW review_progress AS
SELECT corpus, prompt_variant, processing_state, error_code, count(*) AS requests,
       count(gemini_text) AS results_with_text,
       sum(audio_seconds)/3600 AS processed_audio_hours_per_pass,
       avg(inference_seconds) AS mean_inference_seconds, max(updated_at) AS last_update
FROM review_passes GROUP BY corpus,prompt_variant,processing_state,error_code;

CREATE OR REPLACE VIEW review_publications AS
SELECT id AS publication_id,manifest_key,count AS published_examples,created_at FROM publications;
