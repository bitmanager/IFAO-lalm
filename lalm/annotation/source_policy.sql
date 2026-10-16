-- User decision, 2026-10-16: keep existing artifacts, stop new Gemini work.
-- Database guard also covers workers already running older Python code.
CREATE OR REPLACE FUNCTION exclude_openstt_annotation_jobs()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.state IN ('PENDING', 'RETRYABLE_FAILED')
       AND split_part(NEW.sample->'source_provenance'->>'key', '/', 2)
           LIKE 'openstt-%' THEN
        NEW.state := 'NEEDS_REVIEW';
        NEW.error_code := 'corpus_excluded_by_user';
    END IF;
    RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS exclude_openstt_annotation_jobs ON jobs;
CREATE TRIGGER exclude_openstt_annotation_jobs
BEFORE INSERT OR UPDATE OF state, sample ON jobs
FOR EACH ROW EXECUTE FUNCTION exclude_openstt_annotation_jobs();
