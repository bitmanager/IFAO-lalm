"""One account per container; bounded parallel fresh AGY processes."""
import concurrent.futures
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import wave

from lalm.benchmark_langfuse_asr import agy_result, zero_user_audio
from .contract import CONTRACT_SHA, MODEL, SCHEMA, prompt, contract_sha, source_text, digest, normalize_language
from .queue import claim, finish
from .store import bucket, s3

STOP = threading.Event()


def classify(text, timed_out=False):
    # Native streams include model reasoning and transcripts, which are not
    # provider failures (e.g. the model can itself say "forbidden").
    failures = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            failures.append(line)
            continue
        if not isinstance(event, dict):
            continue
        envelope = event.get('result', event)
        if isinstance(envelope, dict):
            failures.append(json.dumps({k: envelope[k] for k in
                ('error', 'error_code', 'code') if k in envelope}))
            if isinstance(envelope.get('status'), int):
                failures.append(f"status {envelope['status']}")
    text = '\n'.join(failures).lower()
    if any(x in text for x in ('individual quota reached', 'quota exhausted', 'quota exceeded')):
        return 'NEEDS_REVIEW', 'quota_exhausted', 'quota_exhausted'
    if any(x in text for x in ('authentication required', 'unauthenticated', 'permission denied',
                                'access denied', 'unauthorized', 'forbidden')):
        return 'NEEDS_REVIEW', 'authentication_or_access_denied', 'authentication_or_access_denied'
    if timed_out:
        return 'NEEDS_REVIEW', 'ambiguous_timeout', None
    if (any(x in text for x in ('too many requests', 'rate limit'))
            or re.search(r'\b(?:code|error_code|status|http)\b[\s\"=:()]*429\b', text)):
        return 'RETRYABLE_FAILED', 'rate_limited', None
    return 'NEEDS_REVIEW', 'native_result_rejected', None


def quota_reset_delay(text):
    match = re.search(r'Resets in ((?:[0-9]+[hms])+)\.', text)
    if not match:
        return None
    seconds = sum(int(n) * {'h': 3600, 'm': 60, 's': 1}[unit]
                  for n, unit in re.findall(r'([0-9]+)([hms])', match[1]))
    return seconds + 60 if 0 < seconds <= 7 * 86400 else None


def run_agy(audio, workspace, variant="verbatim", reference=None):
    command = ['agy', '--model', MODEL, '--disable-slash-commands',
               '--output-format', 'stream-json', '--json-schema', json.dumps(SCHEMA),
               '--print-timeout', '5m', '--add-dir', str(workspace), '-p', prompt(audio, variant, reference)]
    # The model process has its own AGY login but does not inherit DB/S3 credentials.
    child_env = {k: v for k, v in os.environ.items() if k in {
        'PATH', 'HOME', 'LANG', 'LC_ALL', 'TZ', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY',
        'SSL_CERT_FILE', 'SSL_CERT_DIR'}}
    proc = subprocess.Popen(command, cwd=workspace, env=child_env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=330)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr, timed_out


def process(job):
    client = s3()
    prefix = f"annotations/raw/{job['id']}/{job['attempt']}"
    with tempfile.TemporaryDirectory(prefix='annotation-', dir='/work') as temp:
        workspace = Path(temp)
        audio = workspace / 'audio.wav'
        sample = job['sample']
        # Cap inputs before downloading; preprocessor must publish immutable normalized WAVs.
        head = client.head_object(Bucket=sample.get('audio_bucket', bucket()), Key=sample['audio_key'])
        if not 44 < head['ContentLength'] < 100 * 1024 * 1024:
            finish(job, 'PERMANENT_FAILED', error_code='invalid_audio_size')
            return
        client.download_file(sample.get('audio_bucket', bucket()), sample['audio_key'], str(audio))
        if hashlib.sha256(audio.read_bytes()).hexdigest() != sample['audio_sha256']:
            finish(job, 'PERMANENT_FAILED', error_code='audio_hash_mismatch')
            return
        if zero_user_audio(audio):
            finish(job, 'NEEDS_REVIEW', error_code='silent_audio')
            return
        with wave.open(str(audio)) as stream:
            duration = stream.getnframes() / stream.getframerate()
        variant = sample.get("prompt_variant", "verbatim")
        reference = source_text(sample) if variant == "source_correction" else None
        started = time.monotonic()
        rc, stdout, stderr, timed_out = run_agy(audio, workspace, variant, reference)
        elapsed = time.monotonic() - started
        # Native raw output can contain transcript/PII: private S3, never stdout.
        for suffix, body in (('stream.jsonl', stdout), ('stderr.log', stderr),
                             ('request.json', json.dumps({'model': MODEL, 'prompt': prompt(audio, variant, reference),
                              'schema': SCHEMA, 'contract_sha256': contract_sha(variant),
                              'audio_sha256': sample['audio_sha256'], 'exit_code': rc}))):
            client.put_object(Bucket=bucket(), Key=f'{prefix}/{suffix}', Body=body.encode())
        try:
            if rc != 0 or timed_out:
                raise ValueError('native_exit')
            native = agy_result(stdout, audio)
        except (ValueError, KeyError, TypeError):
            state, code, block = classify(stdout + stderr, timed_out)
            delay = quota_reset_delay(stdout + stderr) if block == 'quota_exhausted' else None
            finish(job, state, error_code=code, retry_seconds=delay or 300 * job['attempt'],
                   block='quota_cooldown' if delay is not None else block)
            return
        out = native['structured_output']
        result = {'verbatim_text': out['transcript'], 'language': normalize_language(out['language']), 'language_raw': out['language'],
                  'uncertain_spans': out['uncertain_fragments'], 'native_result': native,
                  'annotation_source': 'agy', 'model': MODEL, 'contract_sha256': contract_sha(variant),
                  'raw_prefix': prefix, 'audio_seconds': duration, 'human_gold': False,
                  'prompt_variant': variant, 'inference_seconds': elapsed,
                  'source_conditioned': variant == 'source_correction',
                  'source_text_sha256': digest(reference) if reference is not None else None}
        state = 'VALIDATED' if result['language'] == 'ru' and out['transcript'].strip() else 'NEEDS_REVIEW'
        if state == 'VALIDATED' and sample.get('review_only'):
            finish(job, 'NEEDS_REVIEW', result, 'pilot_review_only')
        else:
            finish(job, state, result, None if state == 'VALIDATED' else
                   'empty_transcript' if not out['transcript'].strip() else 'non_russian')


def loop(account):
    while not STOP.is_set():
        job = None
        try:
            job = claim(account)
            if job:
                process(job)
                print(json.dumps({'job_id': job['id'], 'attempt': job['attempt'],
                                  'event': 'attempt_finished'}), flush=True)
            else:
                STOP.wait(2)
        except Exception as error:
            # Avoid dumping credential-bearing connection errors or AGY diagnostics.
            print(json.dumps({'event': 'worker_error', 'type': type(error).__name__}), flush=True)
            if job:
                try:
                    finish(job, 'NEEDS_REVIEW', error_code='ambiguous_worker_error')
                except Exception:
                    pass  # Lease expiry independently quarantines the job.
            STOP.wait(5)


def main():
    expected = os.environ.get('AGY_EXPECTED_SHA256')
    if not expected or hashlib.sha256(Path('/opt/bin/agy').read_bytes()).hexdigest() != expected:
        raise RuntimeError('AGY binary missing or changed; rerender and validate the new version')
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: STOP.set())
    account = os.environ['ACCOUNT_ID']
    workers = int(os.environ.get('WORKER_CONCURRENCY', '2'))
    if not 1 <= workers <= 64:
        raise ValueError('invalid_concurrency')
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda _: loop(account), range(workers)))


if __name__ == '__main__':
    main()
