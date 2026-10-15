"""One account per container; bounded parallel fresh AGY processes."""
import concurrent.futures
import hashlib
import json
import os
import random
import re
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import wave

from lalm.benchmark_langfuse_asr import agy_result, zero_user_audio
from .contract import CONTRACT_SHA, MODEL, SCHEMA, prompt, contract_sha, source_text, digest, normalize_language, sample_model
from .queue import claim, finish
from .store import bucket, s3
from .native_auth import repair_trailing_auth
from .cli_hooks import install_cli_hooks, install_global_cli_hooks

STOP = threading.Event()
AUTH_PROBE_LOCK = threading.Lock()
AUTH_PROBE_CACHE = {}
NATIVE_PROJECT_LOCK = threading.Lock()
NATIVE_PROJECT_CACHE = None
NATIVE_API_DISABLED_UNTIL = 0.0
NATIVE_API_FAILURE_LOCK = threading.Lock()
NATIVE_API_CONSECUTIVE_FAILURES = 0
NATIVE_API_GOOD_COUNT = 0
NATIVE_API_DEGRADED = False
def native_api_concurrency():
    return min(8, max(1, int(os.environ.get('NATIVE_API_CONCURRENCY', '5'))))


NATIVE_API_SLOTS = threading.BoundedSemaphore(native_api_concurrency())
NATIVE_API_BASE_PERCENT = {'agy-03': 50, 'agy-06': 20, 'agy-08': 70,
                           'agy-12': 50, 'agy-13': 50, 'agy-16': 35,
                           'agy-18': 20, 'agy-19': 20,
                           'agy-21': 60, 'agy-25': 70}


def classify(text, timed_out=False):
    # Native streams include model reasoning and transcripts, which are not
    # provider failures (e.g. the model can itself say "forbidden").
    failures = []
    service_unavailable = False
    inference_unavailable = False
    inference_transport_failure = False
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
            error = envelope.get('error')
            if isinstance(error, str):
                error = error.removeprefix(
                    'failed to send message: send failed; already reported to the user: ')
            service_unavailable |= (event.get('event') == 'result'
                and envelope.get('status') == 'ERROR'
                and envelope.get('num_turns') == 0 and envelope.get('duration_seconds') == 0
                and error in (
                    'Eligibility check failed: failed to get load code assist response: '
                    'UNAVAILABLE (code 503): The service is currently unavailable.',
                    'Eligibility check failed: '
                    'UNAVAILABLE (code 503): The service is currently unavailable.',
                    'Eligibility check failed: failed to get load code assist response: '
                    'INTERNAL (code 500): Internal error encountered.'))
            inference_unavailable |= (event.get('event') == 'result'
                and envelope.get('status') == 'ERROR'
                and isinstance(envelope.get('num_turns'), int)
                and envelope['num_turns'] > 0 and isinstance(error, str)
                and (error == 'The stream was interrupted. Please continue the task you were working on.'
                     or re.fullmatch(
                         r'API error \(attempt [1-9][0-9]*\): (?:'
                         r'INTERNAL \(code 500\): Internal error encountered\.|'
                         r'UNAVAILABLE \(code 503\): No capacity available for model '
                         r'(?:gemini-3\.8-flash-low|gemini-4-argon-low) on the server)', error) is not None))
            error_lower = error.lower() if isinstance(error, str) else ''
            inference_transport_failure |= (event.get('event') == 'result'
                and envelope.get('status') == 'ERROR'
                and isinstance(envelope.get('num_turns'), int)
                and envelope['num_turns'] > 0
                and 'request failed: post "https://' in error_lower
                and any(marker in error_lower for marker in
                        ('": eof', 'connection reset by peer', 'broken pipe')))
            failures.append(json.dumps({k: envelope[k] for k in
                ('error', 'error_code', 'code') if k in envelope}))
            if isinstance(envelope.get('status'), int):
                failures.append(f"status {envelope['status']}")
    text = '\n'.join(failures).lower()
    if any(x in text for x in ('individual quota reached', 'quota exhausted', 'quota exceeded')):
        return 'NEEDS_REVIEW', 'quota_exhausted', 'quota_exhausted'
    if service_unavailable and not any(x in text for x in
            ('authentication required', 'unauthenticated', 'permission denied',
             'access denied', 'unauthorized', 'forbidden', 'verify your account')):
        if timed_out:
            return 'NEEDS_REVIEW', 'ambiguous_timeout', None
        # An init event prevents safe replay. Quarantine the job and cool down
        # the account; Google's temporary 503 does not revoke authentication.
        return 'NEEDS_REVIEW', 'eligibility_service_unavailable', 'transport_cooldown'
    if any(x in text for x in ('authentication required', 'unauthenticated', 'permission denied',
                                'access denied', 'unauthorized', 'forbidden', 'verify your account')):
        return 'NEEDS_REVIEW', 'authentication_or_access_denied', 'authentication_or_access_denied'
    if 'eligibility check failed' in text:
        # Eligibility failures include transient profile/usage API transport
        # errors. Only explicit denial markers above should quarantine an account.
        return 'NEEDS_REVIEW', 'eligibility_check_failed', 'transport_cooldown'
    if inference_transport_failure:
        # Generation already started, so never replay the job; pause new work
        # briefly on this account after the confirmed network failure.
        return 'NEEDS_REVIEW', 'inference_transport_unavailable', 'transport_cooldown'
    if timed_out:
        return 'NEEDS_REVIEW', 'ambiguous_timeout', None
    if inference_unavailable:
        # Generation started: retain the failed job, never replay it. Stop
        # new dispatches briefly while the native service is overloaded.
        return 'NEEDS_REVIEW', 'inference_service_unavailable', 'transport_cooldown'
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


def predispatch_auth_timeout(stdout, stderr, elapsed):
    """Only the observed zero-turn startup failure is safe for one retry."""
    if elapsed >= 180 or 'authentication timed out' not in stderr.lower():
        return False
    try:
        events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except ValueError:
        return False
    if len(events) != 1 or not isinstance(events[0], dict) or events[0].get('event') != 'result':
        return False
    r = events[0].get('result', {})
    return (isinstance(r, dict) and r.get('status') == 'ERROR' and r.get('num_turns') == 0
            and r.get('duration_seconds') == 0
            and r.get('error') == 'authentication failed or timed out')


def probe_auth(model=MODEL, ttl_seconds=60):
    """Native read-only catalog lookup, no prompt and no credential logging."""
    # One account owns this worker process. Share a bounded read-only probe
    # across its threads instead of amplifying an auth outage with N catalogs.
    with AUTH_PROBE_LOCK:
        cached = AUTH_PROBE_CACHE.get(model)
        if ttl_seconds > 0 and cached and cached[0] > time.monotonic():
            return cached[1]
        try:
            repair_trailing_auth()
            result = subprocess.run(['agy', 'models'], env=agy_env(),
                                    capture_output=True, text=True, timeout=40)
            ready = result.returncode == 0 and model in result.stdout
        except (OSError, ValueError, subprocess.TimeoutExpired):
            ready = False
        AUTH_PROBE_CACHE[model] = (time.monotonic() + max(0, ttl_seconds), ready)
        return ready


def predispatch_eligibility_transient(stdout, timed_out=False):
    """Retry only the observed transport failure before any model activity."""
    if timed_out:
        return False
    try:
        events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except ValueError:
        return False
    if len(events) != 1 or not isinstance(events[0], dict) or events[0].get('event') != 'result':
        return False
    r = events[0].get('result')
    if not isinstance(r, dict):
        return False
    error = r.get('error', '')
    transport = (r'Eligibility check failed: '
                 r'(?:failed to get (?:load code assist response|profile picture): )?'
                 r'(?:Post|Get) "https://[^"\s]+":? '
                 r'(?:(?:unexpected )?EOF|net/http: TLS handshake timeout)')
    unavailable = (r'Eligibility check failed: (?:failed to get load code assist response: )?'
                   r'UNAVAILABLE \(code 503\): The service is currently unavailable\.')
    return (r.get('status') == 'ERROR' and r.get('num_turns') == 0
            and r.get('duration_seconds') == 0 and isinstance(error, str)
            and (re.fullmatch(transport, error) is not None
                 or re.fullmatch(unavailable, error) is not None
                 or error == 'Eligibility check failed: request failed (code 502): HTML response "Error 502 (Server Error)!!1"'
                 or error == 'Eligibility check failed: failed to get load code assist response: INTERNAL (code 500): Internal error encountered.'
                 or error == 'Eligibility check failed: failed to fetch user info: 500 Internal Server Error'))



def predispatch_model_catalog_transient(stdout, stderr, model, returncode, timed_out=False):
    """A catalog miss is replayable only with the exact zero-turn CLI failure."""
    expected = (f'error: invalid model selection (--model "{model}" --effort ""): '
                f'model {model} is not recognized as a known model or custom model in settings')
    if timed_out or returncode != 1 or stderr.strip() != expected:
        return False
    try:
        events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except ValueError:
        return False
    if len(events) != 1 or not isinstance(events[0], dict) or events[0].get('event') != 'result':
        return False
    result = events[0].get('result')
    return (isinstance(result, dict) and result.get('status') == 'ERROR'
            and result.get('num_turns') == 0 and result.get('duration_seconds') == 0)


def agy_env():
    return {k: v for k, v in os.environ.items() if k in {
        'PATH', 'HOME', 'LANG', 'LC_ALL', 'TZ', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY',
        'SSL_CERT_FILE', 'SSL_CERT_DIR'}}


def run_agy(audio, workspace, variant="verbatim", reference=None, model=MODEL):
    try:
        if repair_trailing_auth():
            print(json.dumps({'event': 'auth_storage_suffix_repaired',
                              'account': os.environ.get('ACCOUNT_ID')}), flush=True)
    except (OSError, ValueError):
        # Nothing has been submitted to AGY. Keep the failure explicitly zero-turn.
        return 1, json.dumps({'event': 'result', 'result': {'status': 'ERROR',
            'num_turns': 0, 'duration_seconds': 0,
            'error': 'auth_storage_invalid_before_dispatch'}}), '', False
    command = ['agy', '--model', model, '--disable-slash-commands',
               '--output-format', 'stream-json', '--json-schema', json.dumps(SCHEMA),
               '--print-timeout', '5m', '--add-dir', str(workspace), '-p', prompt(audio, variant, reference)]
    # The model process has its own AGY login but does not inherit DB/S3 credentials.
    child_env = agy_env()
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


def native_api_selected(job, duration, size, variant, reference, model):
    account = os.environ.get('ACCOUNT_ID')
    if account not in NATIVE_API_BASE_PERCENT or not job['sample'].get('review_only'):
        return False
    base_percent = int(os.environ.get('NATIVE_API_PERCENT', str(NATIVE_API_BASE_PERCENT[account])))
    max_percent = int(os.environ.get('NATIVE_API_MAX_PERCENT', '50'))
    if not 0 <= base_percent <= 100 or not 0 <= max_percent <= 100:
        return False
    with NATIVE_API_FAILURE_LOCK:
        if NATIVE_API_DEGRADED:
            base_percent = min(base_percent, 20)
        percent = min(max_percent, max(base_percent, 50 if NATIVE_API_GOOD_COUNT >= 100
                      else 35 if NATIVE_API_GOOD_COUNT >= 25 else base_percent))
    if (model != 'gemini-3.8-flash-low'
            or time.monotonic() < NATIVE_API_DISABLED_UNTIL
            or not 1 <= duration <= 20 or size >= 2 * 1024**2
            or (variant == 'source_correction' and reference is None)):
        return False
    return int(job['id'][:8], 16) % 100 < percent


def native_project():
    global NATIVE_PROJECT_CACHE
    from .native_api import discover, token
    with NATIVE_PROJECT_LOCK:
        # Avoid one discovery request per audio and refresh before expiry.
        if NATIVE_PROJECT_CACHE and NATIVE_PROJECT_CACHE[1] > time.monotonic():
            token(min_remaining=120)
            return NATIVE_PROJECT_CACHE[0]
        token(min_remaining=120)
        project = discover()
        NATIVE_PROJECT_CACHE = (project, time.monotonic() + 900)
        return project


def process_native_api(job, audio, duration, prefix, variant, reference, model, client):
    global NATIVE_API_DISABLED_UNTIL, NATIVE_API_CONSECUTIVE_FAILURES, NATIVE_API_GOOD_COUNT, NATIVE_API_DEGRADED
    # A failure before generate has sent no audio. Once generate starts, any
    # uncertain outcome is retained for review and never replayed automatically.
    try:
        project = native_project()
    except Exception as error:
        finish(job, 'RETRYABLE_FAILED', error_code='native_api_predispatch_'+type(error).__name__,
               retry_seconds=120, block='transport_cooldown')
        return
    try:
        from .native_api import generate
        native, raw = generate(audio, project, variant, reference, model)
    except Exception as error:
        finish(job, 'NEEDS_REVIEW', error_code='native_api_ambiguous_'+type(error).__name__)
        return
    client.put_object(Bucket=bucket(), Key=f'{prefix}/native-api-result.json',
                      Body=json.dumps(native, ensure_ascii=False).encode())
    if raw:
        client.put_object(Bucket=bucket(), Key=f'{prefix}/native-api-response.json', Body=raw)
    if native.get('status') != 'complete':
        http_status = native.get('http_status')
        with NATIVE_API_FAILURE_LOCK:
            NATIVE_API_CONSECUTIVE_FAILURES += 1
            NATIVE_API_GOOD_COUNT = 0
            NATIVE_API_DEGRADED = True
            if (NATIVE_API_CONSECUTIVE_FAILURES >= 3 or http_status in (None, 401, 403, 429, 0)
                    or (http_status or 0) >= 500):
                NATIVE_API_DISABLED_UNTIL = max(NATIVE_API_DISABLED_UNTIL, time.monotonic() + 600)
        finish(job, 'NEEDS_REVIEW', error_code=native.get('error_code', 'native_api_result_rejected'))
        return
    with NATIVE_API_FAILURE_LOCK:
        NATIVE_API_CONSECUTIVE_FAILURES = 0
        NATIVE_API_GOOD_COUNT += 1
        good_count = NATIVE_API_GOOD_COUNT
        if good_count >= 25:
            NATIVE_API_DEGRADED = False
    if good_count in (25, 100):
        print(json.dumps({'event': 'native_api_route_ramped', 'account': job['account_id'],
                          'consecutive_valid': good_count, 'target_percent': 35 if good_count == 25 else 50}),
              flush=True)
    result = {'verbatim_text': native['verbatim_text'], 'language': native['language'],
              'language_raw': native['structured_output']['language'],
              'uncertain_spans': native['uncertain_spans'], 'native_result': native,
              'annotation_source': 'agy_native_api', 'transport': 'native_api_v1',
              'model': model, 'contract_sha256': native['contract_sha256'],
              'raw_prefix': prefix, 'audio_seconds': duration, 'human_gold': False,
              'prompt_variant': variant, 'inference_seconds': native['inference_seconds'],
              'source_conditioned': variant == 'source_correction',
              'source_text_sha256': digest(reference) if reference is not None else None}
    state = 'NEEDS_REVIEW'  # All current routed jobs are review-only.
    code = ('pilot_review_only' if result['language'] == 'ru' and result['verbatim_text'].strip()
            else 'empty_transcript' if not result['verbatim_text'].strip() else 'non_russian')
    finish(job, state, result, code)


def process(job):
    client = s3()
    prefix = f"annotations/raw/{job['id']}/{job['attempt']}"
    with tempfile.TemporaryDirectory(prefix='annotation-', dir='/work') as temp:
        workspace = Path(temp)
        audio = workspace / 'audio.wav'
        sample = job['sample']
        model = sample_model(sample)
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
        if (native_api_selected(job, duration, head['ContentLength'], variant, reference, model)
                and NATIVE_API_SLOTS.acquire(blocking=False)):
            try:
                process_native_api(job, audio, duration, prefix, variant, reference, model, client)
            finally:
                NATIVE_API_SLOTS.release()
            return
        tool_policy_sha256 = install_cli_hooks(workspace, os.environ.get('ACCOUNT_ID'))
        started = time.monotonic()
        rc, stdout, stderr, timed_out = run_agy(audio, workspace, variant, reference, model)
        elapsed = time.monotonic() - started
        # Preserve the failed startup before one bounded recovery. No init/tool/
        # generation event is allowed; ambiguous inference is never repeated.
        if not timed_out and predispatch_auth_timeout(stdout, stderr, elapsed):
            for suffix, body in (('stream.jsonl', stdout), ('stderr.log', stderr)):
                client.put_object(Bucket=bucket(), Key=f'{prefix}/auth-startup-failure/{suffix}', Body=body.encode())
            recovered = probe_auth(model)
            client.put_object(Bucket=bucket(), Key=f'{prefix}/auth-startup-failure/probe.json',
                              Body=json.dumps({'models_access_ok': recovered}).encode())
            if recovered:
                rc, stdout, stderr, timed_out = run_agy(audio, workspace, variant, reference, model)
                elapsed = time.monotonic() - started
            if not recovered or (not timed_out and predispatch_auth_timeout(stdout, stderr, elapsed)):
                # A failed read-only probe is inconclusive during a network/auth
                # outage. Retry this zero-turn job later and cool down the slot;
                # only explicit access-denial responses block the account.
                finish(job, 'RETRYABLE_FAILED', error_code='auth_startup_transport_transient',
                       retry_seconds=random.uniform(30, 90) * 2 ** min(job['attempt'] - 1, 2),
                       block='transport_cooldown')
                return
        # Native raw output can contain transcript/PII: private S3, never stdout.
        for suffix, body in (('stream.jsonl', stdout), ('stderr.log', stderr),
                             ('request.json', json.dumps({'model': model, 'prompt': prompt(audio, variant, reference),
                              'schema': SCHEMA, 'contract_sha256': contract_sha(variant, model),
                              'cli_sha256': os.environ.get('AGY_EXPECTED_SHA256'),
                              'tool_policy_sha256': tool_policy_sha256,
                              'audio_sha256': sample['audio_sha256'], 'exit_code': rc}))):
            client.put_object(Bucket=bucket(), Key=f'{prefix}/{suffix}', Body=body.encode())
        try:
            if rc != 0 or timed_out:
                raise ValueError('native_exit')
            native = agy_result(stdout, audio, expected_model=model)
        except (ValueError, KeyError, TypeError):
            if stdout == json.dumps({'event': 'result', 'result': {'status': 'ERROR',
                    'num_turns': 0, 'duration_seconds': 0,
                    'error': 'auth_storage_invalid_before_dispatch'}}):
                finish(job, 'RETRYABLE_FAILED', error_code='auth_storage_invalid',
                       retry_seconds=60, block='transport_cooldown')
                return
            if predispatch_model_catalog_transient(stdout, stderr, model, rc, timed_out):
                finish(job, 'RETRYABLE_FAILED', error_code='model_catalog_unavailable_before_dispatch',
                       retry_seconds=random.uniform(30, 90) * 2 ** min(job['attempt'] - 1, 2),
                       block='transport_cooldown')
                return
            if predispatch_eligibility_transient(stdout, timed_out):
                # Existing queue caps attempts at three and cools down the account.
                # Some CLI versions return exit code 0 for zero-turn failures.
                # Raw artifacts above retain every attempt; no sleeping worker slot.
                finish(job, 'RETRYABLE_FAILED', error_code='eligibility_transport_transient',
                       retry_seconds=random.uniform(30, 90) * 2 ** min(job['attempt'] - 1, 2))
                return
            state, code, block = classify(stdout + stderr, timed_out)
            delay = quota_reset_delay(stdout + stderr) if block == 'quota_exhausted' else None
            if block == 'quota_exhausted' and delay is not None:
                block = 'quota_cooldown'
            elif block == 'transport_cooldown':
                delay = random.uniform(30, 90)
            finish(job, state, error_code=code, retry_seconds=delay or 300 * job['attempt'],
                   block=block)
            return
        out = native['structured_output']
        result = {'verbatim_text': out['transcript'], 'language': normalize_language(out['language']), 'language_raw': out['language'],
                  'uncertain_spans': out['uncertain_fragments'], 'native_result': native,
                  'annotation_source': 'agy', 'model': model, 'contract_sha256': contract_sha(variant, model),
                  'cli_sha256': os.environ.get('AGY_EXPECTED_SHA256'),
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
    print(json.dumps({'event': 'cli_global_tool_policy', 'account': account,
                      **install_global_cli_hooks()}), flush=True)
    if account == 'agy-03' and os.environ.get('NATIVE_SHADOW_ENABLED', 'true') == 'true':
        from .native_shadow_queue import loop as shadow_loop
        threading.Thread(target=shadow_loop, args=(STOP,), daemon=True,
                         name='native-shadow').start()
    workers = int(os.environ.get('WORKER_CONCURRENCY', '2'))
    if not 1 <= workers <= 64:
        raise ValueError('invalid_concurrency')
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda _: loop(account), range(workers)))


if __name__ == '__main__':
    main()
