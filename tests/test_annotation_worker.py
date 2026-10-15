"""Error handling must not turn failed native results into duplicate requests."""
import unittest
from lalm.annotation.worker import classify, quota_reset_delay, predispatch_auth_timeout, probe_auth, predispatch_eligibility_transient


class WorkerErrorTests(unittest.TestCase):
    def test_model_catalog_retry_requires_observed_zero_turn_failure(self):
        import json
        from lalm.annotation.worker import predispatch_model_catalog_transient
        model = 'gemini-3.8-flash-low'
        stderr = (f'error: invalid model selection (--model "{model}" --effort ""): '
                  f'model {model} is not recognized as a known model or custom model in settings\n')
        event = {'event': 'result', 'result': {'status': 'ERROR', 'num_turns': 0,
                                              'duration_seconds': 0}}
        raw = json.dumps(event)
        self.assertTrue(predispatch_model_catalog_transient(raw, stderr, model, 1))
        for bad in ('', 'not json', raw+'\n'+json.dumps({'event': 'init'})):
            self.assertFalse(predispatch_model_catalog_transient(bad, stderr, model, 1))
        self.assertFalse(predispatch_model_catalog_transient(raw, stderr, model, 1, True))
        self.assertFalse(predispatch_model_catalog_transient(raw, stderr, model, 0))
        self.assertFalse(predispatch_model_catalog_transient(raw, stderr, 'gemini-4-argon-low', 1))
        event['result']['num_turns'] = 1
        self.assertFalse(predispatch_model_catalog_transient(json.dumps(event), stderr, model, 1))

    def setUp(self):
        from unittest.mock import patch
        self.enterContext(patch('lalm.annotation.worker.repair_trailing_auth', return_value=False))
        self.enterContext(patch('lalm.annotation.worker.NATIVE_NETWORK_BACKOFF_STEP', 0))
        self.enterContext(patch('lalm.annotation.worker.NATIVE_NETWORK_LAST_FAILURE', 0.0))

    def test_native_api_concurrency_has_hard_upper_bound(self):
        from unittest.mock import patch
        from lalm.annotation.worker import native_api_concurrency
        with patch.dict('os.environ', {'NATIVE_API_CONCURRENCY': '8'}):
            self.assertEqual(native_api_concurrency(), 8)
        with patch.dict('os.environ', {'NATIVE_API_CONCURRENCY': '100'}):
            self.assertEqual(native_api_concurrency(), 8)

    def test_cli_limit_bounds_fallback_and_releases_after_failure(self):
        import concurrent.futures
        import threading
        import time
        from unittest.mock import patch
        import lalm.annotation.worker as worker
        lock = threading.Lock()
        active = peak = 0
        slots = threading.BoundedSemaphore(2)

        def dispatch(audio, *args):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(.02)
                if audio == 'bad':
                    raise OSError('dispatch failed')
                return audio
            finally:
                with lock:
                    active -= 1

        with patch.object(worker, 'CLI_SLOTS', slots), \
             patch.object(worker, '_run_agy', side_effect=dispatch):
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(worker.run_agy, audio, '/work')
                           for audio in ['bad', 'a', 'b', 'c', 'd', 'e']]
                with self.assertRaises(OSError):
                    futures[0].result(timeout=2)
                self.assertEqual([f.result(timeout=2) for f in futures[1:]],
                                 ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(peak, 2)
        self.assertTrue(slots.acquire(blocking=False))
        self.assertTrue(slots.acquire(blocking=False))
        self.assertFalse(slots.acquire(blocking=False))
        slots.release()
        slots.release()

    def test_cli_wait_renews_lease_and_stale_work_never_dispatches(self):
        import threading
        from unittest.mock import Mock, patch, call
        import lalm.annotation.worker as worker
        job = {'id': 'job', 'lease_token': 'lease'}
        slots = Mock()
        slots.acquire.side_effect = [False, True]
        with patch.object(worker, 'CLI_SLOTS', slots), \
             patch.object(worker, 'renew_lease') as renew, \
             patch.object(worker, '_run_agy', return_value='ok') as dispatch:
            self.assertEqual(worker.run_agy('audio', '/work', job=job), 'ok')
            self.assertEqual(renew.call_args_list, [call(job), call(job)])
            dispatch.assert_called_once()
            slots.release.assert_called_once()
        slots = threading.BoundedSemaphore(1)
        with patch.object(worker, 'CLI_SLOTS', slots), \
             patch.object(worker, 'renew_lease', side_effect=RuntimeError('stale_lease')), \
             patch.object(worker, '_run_agy') as dispatch:
            with self.assertRaisesRegex(RuntimeError, 'stale_lease'):
                worker.run_agy('audio', '/work', job=job)
            dispatch.assert_not_called()
        self.assertTrue(slots.acquire(blocking=False))
        slots.release()

    def test_native_route_is_bounded_to_review_only_short_flash_audio(self):
        from unittest.mock import patch
        from lalm.annotation.worker import native_api_selected
        job = {'id': '0'*64, 'sample': {'review_only': True}}
        with patch.dict('os.environ', {'ACCOUNT_ID': 'agy-03', 'NATIVE_API_PERCENT': '20',
                                      'NATIVE_API_MAX_PERCENT': '50'}), \
             patch('lalm.annotation.worker.NATIVE_API_DISABLED_UNTIL', 0):
            self.assertTrue(native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
            self.assertFalse(native_api_selected(job, 21, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
            self.assertFalse(native_api_selected(job, 4, 2*1024**2, 'verbatim', None, 'gemini-3.8-flash-low'))
            self.assertFalse(native_api_selected(job, 4, 1000, 'source_correction', None, 'gemini-3.8-flash-low'))
            self.assertFalse(native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-4-argon-low'))
            job['sample']['review_only'] = False
            self.assertFalse(native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
        job['sample']['review_only'] = True
        for account in ('agy-05', 'agy-10', 'agy-11', 'agy-14'):
            with patch.dict('os.environ', {'ACCOUNT_ID': account, 'NATIVE_API_PERCENT': '100',
                                          'NATIVE_API_MAX_PERCENT': '100'}), \
                 patch('lalm.annotation.worker.NATIVE_API_DISABLED_UNTIL', 0):
                self.assertTrue(native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
        with patch.dict('os.environ', {'ACCOUNT_ID': 'agy-16', 'NATIVE_API_PERCENT': '20'}), \
             patch('lalm.annotation.worker.NATIVE_API_DISABLED_UNTIL', 0):
            self.assertTrue(native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
        with patch.dict('os.environ', {'ACCOUNT_ID': 'agy-21', 'NATIVE_API_PERCENT': '20'}), \
             patch('lalm.annotation.worker.NATIVE_API_DISABLED_UNTIL', 0):
            self.assertTrue(native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
        with patch.dict('os.environ', {'ACCOUNT_ID': 'agy-08', 'NATIVE_API_PERCENT': '20'}), \
             patch('lalm.annotation.worker.NATIVE_API_DISABLED_UNTIL', 0):
            self.assertTrue(native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
        with patch.dict('os.environ', {'ACCOUNT_ID': 'agy-25', 'NATIVE_API_PERCENT': '20'}), \
             patch('lalm.annotation.worker.NATIVE_API_DISABLED_UNTIL', 0):
            self.assertTrue(native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
        with patch.dict('os.environ', {'ACCOUNT_ID': 'agy-19', 'NATIVE_API_PERCENT': '20',
                                      'NATIVE_API_MAX_PERCENT': '20'}), \
             patch('lalm.annotation.worker.NATIVE_API_DISABLED_UNTIL', 0):
            self.assertTrue(native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))

    def test_dispatched_native_api_failure_is_quarantined_without_replay(self):
        from pathlib import Path
        from unittest.mock import Mock, patch
        from lalm.annotation.worker import process_native_api
        client = Mock()
        failed = {'status': 'failed_do_not_auto_retry', 'http_status': 503,
                  'error_code': 'native_generation_http_503'}
        with patch('lalm.annotation.worker.native_project', return_value='project'), \
             patch('lalm.annotation.native_api.generate', return_value=(failed, b'error')) as generate, \
             patch('lalm.annotation.worker.finish') as finish:
            process_native_api({'id': 'a'}, Path('audio.wav'), 4, 'raw/a/1',
                               'verbatim', None, 'gemini-3.8-flash-low', client)
        generate.assert_called_once()
        self.assertEqual(finish.call_args.args[1:], ('NEEDS_REVIEW',))
        self.assertEqual(finish.call_args.kwargs['error_code'], 'native_generation_http_503')

    def test_native_transport_drop_suspends_route_without_replay(self):
        from pathlib import Path
        from unittest.mock import Mock, patch
        import lalm.annotation.worker as worker
        failed = {'status': 'failed_do_not_auto_retry', 'http_status': None,
                  'error_type': 'RemoteDisconnected'}
        with patch.object(worker, 'NATIVE_API_DISABLED_UNTIL', 0), \
             patch.object(worker, 'NATIVE_API_CONSECUTIVE_FAILURES', 0), \
             patch.object(worker, 'NATIVE_API_GOOD_COUNT', 0), \
             patch.object(worker, 'NATIVE_API_DEGRADED', False), \
             patch('lalm.annotation.worker.time.monotonic', return_value=1000), \
             patch('lalm.annotation.worker.native_project', return_value='project'), \
             patch('lalm.annotation.native_api.generate', return_value=(failed, b'')) as generate, \
             patch('lalm.annotation.worker.finish') as finish:
            worker.process_native_api({'id': 'a'}, Path('audio.wav'), 4, 'raw/a/1',
                                      'verbatim', None, 'gemini-3.8-flash-low', Mock())
            self.assertEqual(worker.NATIVE_API_DISABLED_UNTIL, 1060)
            generate.assert_called_once()
            self.assertEqual(finish.call_args.args[1], 'NEEDS_REVIEW')
            generate.reset_mock()
            worker.NATIVE_API_DISABLED_UNTIL = 1500
            worker.process_native_api({'id': 'b'}, Path('audio.wav'), 4, 'raw/b/1',
                                      'verbatim', None, 'gemini-3.8-flash-low', Mock())
            self.assertEqual(worker.NATIVE_API_DISABLED_UNTIL, 1500)
            generate.assert_called_once()

    def test_network_pause_grows_on_recurrence_and_coalesces_inflight_failures(self):
        from unittest.mock import patch
        import lalm.annotation.worker as worker
        native = {'http_status': None, 'error_type': 'RemoteDisconnected'}
        with patch.object(worker, 'NATIVE_API_DISABLED_UNTIL', 0):
            now = 1000
            for expected in (60, 120, 240, 480, 600, 600):
                self.assertEqual(worker.native_api_cooldown(native, now), expected)
                worker.NATIVE_API_DISABLED_UNTIL = now + expected
                self.assertEqual(worker.native_api_cooldown(native, now + 1), expected)
                now += expected + 2
            self.assertEqual(worker.native_api_cooldown(native, now + 901), 60)

    def test_http_auth_rate_and_unknown_failures_keep_long_pause(self):
        from lalm.annotation.worker import native_api_cooldown
        for status in (200, 401, 403, 429, 500, 503, 0):
            self.assertEqual(native_api_cooldown(
                {'http_status': status, 'error_type': 'RemoteDisconnected'}, 1000), 600)
        self.assertEqual(native_api_cooldown(
            {'http_status': None, 'error_type': 'UnknownError'}, 1000), 600)

    def test_three_bad_native_responses_suspend_new_api_routes(self):
        from pathlib import Path
        from unittest.mock import Mock, patch
        import lalm.annotation.worker as worker
        failed = {'status': 'failed_do_not_auto_retry', 'http_status': 200}
        with patch.object(worker, 'NATIVE_API_CONSECUTIVE_FAILURES', 0), \
             patch.object(worker, 'NATIVE_API_DISABLED_UNTIL', 0), \
             patch('lalm.annotation.worker.native_project', return_value='project'), \
             patch('lalm.annotation.native_api.generate', return_value=(failed, b'bad')), \
             patch('lalm.annotation.worker.finish'):
            for _ in range(3):
                worker.process_native_api({'id': 'a'}, Path('audio.wav'), 4, 'raw/a/1',
                                          'verbatim', None, 'gemini-3.8-flash-low', Mock())
            self.assertEqual(worker.NATIVE_API_CONSECUTIVE_FAILURES, 3)
            self.assertGreater(worker.NATIVE_API_DISABLED_UNTIL, worker.time.monotonic())

    def test_valid_native_responses_raise_route_only_at_thresholds(self):
        from unittest.mock import patch
        import lalm.annotation.worker as worker
        sample = {'review_only': True}
        thirty = {'id': '0000001e'+'0'*56, 'sample': sample}
        forty_five = {'id': '0000002d'+'0'*56, 'sample': sample}
        with patch.dict('os.environ', {'ACCOUNT_ID': 'agy-03', 'NATIVE_API_PERCENT': '20',
                                      'NATIVE_API_MAX_PERCENT': '50'}), \
             patch.object(worker, 'NATIVE_API_DISABLED_UNTIL', 0), \
             patch.object(worker, 'NATIVE_API_GOOD_COUNT', 0), \
             patch.object(worker, 'NATIVE_API_DEGRADED', False):
            self.assertFalse(worker.native_api_selected(thirty, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
            worker.NATIVE_API_GOOD_COUNT = 25
            self.assertTrue(worker.native_api_selected(thirty, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
            self.assertFalse(worker.native_api_selected(forty_five, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
            worker.NATIVE_API_GOOD_COUNT = 100
            self.assertTrue(worker.native_api_selected(forty_five, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))

    def test_native_route_respects_account_ceiling_after_ramp(self):
        from unittest.mock import patch
        import lalm.annotation.worker as worker
        twenty_five = {'id': '00000019'+'0'*56, 'sample': {'review_only': True}}
        with patch.dict('os.environ', {'ACCOUNT_ID': 'agy-19', 'NATIVE_API_PERCENT': '20',
                                      'NATIVE_API_MAX_PERCENT': '20'}), \
             patch.object(worker, 'NATIVE_API_DISABLED_UNTIL', 0), \
             patch.object(worker, 'NATIVE_API_GOOD_COUNT', 100):
            self.assertFalse(worker.native_api_selected(twenty_five, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))

    def test_native_route_can_use_higher_verified_account_share(self):
        from unittest.mock import patch
        import lalm.annotation.worker as worker
        sixty_five = {'id': '00000041'+'0'*56, 'sample': {'review_only': True}}
        with patch.dict('os.environ', {'ACCOUNT_ID': 'agy-08', 'NATIVE_API_PERCENT': '70',
                                      'NATIVE_API_MAX_PERCENT': '70'}), \
             patch.object(worker, 'NATIVE_API_DISABLED_UNTIL', 0), \
             patch.object(worker, 'NATIVE_API_DEGRADED', False), \
             patch.object(worker, 'NATIVE_API_GOOD_COUNT', 100):
            self.assertTrue(worker.native_api_selected(sixty_five, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))

    def test_stable_restart_floor_falls_back_after_bad_api_response(self):
        from unittest.mock import patch
        import lalm.annotation.worker as worker
        job = {'id': '0000002d'+'0'*56, 'sample': {'review_only': True}}
        with patch.dict('os.environ', {'ACCOUNT_ID': 'agy-03'}, clear=True), \
             patch.object(worker, 'NATIVE_API_DISABLED_UNTIL', 0), \
             patch.object(worker, 'NATIVE_API_GOOD_COUNT', 0), \
             patch.object(worker, 'NATIVE_API_DEGRADED', False):
            self.assertTrue(worker.native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
            worker.NATIVE_API_DEGRADED = True
            self.assertFalse(worker.native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))

    def test_invalid_local_auth_never_starts_model_process(self):
        import json
        from pathlib import Path
        from unittest.mock import patch
        from lalm.annotation.worker import run_agy
        with patch('lalm.annotation.worker.repair_trailing_auth', side_effect=ValueError), \
             patch('lalm.annotation.worker.subprocess.Popen') as spawn:
            rc, stdout, _, timed_out = run_agy(Path('audio.wav'), Path('/work'))
        spawn.assert_not_called()
        self.assertEqual(rc, 1)
        self.assertFalse(timed_out)
        self.assertEqual(json.loads(stdout)['result']['num_turns'], 0)

    def test_eligibility_500_is_not_an_auth_denial_or_an_inference_replay(self):
        import json
        error = ('Eligibility check failed: failed to get load code assist response: '
                 'INTERNAL (code 500): Internal error encountered.')
        raw = json.dumps({'event': 'result', 'result': {'status': 'ERROR',
                         'num_turns': 0, 'duration_seconds': 0, 'error': error}})
        self.assertTrue(predispatch_eligibility_transient(raw))
        self.assertFalse(predispatch_eligibility_transient(raw, True))
        initialized = json.dumps({'event': 'init'})+'\n'+raw
        self.assertFalse(predispatch_eligibility_transient(initialized))
        self.assertEqual(classify(initialized),
                         ('NEEDS_REVIEW', 'eligibility_service_unavailable', 'transport_cooldown'))
        self.assertEqual(classify(initialized, True), ('NEEDS_REVIEW', 'ambiguous_timeout', None))
        for denial in ('Authentication required', '403 Forbidden', 'Verify your account'):
            self.assertEqual(classify(initialized+'\n'+denial)[2], 'authentication_or_access_denied')
        model_text = json.dumps({'event': 'step_update', 'step_update':
                                {'step_type': 'agent_response', 'text_delta': error}})
        self.assertEqual(classify(model_text), ('NEEDS_REVIEW', 'native_result_rejected', None))

    def test_generation_service_errors_cool_account_without_replay(self):
        import json
        for message in (
            'API error (attempt 3): INTERNAL (code 500): Internal error encountered.',
            'API error (attempt 7): UNAVAILABLE (code 503): No capacity available for model gemini-3.8-flash-low on the server',
            'The stream was interrupted. Please continue the task you were working on.',
        ):
            event = {'event': 'result', 'result': {'status': 'ERROR', 'num_turns': 1, 'error': message}}
            raw = json.dumps({'event': 'init'})+'\n'+json.dumps(event)
            self.assertFalse(predispatch_eligibility_transient(raw))
            self.assertEqual(classify(raw),
                             ('NEEDS_REVIEW', 'inference_service_unavailable', 'transport_cooldown'))
            self.assertEqual(classify(raw, True), ('NEEDS_REVIEW', 'ambiguous_timeout', None))
            event['result']['status'] = 'SUCCESS'
            self.assertEqual(classify(json.dumps(event)), ('NEEDS_REVIEW', 'native_result_rejected', None))
            self.assertEqual(classify(message), ('NEEDS_REVIEW', 'native_result_rejected', None))

    def test_generation_cooldown_cannot_hide_real_auth_failure(self):
        import json
        message = 'API error (attempt 3): INTERNAL (code 500): Internal error encountered.'
        raw = json.dumps({'event': 'result', 'result': {'status': 'ERROR', 'num_turns': 1, 'error': message}})
        self.assertEqual(classify(raw+'\nAuthentication required')[2], 'authentication_or_access_denied')

    def test_service_503_after_init_quarantines_job_and_cools_account(self):
        import json
        error = ('failed to send message: send failed; already reported to the user: '
                 'Eligibility check failed: failed to get load code assist response: '
                 'UNAVAILABLE (code 503): The service is currently unavailable.')
        event = {'event': 'result', 'result': {'status': 'ERROR', 'num_turns': 0,
                 'duration_seconds': 0, 'error': error}}
        raw = json.dumps({'event': 'init'})+'\n'+json.dumps(event)
        self.assertFalse(predispatch_eligibility_transient(raw))
        self.assertEqual(classify(raw),
                         ('NEEDS_REVIEW', 'eligibility_service_unavailable', 'transport_cooldown'))
        self.assertEqual(classify(raw, True), ('NEEDS_REVIEW', 'ambiguous_timeout', None))
        event['result']['error'] = error.replace('failed to get load code assist response: ', '')
        self.assertEqual(classify(json.dumps(event)),
                         ('NEEDS_REVIEW', 'eligibility_service_unavailable', 'transport_cooldown'))
        for message in ('Eligibility check failed: Verify your account',
                        'Eligibility check failed: 403 Forbidden'):
            event['result']['error'] = message
            self.assertEqual(classify(json.dumps(event))[2], 'authentication_or_access_denied')

    def test_profile_failures_retry_only_before_generation(self):
        import json
        for error in (
            'Eligibility check failed: request failed (code 502): HTML response "Error 502 (Server Error)!!1"',
            'Eligibility check failed: failed to get load code assist response: Post "https://example.test/check": EOF',
            'Eligibility check failed: UNAVAILABLE (code 503): The service is currently unavailable.',
            'Eligibility check failed: failed to get profile picture: Get "https://example.test/avatar": net/http: TLS handshake timeout',
            'Eligibility check failed: failed to get profile picture: Get "https://example.test/avatar": unexpected EOF',
            'Eligibility check failed: failed to fetch user info: 500 Internal Server Error',
        ):
            event = {'event': 'result', 'result': {'status': 'ERROR', 'num_turns': 0,
                     'duration_seconds': 0, 'error': error}}
            raw = json.dumps(event)
            self.assertTrue(predispatch_eligibility_transient(raw))
            self.assertFalse(predispatch_eligibility_transient(raw, True))
            self.assertFalse(predispatch_eligibility_transient(raw+'\n'+json.dumps({'event':'init'})))
            event['result']['num_turns'] = 1
            self.assertFalse(predispatch_eligibility_transient(json.dumps(event)))
        for error in (
            'Eligibility check failed: failed to fetch user info: 401 Unauthorized',
            'Eligibility check failed: failed to fetch user info: 403 Forbidden',
            'Eligibility check failed: failed to get profile picture: Get "https://example.test/avatar": 403 Forbidden',
        ):
            event['result'].update(num_turns=0, error=error)
            self.assertFalse(predispatch_eligibility_transient(json.dumps(event)))

    def test_eligibility_eof_requires_zero_activity_and_exact_transport_error(self):
        import json
        event = {'event': 'result', 'result': {'status': 'ERROR', 'num_turns': 0,
                 'duration_seconds': 0, 'error': 'Eligibility check failed: Post "https://example.test/check": EOF'}}
        raw = json.dumps(event)
        self.assertTrue(predispatch_eligibility_transient(raw))
        self.assertFalse(predispatch_eligibility_transient(raw, True))
        for other in ('', 'null', 'garbage', raw+'\n'+json.dumps({'event':'init'})):
            self.assertFalse(predispatch_eligibility_transient(other))
        for error in ('Eligibility check failed: Verify your account', 'Authentication required',
                      'Eligibility check failed: Post "https://example.test/check": 403 Forbidden',
                      'Individual quota reached', 'EOF'):
            event['result']['error'] = error
            self.assertFalse(predispatch_eligibility_transient(json.dumps(event)))
        event['result']['error'] = 'Eligibility check failed: Post "https://example.test/check": EOF'
        event['result']['error'] = ('Eligibility check failed: failed to get load code assist response: '
                                    'UNAVAILABLE (code 503): The service is currently unavailable.')
        self.assertTrue(predispatch_eligibility_transient(json.dumps(event)))
        event['result']['num_turns'] = 1
        self.assertFalse(predispatch_eligibility_transient(json.dumps(event)))

    def test_startup_retry_excludes_any_inference_or_ambiguous_output(self):
        import json
        error = {'event': 'result', 'result': {'status': 'ERROR', 'num_turns': 0,
                 'duration_seconds': 0, 'error': 'authentication failed or timed out'}}
        raw = json.dumps(error)
        stderr = 'Authentication required. Error: authentication timed out.'
        self.assertTrue(predispatch_auth_timeout(raw, stderr, 65))
        for other in ('', 'garbage', 'null', raw+'\n'+json.dumps({'event':'init'}),
                      json.dumps({'event':'result','result':None})):
            self.assertFalse(predispatch_auth_timeout(other, stderr, 65))
        self.assertTrue(predispatch_auth_timeout(raw, stderr, 134))
        self.assertFalse(predispatch_auth_timeout(raw, stderr, 180))
        self.assertFalse(predispatch_auth_timeout(raw, 'Verify your account', 65))
        error['result']['num_turns'] = 1
        self.assertFalse(predispatch_auth_timeout(json.dumps(error), stderr, 65))

    def test_probe_requires_success_and_expected_model(self):
        from unittest.mock import patch
        from subprocess import CompletedProcess, TimeoutExpired
        from lalm.annotation.contract import MODEL
        for rc, text, expected in [(0, MODEL, True), (1, MODEL, False), (0, '', False)]:
            with patch('lalm.annotation.worker.subprocess.run', return_value=CompletedProcess([],rc,text,'')):
                self.assertEqual(probe_auth(ttl_seconds=0), expected)
        with patch('lalm.annotation.worker.subprocess.run', side_effect=TimeoutExpired('agy',40)):
            self.assertFalse(probe_auth(ttl_seconds=0))

    def test_concurrent_startup_failures_share_one_failed_catalog(self):
        from concurrent.futures import ThreadPoolExecutor
        from subprocess import TimeoutExpired
        from unittest.mock import patch
        import time
        def unavailable(*args, **kwargs):
            time.sleep(.03)
            raise TimeoutExpired('agy', 40)
        with patch('lalm.annotation.worker.AUTH_PROBE_CACHE', {}), \
             patch('lalm.annotation.worker.subprocess.run', side_effect=unavailable) as catalog:
            with ThreadPoolExecutor(max_workers=20) as pool:
                self.assertEqual(list(pool.map(lambda _: probe_auth(), range(20))), [False]*20)
            self.assertEqual(catalog.call_count, 1)

    def test_verification_required_blocks_account(self):
        self.assertEqual(classify('Eligibility check failed: Verify your account')[2],
                         'authentication_or_access_denied')

    def test_model_text_cannot_block_an_account(self):
        import json
        for text in ('forbidden', 'Authentication required', 'quota exhausted', 'rate limit'):
            raw = '\n'.join((json.dumps({'event': 'step_update', 'step_update':
                {'step_type': 'agent_response', 'text_delta': text}}),
                json.dumps({'event': 'result', 'result': {'status': 'SUCCESS',
                    'num_turns': 2, 'structured_output': {'transcript': text}}})))
            self.assertEqual(classify(raw), ('NEEDS_REVIEW', 'native_result_rejected', None))
        self.assertEqual(classify('{"event":"result","result":{"status":"ERROR",'
            '"error":"Authentication required"}}')[2], 'authentication_or_access_denied')

    def test_conversation_id_digits_are_not_a_rate_limit(self):
        raw = '{"status":"SUCCESS","num_turns":2,"conversation_id":"abc429ef-1234"}'
        self.assertEqual(classify(raw), ('NEEDS_REVIEW', 'native_result_rejected', None))
        self.assertEqual(classify('{"transcript":"429"}')[0], 'NEEDS_REVIEW')

    def test_explicit_service_rate_error_keeps_backoff(self):
        for text in ('RESOURCE_EXHAUSTED (code 429)', '{"error_code":429}', 'Too many requests'):
            self.assertEqual(classify(text),
                ('NEEDS_REVIEW', 'ambiguous_rate_limit_after_dispatch', 'transport_cooldown'))

    def test_429_after_audio_and_structured_generation_never_retries(self):
        import json
        events = [
            {'event': 'init', 'init': {'model': 'gemini-3.8-flash-low'}},
            {'event': 'step_update', 'step_update': {
                'step_type': 'tool', 'tool_name': 'view_file', 'state': 'DONE'}},
            {'event': 'result', 'result': {'status': 'ERROR', 'num_turns': 1,
                'error': 'RESOURCE_EXHAUSTED (code 429)',
                'structured_output': {'transcript': 'test'}}}]
        raw = '\n'.join(json.dumps(e) for e in events)
        self.assertEqual(classify(raw),
            ('NEEDS_REVIEW', 'ambiguous_rate_limit_after_dispatch', 'transport_cooldown'))

    def test_zero_turn_429_alone_does_not_prove_no_dispatch(self):
        import json
        raw = json.dumps({'event': 'result', 'result': {'status': 'ERROR',
            'num_turns': 0, 'duration_seconds': 0, 'error': 'Too many requests'}})
        self.assertEqual(classify(raw),
            ('NEEDS_REVIEW', 'ambiguous_rate_limit_after_dispatch', 'transport_cooldown'))

    def test_quota_and_auth_stay_distinct(self):
        self.assertEqual(classify('Individual quota reached. Resets in 2h10m6s.')[2], 'quota_exhausted')
        self.assertEqual(classify('Authentication required')[2], 'authentication_or_access_denied')

    def test_only_explicit_bounded_reset_time_resumes_quota(self):
        self.assertEqual(quota_reset_delay('Resets in 2h10m6s.'), 7866)
        for text in ('quota exhausted', 'Resets in 0s.', 'Resets in 999h.'):
            self.assertIsNone(quota_reset_delay(text))


if __name__ == '__main__':
    unittest.main()
