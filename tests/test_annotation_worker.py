"""Error handling must not turn failed native results into duplicate requests."""
import unittest
from lalm.annotation.worker import classify, quota_reset_delay, predispatch_auth_timeout, probe_auth, predispatch_eligibility_transient


class WorkerErrorTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        self.enterContext(patch('lalm.annotation.worker.repair_trailing_auth', return_value=False))

    def test_native_route_is_bounded_to_review_only_short_flash_audio(self):
        from unittest.mock import patch
        from lalm.annotation.worker import native_api_selected
        job = {'id': '0'*64, 'sample': {'review_only': True}}
        with patch.dict('os.environ', {'ACCOUNT_ID': 'agy-03', 'NATIVE_API_PERCENT': '20'}), \
             patch('lalm.annotation.worker.NATIVE_API_DISABLED_UNTIL', 0):
            self.assertTrue(native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
            self.assertFalse(native_api_selected(job, 21, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
            self.assertFalse(native_api_selected(job, 4, 2*1024**2, 'verbatim', None, 'gemini-3.8-flash-low'))
            self.assertFalse(native_api_selected(job, 4, 1000, 'source_correction', None, 'gemini-3.8-flash-low'))
            self.assertFalse(native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-4-argon-low'))
            job['sample']['review_only'] = False
            self.assertFalse(native_api_selected(job, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
        job['sample']['review_only'] = True
        with patch.dict('os.environ', {'ACCOUNT_ID': 'agy-16', 'NATIVE_API_PERCENT': '20'}), \
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
        with patch.dict('os.environ', {'ACCOUNT_ID': 'agy-03', 'NATIVE_API_PERCENT': '20'}), \
             patch.object(worker, 'NATIVE_API_DISABLED_UNTIL', 0), \
             patch.object(worker, 'NATIVE_API_GOOD_COUNT', 0):
            self.assertFalse(worker.native_api_selected(thirty, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
            worker.NATIVE_API_GOOD_COUNT = 25
            self.assertTrue(worker.native_api_selected(thirty, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
            self.assertFalse(worker.native_api_selected(forty_five, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))
            worker.NATIVE_API_GOOD_COUNT = 100
            self.assertTrue(worker.native_api_selected(forty_five, 4, 1000, 'verbatim', None, 'gemini-3.8-flash-low'))

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
            self.assertEqual(classify(text), ('RETRYABLE_FAILED', 'rate_limited', None))

    def test_quota_and_auth_stay_distinct(self):
        self.assertEqual(classify('Individual quota reached. Resets in 2h10m6s.')[2], 'quota_exhausted')
        self.assertEqual(classify('Authentication required')[2], 'authentication_or_access_denied')

    def test_only_explicit_bounded_reset_time_resumes_quota(self):
        self.assertEqual(quota_reset_delay('Resets in 2h10m6s.'), 7866)
        for text in ('quota exhausted', 'Resets in 0s.', 'Resets in 999h.'):
            self.assertIsNone(quota_reset_delay(text))


if __name__ == '__main__':
    unittest.main()
