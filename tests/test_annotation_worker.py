"""Error handling must not turn failed native results into duplicate requests."""
import unittest
from lalm.annotation.worker import classify, quota_reset_delay, predispatch_auth_timeout, probe_auth, predispatch_eligibility_transient


class WorkerErrorTests(unittest.TestCase):
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
                self.assertEqual(probe_auth(), expected)
        with patch('lalm.annotation.worker.subprocess.run', side_effect=TimeoutExpired('agy',40)):
            self.assertFalse(probe_auth())

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
