"""Error handling must not turn failed native results into duplicate requests."""
import unittest
from lalm.annotation.worker import classify, quota_reset_delay


class WorkerErrorTests(unittest.TestCase):
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
