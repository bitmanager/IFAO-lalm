import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from infra.annotation import check_usage


class UsageTests(unittest.TestCase):
    def test_gemini_zero_is_not_replaced_by_other_models_quota(self):
        text = ('GEMINI MODELS\nWeekly Limit Remaining\n[░░] 0.00%\n'
                'Refreshes in 143h 48m\nCLAUDE AND GPT MODELS\n'
                'Weekly Limit Remaining\n[██] 100.00%\nQuota available\n'
                'Five Hour Limit Remaining\n[██] 100.00%\nQuota available')
        self.assertEqual(check_usage.quota_fields(text),
                         {'weekly_pct': 0.0, 'weekly_reset': '143h 48m'})

    def test_paid_plan_has_both_windows(self):
        text = ('Weekly Limit Remaining\n[██] 50.45%\nRefreshes in 155h 57m\n'
                'Five Hour Limit Remaining\n[██] 65.76%\nRefreshes in 3h 40m')
        self.assertEqual(check_usage.quota_fields(text),
                         {'weekly_pct': 50.45, 'weekly_reset': '155h 57m',
                          'five_hour_pct': 65.76, 'five_hour_reset': '3h 40m'})

    def test_discovery_result_excludes_unexpected_sensitive_fields(self):
        args = SimpleNamespace(remote_dir='/code', ssh='/ssh', host='dev-1')
        response = SimpleNamespace(returncode=0, stdout=json.dumps(
            {'http_status': 200, 'id': 'free-tier', 'name': 'Starter',
             'access_token': 'SECRET', 'privacyNotice': 'PRIVATE'}))
        with patch.object(check_usage, 'ARGS', args, create=True), \
             patch.object(check_usage.subprocess, 'run', return_value=response) as run:
            result = check_usage.check_plan('agy-11')
        self.assertEqual(result['paid_tier_id'], 'free-tier')
        self.assertEqual(result['paid_tier_http_status'], 200)
        self.assertNotIn('SECRET', json.dumps(result))
        self.assertNotIn('PRIVATE', json.dumps(result))
        self.assertIn('loadCodeAssist', run.call_args.args[0][-1])
        self.assertNotIn('generateContent', run.call_args.args[0][-1])

    def test_probe_failure_does_not_expose_subprocess_diagnostics(self):
        args = SimpleNamespace(remote_dir='/code', ssh='/ssh', host='dev-1')
        response = SimpleNamespace(returncode=1, stdout='SECRET', stderr='SECRET')
        with patch.object(check_usage, 'ARGS', args, create=True), \
             patch.object(check_usage.subprocess, 'run', return_value=response):
            self.assertEqual(check_usage.check_plan('agy-11'),
                             {'paid_tier_error': 'probe_process_failed'})

    def test_service_outage_is_not_reported_as_a_starter_downgrade(self):
        args = SimpleNamespace(remote_dir='/code', ssh='/ssh', host='dev-1')
        response = SimpleNamespace(returncode=0, stdout=json.dumps(
            {'http_status': 503, 'id': 'free-tier'}))
        with patch.object(check_usage, 'ARGS', args, create=True), \
             patch.object(check_usage.subprocess, 'run', return_value=response):
            result = check_usage.check_plan('agy-11')
        self.assertEqual(result['paid_tier_error'], 'discovery_failed')
        self.assertEqual(result['paid_tier_http_status'], 503)
        self.assertNotIn('paid_tier_id', result)


if __name__ == '__main__':
    unittest.main()
