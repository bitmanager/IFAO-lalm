import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from infra.annotation import check_usage


class UsageTests(unittest.TestCase):
    def failed_tui_probe(self, startup, alive, cleanup_timeout=False):
        process = Mock()
        process.expect.side_effect = [startup]
        if startup == 3:
            process.expect.side_effect = [3, check_usage.pexpect.TIMEOUT('PRIVATE'),
                                         check_usage.pexpect.TIMEOUT('PRIVATE') if cleanup_timeout else 0]
        else:
            process.expect.side_effect = [startup, 0]
        process.isalive.side_effect = alive
        args = SimpleNamespace(remote_dir='/code', ssh='/ssh', host='dev-1')
        with patch.object(check_usage, 'ARGS', args, create=True), \
             patch.object(check_usage.pexpect, 'spawn', return_value=process) as spawn, \
             patch.object(check_usage.time, 'sleep'), \
             patch.object(check_usage, 'check_plan') as plan:
            result = check_usage.check(10)
        process.close.assert_called_once_with(force=True)
        plan.assert_not_called()
        self.assertIn('timeout --foreground --kill-after=5s 180s agy', spawn.call_args.args[1][-1])
        self.assertNotIn('PRIVATE', json.dumps(result))
        return result, process

    def test_usage_timeout_closes_its_tui_and_retains_error(self):
        result, process = self.failed_tui_probe(3, [True, True, False])
        self.assertEqual(result['error'], 'TIMEOUT')
        self.assertEqual(result['stage'], 'usage')
        self.assertEqual(result['cleanup_method'], 'error_double_interrupt')
        self.assertTrue(result['cleanup_confirmed'])
        process.send.assert_called_once_with('/usage\r')
        self.assertEqual(process.sendcontrol.call_args_list, [call('c'), call('c')])

    def test_setup_failure_interrupts_without_accepting_or_sending_prompt(self):
        result, process = self.failed_tui_probe(1, [True, True, False])
        self.assertEqual(result['stage'], 'interactive_setup')
        self.assertTrue(result['cleanup_confirmed'])
        process.send.assert_not_called()

    def test_failed_interrupt_does_not_claim_cleanup(self):
        result, process = self.failed_tui_probe(3, [True, True, True], cleanup_timeout=True)
        self.assertFalse(result['cleanup_confirmed'])
        self.assertEqual(result['error'], 'TIMEOUT')

    def test_merge_replaces_raw_capture_and_clears_unobserved_fields(self):
        existing = [
            {'id': 'agy-01', 'at': '2026-10-16T06:00:00+00:00',
             'usage': 'older screen', 'weekly_pct': 90, 'five_hour_pct': 70,
             'paid_tier_id': 'g1-ultra-tier', 'cleanup_method': 'old'},
            {'id': 'agy-02', 'at': '2026-10-16T05:00:00+00:00', 'usage': 'unchanged'},
        ]
        before = json.loads(json.dumps(existing))
        fresh = {'account': 'agy-01', 'checked_at': '2026-10-16T07:00:00+00:00',
                 'usage': 'fresh screen', 'weekly_pct': 80, 'cleanup_confirmed': True}
        rows = check_usage.merge_results(existing, [fresh])
        self.assertEqual(rows[0]['usage'], fresh['usage'])
        self.assertEqual(rows[0]['at'], fresh['checked_at'])
        self.assertEqual(rows[0]['weekly_pct'], 80)
        for field in ('five_hour_pct', 'paid_tier_id', 'cleanup_method'):
            self.assertNotIn(field, rows[0])
        self.assertEqual(rows[1], before[1])
        self.assertEqual(existing, before)

    def test_failed_usage_read_cannot_freshen_old_quota_or_screen(self):
        old = [{'id': 'agy-01', 'at': '2026-10-16T06:00:00+00:00',
                'usage': 'old screen', 'weekly_pct': 90, 'paid_tier_id': 'g1-ultra-tier'}]
        fresh = {'account': 'agy-01', 'checked_at': '2026-10-16T07:00:00+00:00',
                 'error': 'TIMEOUT', 'stage': 'usage'}
        row = check_usage.merge_results(old, [fresh])[0]
        self.assertEqual(row['error'], 'TIMEOUT')
        for field in ('usage', 'weekly_pct', 'paid_tier_id'):
            self.assertNotIn(field, row)

    def test_failed_tier_probe_does_not_keep_old_plan_with_new_timestamp(self):
        old = [{'id': 'agy-01', 'at': '2026-10-16T06:00:00+00:00',
                'paid_tier_id': 'g1-ultra-tier', 'paid_tier_http_status': 200}]
        fresh = {'account': 'agy-01', 'checked_at': '2026-10-16T07:00:00+00:00',
                 'usage': 'new screen', 'weekly_pct': 20, 'paid_tier_error': 'probe_unavailable'}
        row = check_usage.merge_results(old, [fresh])[0]
        self.assertEqual(row['paid_tier_error'], 'probe_unavailable')
        self.assertNotIn('paid_tier_id', row)
        self.assertNotIn('paid_tier_http_status', row)

    def test_older_observation_cannot_replace_a_newer_snapshot(self):
        old = [{'id': 'agy-01', 'at': '2026-10-16T07:01:00+00:00', 'usage': 'newer'}]
        fresh = {'account': 'agy-01', 'checked_at': '2026-10-16T07:00:00Z', 'usage': 'older'}
        self.assertEqual(check_usage.merge_results(old, [fresh]), old)

    def test_merge_new_account_excludes_unexpected_sensitive_fields(self):
        fresh = {'account': 'agy-01', 'checked_at': '2026-10-16T07:00:00+00:00',
                 'weekly_pct': 20, 'access_token': 'SECRET'}
        rows = check_usage.merge_results([], [fresh])
        self.assertEqual(rows[0]['id'], 'agy-01')
        self.assertNotIn('SECRET', json.dumps(rows))

    def test_failed_publication_does_not_expose_remote_diagnostics(self):
        args = SimpleNamespace(remote_dir='/code', ssh='/ssh', host='dev-1')
        response = SimpleNamespace(returncode=1, stdout='SECRET', stderr='SECRET')
        with patch.object(check_usage, 'ARGS', args, create=True), \
             patch.object(check_usage.subprocess, 'run', return_value=response):
            with self.assertRaisesRegex(RuntimeError, '^usage_publish_failed$'):
                check_usage.publish_results([])

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
