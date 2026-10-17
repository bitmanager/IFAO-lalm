import copy
import unittest

from infra.annotation.route_account import routed_compose


class RouteAccountTests(unittest.TestCase):
    def setUp(self):
        self.compose = {
            'networks': {'old': {}, 'new': {}},
            'services': {
                'postgres': {'networks': ['old', 'new']},
                's3-relay': {'networks': {'old': {}, 'new': {}}},
                'account': {'environment': {'ACCOUNT_ID': 'account',
                            'HTTPS_PROXY': 'http://old-proxy:1080',
                            'HTTP_PROXY': 'http://old-proxy:1080'},
                            'networks': ['old'],
                            'depends_on': {'postgres': {'condition': 'service_healthy'},
                                           'old-proxy': {'condition': 'service_started'}}},
                'old-proxy': {'networks': ['old']},
                'new-proxy': {'networks': ['new'], 'profiles': ['vpn-tests']},
            },
        }

    def test_moves_only_selected_account_and_preserves_dependencies(self):
        result = routed_compose(copy.deepcopy(self.compose), 'account', 'new-proxy', 'new')
        account = result['services']['account']
        self.assertEqual(account['networks'], ['new'])
        self.assertEqual(account['environment']['HTTPS_PROXY'], 'http://new-proxy:1080')
        self.assertEqual(set(account['depends_on']), {'postgres', 'new-proxy'})
        self.assertIn('account', result['services']['new-proxy']['profiles'])
        self.assertEqual(result['services']['old-proxy'], self.compose['services']['old-proxy'])

    def test_rejects_proxy_outside_account_network(self):
        self.compose['services']['new-proxy']['networks'] = ['old']
        with self.assertRaisesRegex(ValueError, 'proxy_cannot_reach'):
            routed_compose(self.compose, 'account', 'new-proxy', 'new')


if __name__ == '__main__':
    unittest.main()
