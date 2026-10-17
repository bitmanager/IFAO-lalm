import copy
import unittest
from infra.annotation.prepare_slots import add_slots


class SlotTests(unittest.TestCase):
    def test_only_egress_shared_not_credentials(self):
        base = {'services': {'agy-01': {
            'volumes': ['agy-01-home:/home/worker', '/code:/opt/code:ro'],
            'environment': {'ACCOUNT_ID': 'agy-01', 'HTTPS_PROXY':'http://agy-01-proxy:1080'},
            'networks':['agy-01-private'], 'depends_on': {'agy-01-proxy':{}}},
            'agy-01-proxy': {'profiles':['agy-01']}}, 'volumes':{'agy-01-home':{}}}
        original = copy.deepcopy(base)
        result = add_slots(base, {'agy-08':'agy-01'})
        self.assertEqual(base, original)
        self.assertEqual(result['services']['agy-01'], original['services']['agy-01'])
        worker = result['services']['agy-08']
        self.assertIn('agy-08-home:/home/worker', worker['volumes'])
        self.assertNotIn('agy-01-home:/home/worker', worker['volumes'])
        self.assertEqual(worker['networks'], ['agy-01-private'])
        self.assertIn('agy-08', result['services']['agy-01-proxy']['profiles'])
        with self.assertRaises(ValueError):
            add_slots(result, {'agy-08':'agy-01'})


if __name__ == '__main__':
    unittest.main()
