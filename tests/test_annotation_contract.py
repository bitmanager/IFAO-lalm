import copy
import importlib.util
from pathlib import Path
import unittest

from lalm.annotation.contract import PREPROCESSING, job_id, validate_sample


def load_script(name):
    path = Path(__file__).resolve().parents[1] / 'infra/annotation' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnnotationContractTests(unittest.TestCase):
    def sample(self):
        return {'sample_id': 'one', 'conversation_id': 'call', 'audio_key': 'audio/one.wav',
                'audio_sha256': 'a' * 64, 'split': 'train', 'preprocessing_version': PREPROCESSING,
                'speech_qc': {'audio_sha256': 'a' * 64, 'speech_detected': True,
                              'backend': 'silero', 'version': 'pinned'}}

    def test_rejects_holdout_and_stale_qc(self):
        sample = self.sample()
        for split in ('benchmark', 'test', 'human_gold'):
            with self.assertRaises(ValueError):
                validate_sample(dict(sample, split=split))
        sample['speech_qc']['audio_sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            validate_sample(sample)

    def test_identity_ignores_relocated_audio_but_not_content(self):
        sample = self.sample()
        self.assertEqual(job_id(sample), job_id(dict(sample, audio_key='moved/file.wav')))
        other = copy.deepcopy(sample)
        other['audio_sha256'] = other['speech_qc']['audio_sha256'] = 'b' * 64
        self.assertNotEqual(job_id(sample), job_id(other))

    def test_accounts_have_separate_internal_networks_and_homes(self):
        compose = load_script('render').build([{'id': 'agy-01', 'concurrency': 20},
                                               {'id': 'agy-02', 'concurrency': 2}], '/bin/agy')
        for aid in ('agy-01', 'agy-02'):
            worker = compose['services'][aid]
            self.assertEqual(worker['networks'], [aid + '-private'])
            self.assertTrue(compose['networks'][aid + '-private']['internal'])
            self.assertIn(aid + '-home:/home/worker', worker['volumes'])
            self.assertNotIn('egress', worker['networks'])
            self.assertNotIn('ports', compose['services'][aid + '-proxy'])

    def test_vless_never_adds_direct_fallback(self):
        convert = load_script('configure_vless').convert
        link = 'vless://5783a3e7-e373-51cd-8642-c83782b807c5@example.com:443?security=tls&type=ws&path=%2Fws'
        config = convert(link)
        self.assertEqual([x['protocol'] for x in config['outbounds']], ['vless'])
        self.assertFalse(config['outbounds'][0]['streamSettings']['tlsSettings']['allowInsecure'])
        for suffix in ('&allowInsecure=1', '&type=grpc', '&unknown=x'):
            with self.assertRaises(ValueError):
                convert(link + suffix)


if __name__ == '__main__':
    unittest.main()
