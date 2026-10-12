import copy
import importlib.util
from pathlib import Path
import unittest

from lalm.annotation.contract import PREPROCESSING, job_id, validate_sample, normalize_language


def load_script(name):
    path = Path(__file__).resolve().parents[1] / 'infra/annotation' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnnotationContractTests(unittest.TestCase):
    def test_language_aliases_do_not_infer_ambiguous_or_foreign_speech(self):
        for value in ('ru', 'Russian', ' русский ', 'ru_RU'):
            self.assertEqual(normalize_language(value), 'ru')
        self.assertEqual(normalize_language('ar'), 'ar')
        self.assertEqual(normalize_language('Russian/Ukrainian'), 'russian/ukrainian')
        self.assertEqual(normalize_language(''), '')

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

    def test_grpc_reality_preserves_transport_and_rejects_vision(self):
        convert = load_script('configure_vless').convert
        link = ('vless://5783a3e7-e373-51cd-8642-c83782b807c5@example.com:443'
                '?security=reality&type=grpc&sni=example.com&pbk=test&sid=1234'
                '&fp=firefox&alpn=h2&serviceName=audio')
        stream = convert(link)['outbounds'][0]['streamSettings']
        self.assertEqual(stream['grpcSettings']['serviceName'], 'audio')
        self.assertEqual(stream['realitySettings']['fingerprint'], 'firefox')
        self.assertEqual(stream['realitySettings']['shortId'], '1234')
        for invalid in (link + '&flow=xtls-rprx-vision', link.replace('type=grpc', 'type=ws')):
            with self.assertRaises(ValueError):
                convert(invalid)

    def test_server_uses_existing_s3_and_no_minio(self):
        mod = load_script('render')
        accounts = [{'id': 'agy-01', 'concurrency': 5}]
        compose = mod.use_external_storage(mod.build(accounts, '/bin/agy'), accounts)
        self.assertNotIn('minio', compose['services'])
        self.assertNotIn('s3-data', compose['volumes'])
        for name in ('airflow-api', 'airflow-scheduler', 'airflow-dag-processor', 'agy-01'):
            svc = compose['services'][name]
            self.assertEqual(svc['environment']['S3_BUCKET'], 'speech-annotations')
            self.assertEqual(len(svc['volumes']), len(set(svc['volumes'])))
        self.assertTrue(compose['networks']['agy-01-private']['internal'])

    def test_vless_never_adds_direct_fallback(self):
        convert = load_script('configure_vless').convert
        link = 'vless://5783a3e7-e373-51cd-8642-c83782b807c5@example.com:443?security=tls&type=ws&path=%2Fws'
        config = convert(link)
        self.assertEqual([x['protocol'] for x in config['outbounds']], ['vless'])
        self.assertFalse(config['outbounds'][0]['streamSettings']['tlsSettings']['allowInsecure'])
        for suffix in ('&allowInsecure=1', '&type=grpc', '&unknown=x'):
            with self.assertRaises(ValueError):
                convert(link + suffix)


class MultiPassContractTests(unittest.TestCase):
    def test_source_conditioned_identity_tracks_original_label(self):
        from lalm.annotation.contract import prompt, source_text
        sample = dict(AnnotationContractTests().sample(), prompt_variant='source_correction',
                      source_provenance={'original_label': 'нет, пятнадцать'})
        changed = dict(sample, source_provenance={'original_label': 'да, пятьдесят'})
        self.assertNotEqual(job_id(sample), job_id(changed))
        self.assertEqual(job_id(dict(sample, prompt_variant='verbatim')),
                         job_id(dict(changed, prompt_variant='verbatim')))
        self.assertIn('нет, пятнадцать', prompt('/work/audio.wav', 'source_correction', source_text(sample)))
        self.assertNotIn('нет, пятнадцать', prompt('/work/audio.wav', 'verbatim'))
        for label in (None, {}, '', 'x' * 12001):
            with self.assertRaisesRegex(ValueError, 'missing_source_annotation'):
                job_id(dict(sample, source_provenance={'original_label': label}))

    def test_prompt_variants_have_distinct_job_identity(self):
        from lalm.annotation.contract import contract_sha, prompt
        sample = AnnotationContractTests().sample()
        ids = {job_id(dict(sample, prompt_variant=v)) for v in ('verbatim', 'coverage', 'literal_audit')}
        self.assertEqual(len(ids), 3)
        self.assertNotEqual(contract_sha('coverage'), contract_sha('literal_audit'))
        self.assertIn('независимую', prompt('/work/audio.wav', 'literal_audit'))
        with self.assertRaises(ValueError):
            job_id(dict(sample, prompt_variant='typo'))


if __name__ == '__main__':
    unittest.main()
