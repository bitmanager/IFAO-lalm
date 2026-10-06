"""CPU-only voice selection checks; no service, synthesis or model loading."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('own_tts_adapter', Path(__file__).resolve().parents[1] / 'utils/own_tts_adapter.py')
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class VoiceMapTest(unittest.TestCase):
    def setUp(self):
        self.record = {'request_id': 'dialog1', 'config': {'language': 'ru'}, 'participants': [
            {'name': 'Аня', 'role': 'user', 'gender': 'female'},
            {'name': 'Маша', 'role': 'assistant', 'gender': 'female'}]}
        self.voices = [{'voice_id': v, 'gender': 'female', 'language': 'ru', 'tags': []}
                       for v in ['anastasia-1', 'ekaterina-1']]
        self.mapping = {'dialog1': {'Аня': 'anastasia-1', 'Маша': 'ekaterina-1'}}

    def test_valid_exact_mapping(self):
        before = copy.deepcopy(self.record)
        adapter.validate_voice_map(self.mapping, [self.record], self.voices)
        self.assertEqual(before, self.record)

    def test_invalid_maps(self):
        bad = [None, {}, {'other': self.mapping['dialog1']},
               {**self.mapping, 'extra': {}}, {'dialog1': {'Аня': 'anastasia-1'}},
               {'dialog1': {**self.mapping['dialog1'], 'extra': 'college-01'}},
               {'dialog1': {'Аня': 'unknown', 'Маша': 'ekaterina-1'}},
               {'dialog1': {'Аня': 'anastasia-1', 'Маша': 'anastasia-1'}},
               {'dialog1': {'Аня': [], 'Маша': 'ekaterina-1'}}]
        for mapping in bad:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                adapter.validate_voice_map(mapping, [self.record], self.voices)

    def test_gender_mismatch(self):
        self.voices[0]['gender'] = 'male'
        with self.assertRaises(ValueError):
            adapter.validate_voice_map(self.mapping, [self.record], self.voices)

    def test_duplicate_json_keys(self):
        with self.assertRaises(ValueError):
            json.loads('{"id":{"name":"a","name":"b"}}', object_pairs_hook=adapter.unique_json_object)

    def test_main_rejects_validation_or_unapproved_registry_voice_without_network(self):
        for voice_update in [{'tags': ['validation']}, {'language': 'en'}, {'voice_id': 'private-voice'}]:
            registry = copy.deepcopy(self.voices)
            registry[0].update(voice_update)
            # Keep both gender pools populated, as required by the unchanged wrapper.
            registry.append({'voice_id': 'college-01', 'gender': 'male', 'language': 'ru', 'tags': []})
            with self.subTest(update=voice_update), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)
                (path/'input.jsonl').write_text(json.dumps(self.record)+'\n')
                (path/'map.json').write_text(json.dumps(self.mapping))
                responses = [io.BytesIO(json.dumps({'resolved_revision': '384ffff264f0407498f3ca7138871b9cf03f69f6'}).encode()),
                             io.BytesIO(json.dumps({'voices': registry}).encode())]
                argv = ['adapter', '--input', str(path/'input.jsonl'), '--output', str(path/'out'),
                        '--upstream', str(path), '--voice-map', str(path/'map.json')]
                with patch.object(sys, 'argv', argv), patch.object(adapter, 'urlopen', side_effect=responses), self.assertRaises(ValueError):
                    adapter.main()

    def test_conflicting_map_preserves_existing_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'input.jsonl').write_text(json.dumps(self.record)+'\n')
            (path/'map.json').write_text(json.dumps(self.mapping))
            (path/'tts.py').write_text('# frozen upstream placeholder for hashing')
            out = path/'out'
            out.mkdir()
            previous = {'input_sha256': hashlib.sha256((path/'input.jsonl').read_bytes()).hexdigest(),
                        'voice_map': {'dialog1': {'Аня': 'ekaterina-1', 'Маша': 'anastasia-1'}}}
            old_bytes = json.dumps(previous).encode()
            (out/'provenance.json').write_bytes(old_bytes)
            registry = self.voices + [{'voice_id': 'college-01', 'gender': 'male', 'language': 'ru', 'tags': []}]
            responses = [io.BytesIO(json.dumps({'resolved_revision': '384ffff264f0407498f3ca7138871b9cf03f69f6'}).encode()),
                         io.BytesIO(json.dumps({'voices': registry, 'remote_revision': 'registry-fixed'}).encode())]
            argv = ['adapter', '--input', str(path/'input.jsonl'), '--output', str(out),
                    '--upstream', str(path), '--voice-map', str(path/'map.json')]
            with patch.object(sys, 'argv', argv), patch.object(adapter, 'urlopen', side_effect=responses), patch.object(adapter, 'ProcessPoolExecutor') as pool:
                with self.assertRaisesRegex(ValueError, 'provenance conflicts'):
                    adapter.main()
                pool.assert_not_called()
            self.assertEqual((out/'provenance.json').read_bytes(), old_bytes)

    def test_fixed_map_cache_conflict_and_random_default_restored(self):
        original_assign = lambda *args: 'original-random-assignment'
        native = types.SimpleNamespace(assign_prompts=original_assign, mix_to_multichannel_wav=lambda: None)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'di'
            path.mkdir()
            meta = {'source_script': self.record, 'voice': {'silence': None, **self.mapping['dialog1']}}
            (path/'dialog1.json').write_text(json.dumps(meta))
            (path/'dialog1.complete.json').write_text('{"status":"cached"}')
            base = (self.record, folder, folder, 'unused', {'male': [], 'female': []}, 'train')
            with patch.object(adapter.importlib, 'import_module', return_value=native), patch.dict(sys.modules):
                self.assertEqual(adapter.render_one((*base, self.mapping['dialog1'])), {'status': 'cached'})
                self.assertEqual(native.assign_prompts(None, [], []), meta['voice'])
                self.assertEqual(native.assign_prompts(None, [], []), meta['voice'])
                bad = {'Аня': 'ekaterina-1', 'Маша': 'anastasia-1'}
                with self.assertRaises(ValueError):
                    adapter.render_one((*base, bad))
                self.record['dialogue'] = []
                with self.assertRaises(ValueError):
                    adapter.render_one((*base, self.mapping['dialog1']))
                del self.record['dialogue']
                adapter.render_one((*base, None))
                self.assertIs(native.assign_prompts, original_assign)


if __name__ == '__main__':
    unittest.main()
