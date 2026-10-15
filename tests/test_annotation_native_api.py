import copy
import json
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from lalm.annotation.contract import prompt
from lalm.annotation.native_api import generate, validate_response, native_prompt


class NativeAPITests(unittest.TestCase):
    def response(self):
        return {'response': {'modelVersion': 'gemini-3.8-flash', 'candidates': [{
            'finishReason': 'STOP', 'content': {'parts': [{'text': '```json\n'+json.dumps({
                'audio_heard': True, 'language': 'ru', 'transcript': 'Привет',
                'uncertain_fragments': [], 'context_consistency_notes': []})+'\n```'}]}}]}}

    def test_actual_fenced_format_and_strict_rejection(self):
        out, _ = validate_response(self.response(), 'gemini-3.8-flash-low')
        self.assertEqual(out['transcript'], 'Привет')
        for change in ('model', 'finish', 'tool', 'type', 'audio'):
            r = self.response()
            c = r['response']['candidates'][0]
            if change == 'model': r['response']['modelVersion'] = 'different-model'
            if change == 'finish': c['finishReason'] = 'MAX_TOKENS'
            if change == 'tool': c['content']['parts'].append({'functionCall': {'name': 'finish'}})
            if change in ('type', 'audio'):
                value = copy.deepcopy(out)
                value['audio_heard'] = False if change == 'audio' else 'true'
                c['content']['parts'][0]['text'] = json.dumps(value)
            with self.assertRaises(ValueError): validate_response(r, 'gemini-3.8-flash-low')

    def test_timeout_has_one_dispatch_and_no_tool_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp)/'a.wav'; audio.write_bytes(b'RIFF'+b'0'*100)
            with patch('lalm.annotation.native_api.token', return_value='local-secret'), \
                 patch('lalm.annotation.native_api.post', side_effect=TimeoutError) as send:
                result, _ = generate(audio, 'project')
            self.assertEqual(send.call_count, 1)
            self.assertEqual(result['inference_http_attempts'], 1)
            self.assertEqual(result['auto_retries'], 0)
            payload = send.call_args.args[1]
            self.assertNotIn('tools', payload['request'])
            self.assertIn('inlineData', payload['request']['contents'][0]['parts'][1])
            self.assertEqual(result['status'], 'failed_do_not_auto_retry')

    def test_shared_prompt_keeps_language_policy_and_reference(self):
        for variant in ('verbatim', 'coverage', 'source_correction'):
            reference = 'Черновик' if variant == 'source_correction' else None
            text = native_prompt(variant, reference)
            self.assertNotIn('view_file', text)
            if reference: self.assertIn(reference, text)
        self.assertIn('view_file', prompt('audio.wav'))

    def test_native_prompt_requests_result_object_instead_of_schema_echo(self):
        text = native_prompt('verbatim')
        self.assertNotIn('Return the supplied JSON schema.', text)
        self.assertIn('not the schema definition', text)
        self.assertIn('Put spoken words in transcript, not text.', text)
        for variant in ('verbatim', 'coverage', 'source_correction'):
            instruction = native_prompt(variant, 'Draft' if variant == 'source_correction' else None)
            self.assertIn('audio_heard, language, transcript, uncertain_fragments, context_consistency_notes', instruction)
            self.assertNotIn('responseJsonSchema', instruction)


if __name__ == '__main__':
    unittest.main()
