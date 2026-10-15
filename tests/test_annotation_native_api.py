import copy
import json
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from lalm.annotation.contract import prompt
from lalm.annotation.native_api import generate, validate_response, native_prompt, GENERATION_CONFIG


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

    def test_prompt_feedback_is_terminal_and_not_a_missing_candidate(self):
        blocked = {'response': {'modelVersion': 'gemini-3.8-flash',
                               'promptFeedback': {'blockReason': 'OTHER'}}}
        with self.assertRaisesRegex(ValueError, '^native_prompt_blocked_OTHER$'):
            validate_response(blocked, 'gemini-3.8-flash-low')
        blocked['response']['modelVersion'] = 'different-model'
        with self.assertRaisesRegex(ValueError, '^native_model_mismatch$'):
            validate_response(blocked, 'gemini-3.8-flash-low')
        blocked['response']['modelVersion'] = 'gemini-3.8-flash'
        for reason in ('BLOCK_REASON_UNSPECIFIED', 'future_unknown_reason'):
            blocked['response']['promptFeedback']['blockReason'] = reason
            with self.assertRaisesRegex(ValueError, '^native_incomplete_or_ambiguous_candidate$'):
                validate_response(blocked, 'gemini-3.8-flash-low')

    def test_prompt_block_keeps_raw_response_and_never_retries(self):
        raw = json.dumps({'response': {'modelVersion': 'gemini-3.8-flash',
                         'promptFeedback': {'blockReason': 'OTHER'}}}).encode()
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp)/'a.wav'; audio.write_bytes(b'RIFF'+b'0'*100)
            with patch('lalm.annotation.native_api.token', return_value='local-secret'), \
                 patch('lalm.annotation.native_api.post', return_value=(200, raw)) as send:
                result, saved_raw = generate(audio, 'project')
        send.assert_called_once()
        self.assertEqual(saved_raw, raw)
        self.assertEqual(result['error_code'], 'native_prompt_blocked_OTHER')
        self.assertEqual(result['status'], 'failed_do_not_auto_retry')
        self.assertEqual(result['auto_retries'], 0)

    def test_shared_prompt_keeps_language_policy_and_reference(self):
        for variant in ('verbatim', 'coverage', 'source_correction'):
            reference = 'Черновик' if variant == 'source_correction' else None
            text = native_prompt(variant, reference)
            self.assertNotIn('view_file', text)
            if reference: self.assertIn(reference, text)
        self.assertIn('view_file', prompt('audio.wav'))

    def test_output_budget_is_recorded_and_changes_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp)/'a.wav'; audio.write_bytes(b'RIFF'+b'0'*100)
            with patch('lalm.annotation.native_api.token', return_value='local-secret'), \
                 patch('lalm.annotation.native_api.post', return_value=(200, json.dumps(self.response()).encode())) as send:
                result, _ = generate(audio, 'project')
                payload = send.call_args.args[1]
                self.assertGreaterEqual(payload['request']['generationConfig']['maxOutputTokens'], 4096)
                self.assertEqual(result['generation_config'], payload['request']['generationConfig'])
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(send.call_count, 1)
                with patch.dict(GENERATION_CONFIG, maxOutputTokens=2048):
                    previous_budget, _ = generate(audio, 'project')
                self.assertNotEqual(result['contract_sha256'], previous_budget['contract_sha256'])
                self.assertEqual(result['generation_config']['maxOutputTokens'], 8192)

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
