"""Native Gemini transport for bounded, review-only experiments.

Each generation is one POST. No automatic retry, endpoint fallback or tool loop.
OAuth credentials are read only from the account's own local native CLI file.
"""
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

from .contract import SCHEMA, digest, normalize_language, prompt
from .native_auth import repair_trailing_auth

ENDPOINT = 'https://daily-cloudcode-pa.googleapis.com'
USER_AGENT = 'antigravity/cli/1.3.4 (aidev_client; os_type=linux; arch=amd64; auth_method=consumer)'
# Thinking consumes the output budget too. Observed 2048-token responses spent
# ~1960 tokens thinking and stopped midway through the transcription JSON.
GENERATION_CONFIG = {'maxOutputTokens': 8192, 'temperature': 0.2,
                     'responseMimeType': 'application/json'}
PROMPT_BLOCK_CODES = frozenset('native_prompt_blocked_' + reason for reason in (
    'OTHER', 'SAFETY', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'IMAGE_SAFETY'))


def native_prompt(variant, reference=None):
    path = '<attached_audio>'
    result = prompt(path, variant, reference)
    result = result.replace(
        f'Единственный вход — аудиофайл {path}. Открой его через view_file как аудио. ',
        'Единственный вход — прикреплённое аудио. ')
    result = result.replace(
        f'The only input is audio file {path}. Open it as audio using view_file. ',
        'The only input is the attached audio. ')
    result = result.replace('Return the supplied JSON schema.',
        'Return one JSON object conforming to the supplied schema. '
        'Return the transcription result, not the schema definition.')
    result += (' Use exactly these field names: audio_heard, language, transcript, '
               'uncertain_fragments, context_consistency_notes. '
               'Put spoken words in transcript, not text.')
    if 'view_file' in result:
        raise ValueError('unexpected_native_tool_instruction')
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def token(min_remaining=0):
    path = Path.home() / '.gemini/antigravity-cli/antigravity-oauth-token'
    repair_trailing_auth(path)
    value = json.loads(path.read_text())['token']
    if datetime.fromisoformat(value['expiry'].replace('Z', '+00:00')) <= datetime.now(timezone.utc) + timedelta(seconds=min_remaining):
        raise ValueError('access_token_expired_before_dispatch')
    return value['access_token']


def post(action, payload, access_token, timeout):
    request = urllib.request.Request(ENDPOINT + '/v1internal:' + action,
        data=json.dumps(payload).encode(), headers={
            'Authorization': 'Bearer ' + access_token, 'Content-Type': 'application/json',
            'User-Agent': USER_AGENT})
    try:
        response = urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        body = response.read(2 * 1024**2 + 1)
    if len(body) > 2 * 1024**2:
        raise ValueError('oversized_native_response')
    return response.status, body


def discover():
    status, raw = post('loadCodeAssist', {'metadata': {'ideType': 'ANTIGRAVITY'}}, token(), 35)
    if status != 200:
        raise ValueError('native_discovery_http_' + str(status))
    project = json.loads(raw)['cloudaicompanionProject']
    return project['id'] if isinstance(project, dict) else project


def validate_response(value, model):
    response = value.get('response', value)
    expected_version = model.rsplit('-', 1)[0]
    if response.get('modelVersion') != expected_version:
        raise ValueError('native_model_mismatch')
    candidates = response.get('candidates', [])
    feedback = response.get('promptFeedback')
    if not candidates and isinstance(feedback, dict):
        code = 'native_prompt_blocked_' + str(feedback.get('blockReason', ''))
        if code in PROMPT_BLOCK_CODES:
            raise ValueError(code)
    if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
        raise ValueError('native_incomplete_or_ambiguous_candidate')
    parts = candidates[0].get('content', {}).get('parts', [])
    if any('functionCall' in part or 'inlineData' in part for part in parts):
        raise ValueError('unexpected_native_output_part')
    text = ''.join(part.get('text', '') for part in parts if not part.get('thought')).strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[8:-4]
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != set(SCHEMA['required']):
        raise ValueError('native_schema_keys')
    if result['audio_heard'] is not True:
        raise ValueError('native_audio_not_heard')
    if not all(isinstance(result[k], str) for k in ('language', 'transcript')):
        raise ValueError('native_schema_string_types')
    for key in ('uncertain_fragments', 'context_consistency_notes'):
        if not isinstance(result[key], list) or any(not isinstance(v, str) for v in result[key]):
            raise ValueError('native_schema_list_types')
    if result['context_consistency_notes'] or not result['language'].strip():
        raise ValueError('native_unexpected_context_or_missing_language')
    return result, response


def generate(audio, project, variant='verbatim', reference=None, model='gemini-3.8-flash-low'):
    audio = Path(audio).read_bytes()
    if not audio.startswith(b'RIFF') or not 44 < len(audio) < 2 * 1024**2:
        raise ValueError('probe_requires_short_wav')
    instruction = native_prompt(variant, reference)
    generation_config = dict(GENERATION_CONFIG)
    contract = digest({'transport': 'native_api_v1', 'prompt': instruction,
                       'schema': SCHEMA, 'model': model,
                       'generation_config': generation_config})
    payload = {'project': project, 'model': model, 'userAgent': 'antigravity',
        'requestId': 'agent/native-canary/' + str(uuid.uuid4()), 'requestType': 'agent',
        'request': {'sessionId': '-1', 'contents': [{'role': 'user', 'parts': [
            {'text': instruction + '\nJSON schema: ' + json.dumps(SCHEMA)},
            {'inlineData': {'mimeType': 'audio/wav', 'data': base64.b64encode(audio).decode()}}]}],
            'generationConfig': generation_config}}
    access_token = token()  # Any auth failure here is explicitly before dispatch.
    started = time.monotonic()
    result = {'transport': 'native_api_v1', 'model': model, 'prompt_variant': variant,
        'audio_sha256': hashlib.sha256(audio).hexdigest(), 'contract_sha256': contract,
        'inference_http_attempts': 1, 'auto_retries': 0, 'human_gold': False,
        'review_only': True, 'prompt_format': 'result_object_v1',
        'generation_config': generation_config}
    raw = b''
    try:
        status, raw = post('generateContent', payload, access_token, 90)
        result['http_status'] = status
        if status != 200:
            raise ValueError('native_generation_http_' + str(status))
        out, envelope = validate_response(json.loads(raw), model)
        result.update(verbatim_text=out['transcript'], language=normalize_language(out['language']),
                      uncertain_spans=out['uncertain_fragments'], structured_output=out,
                      model_version=envelope['modelVersion'], usage=envelope.get('usageMetadata'),
                      status='complete')
    except Exception as error:
        result.update(status='failed_do_not_auto_retry', error_type=type(error).__name__)
        if isinstance(error, ValueError) and str(error).startswith('native_'):
            result['error_code'] = str(error)
    result['inference_seconds'] = round(time.monotonic() - started, 3)
    return result, raw
