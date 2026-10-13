"""Check text IDs/embeddings parity and real-audio sensitivity before a new run."""
import json
import os
from pathlib import Path

import soundfile as sf
import torch

import kl_runtime as runtime
from evaluate_kl import tokens
from benchmark_sglang_client import SGLangClient
from lalm_core.model.gigaam_adapter import gigaam_output_length

root = Path('/mnt/local/drive2/checkpoint-response-bench-kliter2-epoch8-v1')
spec = json.loads((root / 'answers204.gpu1.json').read_text())
assert os.environ['CUDA_VISIBLE_DEVICES'] == spec['gpu_uuid']
assert runtime.digest(spec['checkpoint']) == spec['checkpoint_sha256']
client = SGLangClient(spec['checkpoint'], 'http://127.0.0.1:18139')
baseline = {r['id']: r for r in json.loads((root / 'answers204.results.json').read_text())['cases']}
rows = []
for index in (1, 12, 35):
    case = spec['cases'][index]
    assert runtime.digest(case['audio_path']) == case['audio_sha256']
    wave, rate = sf.read(case['audio_path'], dtype='float32')
    assert rate == 16000 and wave.ndim == 1
    wave = torch.from_numpy(wave.copy())
    count = int(gigaam_output_length(torch.tensor(wave.numel()))) // 2
    def prefix(text):
        return tokens(client.tokenizer, [{'role': 'system', 'content': spec['system_prompt']},
                                        *case['history'], {'role': 'user', 'content': text}])
    ids = prefix(case['reference'])
    text_ids = client.generate(ids)
    text_embeds = client.generate(ids, text_embeds=True)
    audio_prefix = prefix('<|audio|>' * count)
    audio = client.generate(audio_prefix, wave)
    silence = client.generate(audio_prefix, torch.zeros_like(wave))
    rows.append(dict(id=case['id'], reference=case['reference'], text_ids=text_ids,
                     text_embeds=text_embeds, audio=audio, silence=silence,
                     hf=baseline[case['id']]['outputs'],
                     text_parity=text_ids['text'] == text_embeds['text'],
                     audio_sensitive=audio['text'] != silence['text']))
    runtime.atomic(root / 'sglang/canary.json', rows)
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
assert all(r['text_parity'] for r in rows), 'Text embeddings do not reproduce text IDs'
assert any(r['audio_sensitive'] for r in rows), 'No audio sensitivity observed'
assert all(r[k]['complete'] for r in rows for k in ('text_ids', 'text_embeds', 'audio', 'silence'))
print('CANARY_PASS', flush=True)
