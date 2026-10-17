"""Stage bounded native-call pilots through the existing annotation manifests.

Native CSV rows remain separate from segment labels. Channels retain their
numbers; this adapter never guesses agent/customer roles or word alignment.
"""
import csv
import hashlib
import json
from pathlib import Path


def current_dialogues(directory):
    def read(name):
        with (Path(directory) / name).open(newline='') as stream:
            return list(csv.DictReader(stream))

    relations = read('relations-native.csv')
    by_id = {row['id']: row for row in relations}
    if len(by_id) != len(relations):
        raise ValueError('duplicate_native_relation')
    result = {}
    for dialog in read('dialogs-native.csv'):
        relation = by_id.get(dialog['current_relation_id'])
        if relation is None:
            raise ValueError('missing_current_relation')
        if (relation['organization_id'] != dialog['organization_id']
                or relation['dialog_id'] != dialog['id']):
            raise ValueError('native_dialogue_join_mismatch')
        if dialog['id'] in result:
            raise ValueError('duplicate_native_dialogue')
        result[dialog['id']] = {'dialog': dialog, 'relation': relation}
    return result


def segment_sample(record, index, source, channel, start, end, sha):
    dialog, relation = record['dialog'], record['relation']
    if not dialog['conversation_id']:
        raise ValueError('missing_native_conversation_id')
    return {
        'sample_id': sha, 'conversation_id': dialog['conversation_id'],
        'audio_bucket': 'speech-corpora',
        'audio_key': f'prepared/annotation-v1/{sha}.wav', 'audio_sha256': sha,
        'source_audio_sha256': source['sha256'], 'split': 'train',
        'preprocessing_version': 'mono-pcm16-16000-v1', 'review_only': True,
        'pilot_id': 'organization-native-pilot-v1-' + index['sha256'],
        'prompt_variant': 'verbatim',
        'source_provenance': {
            'bucket': index['bucket'], 'key': index['key'],
            'source_index_sha256': index['sha256'],
            'source_audio_key': relation['audio_s3_key'],
            'dialog_id': dialog['id'], 'relation_id': relation['id'],
            'organization_id': dialog['organization_id'],
            'channel': channel, 'channel_role': 'unverified',
            'start_seconds': start, 'end_seconds': end,
            'max_overlap_seconds': 0.5,
            'group_identity_verified': True,
            'group_identity_source': 'native_current_relation_join',
            'original_label': None,
            'original_label_scope': 'whole_dialogue_in_source_index_not_segment',
            'canonical_pcm_overlap_checked': False},
        'speech_qc': {'audio_sha256': sha, 'speech_detected': True,
                      'backend': 'webrtcvad', 'version': '2.0.14-mode2-20ms'}}


def stage_pilot(metadata, selection, audio_directory):
    from .prepare import segments
    from .queue import protected
    from .store import connect, dataset_bucket, s3

    records = current_dialogues(metadata)
    selected = json.loads(Path(selection).read_text())
    if not isinstance(selected, list) or not 1 <= len(selected) <= 12:
        raise ValueError('pilot_requires_one_to_twelve_calls')
    client = s3()
    body = client.get_object(Bucket=dataset_bucket(), Key='benchmarks/holdouts.json')['Body']
    try:
        registry = json.loads(body.read())
    finally:
        body.close()
    guarded_audio = set(registry['audio_sha256'])
    guarded_conversations = set(registry['conversation_ids'])
    buffered, seen = [], set()
    # Publish the incoming manifest only after every selected call passes checks.
    for source in selected:
        record = records[source['dialog_id']]
        dialog, relation = record['dialog'], record['relation']
        if (source['source_audio_key'] != relation['audio_s3_key']
                or source['conversation_id'] != dialog['conversation_id']
                or source['relation_id'] != relation['id']):
            raise ValueError('selection_native_join_mismatch')
        raw = (Path(audio_directory) / Path(source['local_path']).name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source['sha256']:
            raise ValueError('changed_pilot_audio')
        if source['sha256'] in guarded_audio or dialog['conversation_id'] in guarded_conversations:
            raise ValueError('protected_native_call')
        corpus = 'organization-calls-' + source['organization'].lower()
        original = json.dumps({'native_record': record, 'source': source},
                              ensure_ascii=False, sort_keys=True).encode()
        digest = hashlib.sha256(original).hexdigest()
        index = {'bucket': dataset_bucket(), 'sha256': digest,
                 'key': f'staging/{corpus}/pilot-v1/{digest}.json'}
        client.put_object(Bucket=index['bucket'], Key=index['key'], Body=original)
        for channel, start, end, data in segments(raw):
            sha = hashlib.sha256(data).hexdigest()
            if sha in guarded_audio:
                raise ValueError('protected_native_segment')
            sample = segment_sample(record, index, source, channel, start, end, sha)
            with connect() as conn:
                if protected(conn, sample):
                    raise ValueError('protected_native_segment')
            if sha in seen:
                raise ValueError('duplicate_pilot_segment_requires_alias_review')
            seen.add(sha)
            client.put_object(Bucket=sample['audio_bucket'], Key=sample['audio_key'], Body=data)
            buffered.append(sample)
    if not buffered:
        raise ValueError('pilot_contains_no_speech')
    payload = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                      for row in buffered).encode()
    key = 'incoming/' + hashlib.sha256(payload).hexdigest() + '.jsonl'
    client.put_object(Bucket=dataset_bucket(), Key=key, Body=payload)
    with connect() as conn:
        conn.execute('INSERT INTO prepared_manifests(object_key,row_count) VALUES (%s,%s) '
                     'ON CONFLICT DO NOTHING', (key, len(buffered)))
    return {'manifest_key': key, 'calls': len(selected), 'segments': len(buffered),
            'review_only': True, 'canonical_pcm_overlap_checked': False}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--audio-directory', required=True)
    args = parser.parse_args()
    print(json.dumps(stage_pilot(args.metadata, args.selection, args.audio_directory)))
