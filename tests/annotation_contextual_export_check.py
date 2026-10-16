"""Check real PostgreSQL cohort selection with temporary rows and no S3 writes."""
import contextlib
import json
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError
from lalm.annotation.contextual_calls import export_completed
from lalm.annotation.store import connect


with connect() as conn:
    conn.execute('CREATE TEMP TABLE jobs (LIKE public.jobs INCLUDING ALL) ON COMMIT DROP')
    for letter, organization, state, normalized, verified in [
        ('a', 'meteor', 'NEEDS_REVIEW', True, True),
        ('b', 'meteor', 'RUNNING', True, True),
        ('c', 'hashtesh', 'NEEDS_REVIEW', True, True),
        ('d', 'meteor', 'NEEDS_REVIEW', True, False),
        ('e', 'meteor', 'NEEDS_REVIEW', False, True),
    ]:
        sample = {
            'pilot_id': ('precision-ru-triad-v1-' if normalized else '')
                        + 'organization-native-pilot-v1-' + letter,
            'source_manifest_key': 'incoming/' + letter * 64 + '.jsonl',
            'source_provenance': {
                'key': f'staging/organization-calls-{organization}/pilot-v1/index.json',
                'group_identity_source': 'native_current_relation_join' if verified else 'unknown'}}
        conn.execute('INSERT INTO jobs(id,sample,state) VALUES (%s,%s::jsonb,%s)',
                     (letter, json.dumps(sample), state))
    client = Mock()
    client.head_object.side_effect = ClientError(
        {'ResponseMetadata': {'HTTPStatusCode': 404}, 'Error': {'Code': 'NoSuchKey'}}, 'HeadObject')

    def export(manifest, organization):
        return {'manifest_key': manifest, 'dataset_key': 'mock', 'rows': 1,
                'conversations': 1, 'evidence': 'mock'}

    with patch('lalm.annotation.store.connect', return_value=contextlib.nullcontext(conn)), \
            patch('lalm.annotation.store.s3', return_value=client), \
            patch('lalm.annotation.store.dataset_bucket', return_value='mock'), \
            patch('lalm.annotation.contextual_calls.export_pilot', side_effect=export) as publish:
        result = export_completed('Meteor')
        assert {r['manifest_key'] for r in result['published']} == {
            'incoming/' + letter * 64 + '.jsonl' for letter in ('a', 'e')}
        assert publish.call_count == 2
        publish.reset_mock()
        result = export_completed('HashTesh')
        assert {r['manifest_key'] for r in result['published']} == {
            'incoming/' + 'c' * 64 + '.jsonl'}
        publish.assert_called_once_with('incoming/' + 'c' * 64 + '.jsonl', 'HashTesh')
        client.put_object.assert_not_called()
    conn.rollback()
print('Context exporter selection verified on PostgreSQL; normalized IDs included, fixtures rolled back')
