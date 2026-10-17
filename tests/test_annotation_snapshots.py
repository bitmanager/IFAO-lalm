"""Inspection exports must preserve typed source labels and publication order."""
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from lalm.annotation.inspection_aggregate import derive_tables


def pass_rows():
    labels = ['true', '{"text":"a"}', True, 1, {'word': 'да'}, ['да'], None]
    provenance = [{'original_label': label} for label in labels]
    provenance += [{'source_label': 'запасная'}, {}, None]
    rows = []
    for index, source in enumerate(provenance):
        for variant in ('verbatim', 'coverage', 'source_correction'):
            pending = variant == 'source_correction'
            rows.append(dict(batch_id='batch', sample_id=str(index), corpus='corpus',
                audio_bucket='audio', audio_key=str(index), audio_sha256=str(index),
                original_annotation='flat representation loses JSON type',
                original_annotation_status='attached', prompt_variant=variant,
                gemini_text=None if pending else 'да',
                processing_state='PENDING' if pending else 'NEEDS_REVIEW',
                error_code=None if pending else 'pilot_review_only',
                review_required=True, human_verified=True,
                updated_at=datetime(2026, 10, 15, tzinfo=timezone.utc), language='ru',
                audio_seconds=None if pending else 2.0,
                inference_seconds=None if pending else 0.5,
                source_provenance=json.dumps(source, ensure_ascii=False) if source is not None else None))
    return rows, labels + ['запасная', None, None]


class SnapshotTests(unittest.TestCase):
    def test_typed_labels_missing_vs_json_null_and_review_flags(self):
        rows, labels = pass_rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pq.write_table(pa.Table.from_pylist(rows), root/'passes.parquet')
            outputs = derive_tables(root/'passes.parquet', root)
            examples = pq.read_table(outputs['examples']).to_pylist()
            self.assertEqual(len(examples), len(labels))
            for row in examples:
                index = int(row['sample_id'])
                if index < 8:
                    self.assertEqual(json.dumps(json.loads(row['original_annotation_json']), sort_keys=True),
                                     json.dumps(labels[index], sort_keys=True))
                else:
                    self.assertIsNone(row['original_annotation_json'])
                self.assertEqual(row['passes_requested'], 3)
                self.assertEqual(row['passes_with_text'], 2)
                self.assertEqual(row['unfinished_passes'], 1)
                self.assertEqual(row['flagged_passes'], 0)
                self.assertTrue(row['review_required'])
                self.assertFalse(row['human_verified'])
            progress = pq.read_table(outputs['progress']).to_pylist()
            self.assertEqual(sum(r['requests'] for r in progress), len(rows))
            self.assertEqual(sum(r['results_with_text'] for r in progress), 20)

    def test_empty_snapshot_keeps_schemas(self):
        rows, _ = pass_rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pq.write_table(pa.Table.from_pylist(rows).slice(0, 0), root/'passes.parquet')
            outputs = derive_tables(root/'passes.parquet', root)
            for path in outputs.values():
                self.assertEqual(pq.read_metadata(path).num_rows, 0)
            self.assertIn('original_annotation_json', pq.read_schema(outputs['examples']).names)

    def test_publication_reads_passes_once_and_switches_pointer_last(self):
        blobs, statements = self.publish()
        manifest = json.loads(blobs['analytics/latest.json'])
        self.assertEqual(manifest['version'], 2)
        self.assertEqual(manifest['purpose'], 'inspection_only_not_training_release')
        self.assertEqual(statements, ['SELECT * FROM review_sources',
                                     'SELECT * FROM review_passes',
                                     'SELECT * FROM review_publications'])
        self.assertEqual(list(blobs)[-1], 'analytics/latest.json')
        for meta in manifest['tables'].values():
            content = blobs[meta['key']]
            self.assertEqual(hashlib.sha256(content).hexdigest(), meta['sha256'])
            self.assertEqual(pq.read_metadata(io.BytesIO(content)).num_rows, meta['rows'])

    def test_bad_provenance_never_replaces_discovery_pointer(self):
        blobs = {}
        with self.assertRaisesRegex(ValueError, 'provenance_must_be_object'):
            self.publish(blobs=blobs, invalid=True)
        self.assertNotIn('analytics/latest.json', blobs)

    def publish(self, blobs=None, invalid=False):
        from lalm.annotation import snapshots
        rows, _ = pass_rows()
        schema = pa.Table.from_pylist(rows).schema
        if invalid:
            rows[0]['source_provenance'] = '[]'
        # Emulate JSONB decoding by psycopg; scalar strings stay scalar strings.
        for row in rows:
            if row['source_provenance'] is not None:
                row['source_provenance'] = json.loads(row['source_provenance'])
        statements = []
        oid = lambda t: 16 if pa.types.is_boolean(t) else 1184 if pa.types.is_timestamp(t) else 701 if pa.types.is_floating(t) else 25

        class Cursor:
            def __init__(self, name):
                self.rows = copy.deepcopy(rows) if name == 'snapshot_passes' else []
                fields = schema if name == 'snapshot_passes' else pa.schema([('id', pa.string())])
                self.description = [SimpleNamespace(name=f.name, type_code=oid(f.type)) for f in fields]

            def execute(self, sql):
                statements.append(sql)

            def fetchmany(self, count):
                result, self.rows = self.rows[:count], self.rows[count:]
                return result

        conn = Mock()
        conn.execute.return_value.fetchone.return_value = {'stamp': datetime(2026, 10, 15, tzinfo=timezone.utc)}
        conn.cursor.side_effect = lambda name: contextlib.nullcontext(Cursor(name))
        client = Mock()
        blobs = {} if blobs is None else blobs

        def put(**kwargs):
            body = kwargs['Body']
            blobs[kwargs['Key']] = body.read() if hasattr(body, 'read') else body

        client.put_object.side_effect = put
        with patch.object(snapshots, 'connect', return_value=contextlib.nullcontext(conn)), \
             patch.object(snapshots, 's3', return_value=client), \
             patch.object(snapshots, 'dataset_bucket', return_value='datasets'):
            snapshots.publish_snapshots()
        return blobs, statements


if __name__ == '__main__':
    unittest.main()
