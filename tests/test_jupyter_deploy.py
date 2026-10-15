"""UI deployment must preserve live worker limits and pinned source versions."""
import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'infra/jupyter/deploy.py'


class JupyterDeployTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        for directory in ('infra/jupyter', 'infra/annotation', 'lalm/annotation', 'notebooks'):
            (self.repo / directory).mkdir(parents=True, exist_ok=True)
        self.script = self.repo / 'infra/jupyter/deploy.py'
        self.script.write_bytes(SOURCE.read_bytes())
        (self.repo / 'lalm/benchmark_langfuse_asr.py').write_text('')
        services = {name: {'volumes': []} for name in
                    ('airflow-api', 'airflow-scheduler', 'airflow-dag-processor',
                     'source-preparer', 'control')}
        services['airflow-api']['ports'] = ['127.0.0.1:18080:8080']
        services['source-preparer']['volumes'] = [
            '/pinned/preparer:/opt/speech/lalm/annotation:ro']
        services['control']['volumes'] = [{'type': 'bind', 'source': '/pinned/control',
            'target': '/opt/speech/lalm/annotation', 'read_only': True}]
        for n in [1, 2, 3, 4, 5, 6, 7, 25]:
            aid = f'agy-{n:02d}'
            services[aid] = {'environment': {'ACCOUNT_ID': aid,
                'WORKER_CONCURRENCY': '3' if n == 7 else '5',
                'NATIVE_API_PERCENT': '70'},
                'volumes': ['/pinned/worker:/opt/speech/lalm/annotation:ro'],
                'stop_grace_period': '360s'}
        self.before = {'services': services, 'volumes': {'existing-data': {}}}
        self.compose_path = self.repo / 'infra/annotation/compose.generated.json'
        self.compose_path.write_text(json.dumps(self.before))
        # A bootstrap example must never override this live capacity.
        (self.repo / 'infra/annotation/accounts.example.json').write_text(json.dumps(
            {'accounts': [{'id': 'agy-07', 'concurrency': 20}]}))

    def run_deploy(self):
        with patch('socket.gethostname', return_value='exp-srv-rtx6000x4-02'):
            runpy.run_path(str(self.script))
        return json.loads(self.compose_path.read_text())

    def test_worker_definitions_and_pinned_mounts_survive_ui_deployment(self):
        expected = copy.deepcopy(self.before)
        result = self.run_deploy()
        for name, service in expected['services'].items():
            if name.startswith('agy-'):
                self.assertEqual(result['services'][name], service)
        for name in ('source-preparer', 'control'):
            self.assertEqual(result['services'][name]['volumes'],
                             expected['services'][name]['volumes'])
        self.assertEqual(result['services']['airflow-scheduler']['volumes'], [
            str(self.repo / 'lalm/annotation') + ':/opt/speech/lalm/annotation:ro'])
        self.assertIn('jupyterhub', result['services'])
        self.assertIn('existing-data', result['volumes'])

    def test_repeat_deployment_preserves_password_and_configuration(self):
        first = self.run_deploy()
        path = self.repo / 'infra/annotation/runtime/jupyter/password'
        before = path.read_bytes()
        stamp = path.stat().st_mtime_ns
        self.assertEqual(self.run_deploy(), first)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
