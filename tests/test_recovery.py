"""Offline checks for reorganized packages and the shared entry point."""
from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from scripts import run_experiments as runner
from demand.synthetic_demand import load_series
from demand.m5_demand import get_representative
from simulation.config import make_m5_config, make_synthetic_config
from project_paths import PROJECT_ROOT


from scripts import run_m5_smooth_batch as batch

class ExtensionTests(unittest.TestCase):
    def test_all_llm_methods_extend_and_resume_without_prefix_calls(self):
        for dataset in ('m5', 'synthetic'):
            record = runner.select_records(dataset)[0]
            for method in ('no_communication', 'no_forecast', 'adjacent', 'controlled_adjacent'):
                with self.subTest(dataset=dataset, method=method), tempfile.TemporaryDirectory() as directory, \
                     patch.dict('os.environ', {'GEMINI_API_KEY': 'offline'}), \
                     patch('agents.gemini_provider.load_dotenv'), \
                     patch('agents.gemini_provider.time.sleep'), \
                     patch('agents.gemini_provider.genai.Client') as client:
                    api = client.return_value.models.generate_content
                    fail = [False]
                    prompts = []

                    def generate(*, model, contents, config):
                        if fail[0]:
                            raise ConnectionError('offline')
                        prompts.append(contents)
                        if 'P6 Controlled adjacent coordination: communication selection' in contents:
                            text = '{"upstream": null, "downstream": null}'
                        else:
                            text = '[[[1]]]'
                            if method == 'adjacent':
                                text += '\n' + ('null' if 'You are the Manufacturer in' in contents else
                                    '{"current_order_to_upstream":1,"planning_insight":"Maintain replenishment."}')
                        return SimpleNamespace(text=text, usage_metadata=None)

                    api.side_effect = generate
                    self.assertEqual(runner.run_record(record, dataset=dataset, method=method,
                                                      periods=2, output_root=directory), 0)
                    folder = Path(directory) / runner.METHODS[method]
                    source = folder / f"{record['id']}_run_001.json"
                    checkpoint = source.with_suffix('.checkpoint.json')
                    original = source.read_bytes(), checkpoint.read_bytes()
                    first = json.loads(original[0])
                    api.reset_mock()
                    self.assertEqual(runner.main(['--extend', str(source), '--periods', '4', '--preflight']), 0)
                    api.assert_not_called()
                    self.assertFalse((folder / f"{record['id']}_run_002.json").exists())
                    # A failed extension saves a new identity; its tail can be resumed.
                    fail[0] = True
                    self.assertEqual(runner.main(['--extend', str(source), '--periods', '4']), 1)
                    tail = folder / f"{record['id']}_run_002.checkpoint.json"
                    self.assertEqual(json.loads(tail.read_text())['result']['completed_periods'], 2)
                    fail[0] = False
                    api.reset_mock()
                    prompts.clear()
                    self.assertEqual(runner.main(['--resume', str(tail)]), 0)
                    calls_per_week = 8 if method == 'controlled_adjacent' else 4
                    self.assertEqual(api.call_count, 2 * calls_per_week)
                    final = json.loads(tail.with_name(tail.name.replace('.checkpoint', '')).read_text())
                    self.assertEqual(final['action_trace'][:8], first['action_trace'])
                    self.assertEqual(final['completed_periods'], 4)
                    self.assertEqual(final['coordination_metrics']['llm_call_count'], 4 * calls_per_week + 3)
                    if method == 'adjacent':
                        self.assertEqual(final['coordination_metrics']['coordination_message_count'], 36)
                    self.assertEqual((source.read_bytes(), checkpoint.read_bytes()), original)
                    self.assertEqual(final['run_metadata']['extension']['continuation_first_period'], 3)
                    rows = json.loads((folder / 'summary.json').read_text())
                    self.assertEqual(len(rows), 2)
                    self.assertEqual(rows[1]['extension']['previous_num_periods'], 2)
                    # A second extension can branch from the successful extended run.
                    self.assertEqual(runner.main(['--extend', str(tail), '--periods', '5']), 0)
                    self.assertEqual(api.call_count, 3 * calls_per_week)

    def test_130_week_extension_matches_uninterrupted_baseline(self):
        for dataset in ('m5', 'synthetic'):
            record = runner.select_records(dataset)[0]
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                self.assertEqual(runner.run_record(record, dataset=dataset, method='ss', periods=130,
                                                  output_root=directory), 0)
                folder = Path(directory) / runner.METHODS['ss']
                source = folder / f"{record['id']}_run_001.json"
                original = source.read_bytes()
                self.assertEqual(runner.main(['--extend', str(source), '--periods', '277']), 0)
                extended = json.loads((folder / f"{record['id']}_run_002.json").read_text())
                from policies.ss import run_sS_baseline
                prefix, config = runner.prepare_config(record, dataset, 277)
                reference = run_sS_baseline(prefix, config=config)
                self.assertEqual(extended['action_trace'], reference['action_trace'])
                extended['evaluation_summary'].pop('time')
                reference['evaluation_summary'].pop('time')
                self.assertEqual(extended['evaluation_summary'], reference['evaluation_summary'])
                self.assertEqual(source.read_bytes(), original)
                for periods in ('130', '129', '278'):
                    with self.assertRaises(ValueError):
                        runner.main(['--extend', str(source), '--periods', periods])

    def test_extension_rejects_changed_input_environment_or_provider(self):
        record = runner.select_records('synthetic')[0]
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('os.environ', {'GEMINI_API_KEY': 'offline'}), \
             patch('agents.gemini_provider.load_dotenv'), \
             patch('agents.gemini_provider.genai.Client') as client:
            api = client.return_value.models.generate_content
            api.return_value = SimpleNamespace(text='[[[1]]]', usage_metadata=None)
            runner.run_record(record, dataset='synthetic', method='no_communication', periods=2,
                              output_root=directory)
            source = Path(directory) / runner.METHODS['no_communication'] / f"{record['id']}_run_001.checkpoint.json"
            original = source.read_bytes()
            for field in ('capacity', 'prompt', 'demand', 'status'):
                data = json.loads(original)
                if field == 'capacity':
                    data['result']['run_metadata']['capacities'][0] += 1
                elif field == 'prompt':
                    data['result']['run_metadata']['provider']['prompt_version'] = 'changed'
                elif field == 'demand':
                    data['demand_sha256'] = 'changed'
                else:
                    data['result']['run_status'] = 'failed'
                source.write_text(json.dumps(data))
                api.reset_mock()
                with self.subTest(field=field), self.assertRaises(ValueError):
                    runner.main(['--extend', str(source), '--periods', '4'])
                api.assert_not_called()
            source.write_bytes(original)

    def test_m5_130_to_131_only_requests_four_new_decisions(self):
        record = runner.select_records('m5')[0]
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('os.environ', {'GEMINI_API_KEY': 'offline'}), \
             patch('agents.gemini_provider.load_dotenv'), \
             patch('agents.gemini_provider.genai.Client') as client:
            api = client.return_value.models.generate_content
            api.return_value = SimpleNamespace(text='[[[1]]]', usage_metadata=None)
            self.assertEqual(runner.run_record(record, dataset='m5', method='no_communication',
                                              periods=130, output_root=directory), 0)
            folder = Path(directory) / runner.METHODS['no_communication']
            source = folder / f"{record['id']}_run_001.json"
            old = json.loads(source.read_text())
            api.reset_mock()
            self.assertEqual(runner.main(['--extend', str(source), '--periods', '131']), 0)
            self.assertEqual(api.call_count, 4)
            final = json.loads((folder / f"{record['id']}_run_002.json").read_text())
            self.assertEqual(final['completed_periods'], 131)
            self.assertEqual(final['action_trace'][:520], old['action_trace'])
            self.assertEqual(final['run_metadata']['extension']['continuation_first_period'], 131)
            self.assertEqual(final['run_metadata']['capacities'], [11] * 4)


class RecoveryTests(unittest.TestCase):
    def test_transient_only(self):
        def result(kind, message=''):
            return {'provider_invocation_trace': [{'failure_category': 'api_transport_failure',
                    'failure_error_type': kind, 'failure_error_message': message}]}
        self.assertTrue(batch.retryable(result('ConnectError')))
        self.assertTrue(batch.retryable(result('ClientError', '429 RESOURCE_EXHAUSTED')))
        self.assertTrue(batch.retryable(result('ServerError', '503 UNAVAILABLE')))
        self.assertFalse(batch.retryable(result('ClientError', '403 permission denied')))
        self.assertFalse(batch.retryable(result('ClientError', '404 model missing')))
        self.assertFalse(batch.retryable({'provider_invocation_trace': []}))
        self.assertTrue(batch.retryable({'error_type': 'GeminiOutputError',
            'provider_invocation_trace': [{'failure_category': 'response_format_or_message_failure'}]}))

    def test_resume_same_identity_then_continue_next_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'batch.json'
            jobs = [dict(method=method, result=str(root / f'{method}.json'),
                         checkpoint=str(root / f'{method}.checkpoint.json'), summary=str(root / 'summary.json'),
                         status='queued', recovery_count=0) for method in batch.BATCH_METHODS]
            state = dict(record_id=batch.RECORD_ID, output_root=str(root), jobs=jobs)
            path.write_text(json.dumps(state))
            calls = []
            def execute(record, **kwargs):
                calls.append(kwargs)
                job = next(j for j in jobs if j['method'] == kwargs['method'])
                failed = len(calls) == 1
                result = dict(run_status='failed' if failed else 'success', completed_periods=12 if failed else 130,
                              error_type='GeminiProviderError' if failed else None,
                              provider_invocation_trace=[dict(failure_category='api_transport_failure',
                                                             failure_error_type='ConnectError')])
                Path(job['result']).write_text(json.dumps(result))
                Path(job['checkpoint']).write_text(json.dumps({'result': result}))
                return int(failed)
            with patch.object(batch.runner, 'select_records', return_value=[{'id': batch.RECORD_ID}]), \
                 patch.object(batch.runner, 'choose_destination', side_effect=lambda folder, record, summary:
                     Path(next(j['result'] for j in jobs if not Path(j['result']).exists()))), \
                 patch.object(batch.runner, 'run_record', side_effect=execute), patch.object(batch.time, 'sleep') as sleep:
                self.assertEqual(batch.run(path), 0)
                sleep.assert_called_once_with(10)
            self.assertIsNone(calls[0]['resume_path'])
            self.assertEqual(calls[1]['resume_path'], Path(jobs[0]['checkpoint']))
            final = json.loads(path.read_text())
            self.assertEqual(final['status'], 'success')
            self.assertEqual(final['jobs'][0]['recovery_count'], 1)
            self.assertTrue(all(j['status'] == 'success' for j in final['jobs']))


class CheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_failed_replace_keeps_previous_copy(self):
        from simulation.experiment_interface import write_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.checkpoint.json'
            write_checkpoint(path, {'completed_periods': 1})
            with patch('simulation.experiment_interface.os.replace', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    write_checkpoint(path, {'completed_periods': 2})
            self.assertEqual(json.loads(path.read_text()), {'completed_periods': 1})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_atomic_checkpoint_retries_temporary_windows_file_lock(self):
        from simulation.experiment_interface import write_checkpoint
        import os
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.checkpoint.json'
            write_checkpoint(path, {'completed_periods': 1})
            replace = os.replace
            attempts = []

            def replace_after_unlock(source, destination):
                attempts.append(destination)
                if len(attempts) == 1:
                    raise PermissionError('temporarily open in a reader')
                replace(source, destination)

            with patch('simulation.experiment_interface.os.replace', side_effect=replace_after_unlock), \
                    patch('simulation.experiment_interface.sleep'):
                write_checkpoint(path, {'completed_periods': 2})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(json.loads(path.read_text()), {'completed_periods': 2})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_checkpoint_rejects_changed_environment_and_reserves_run_id(self):
        from simulation.env import env_creator
        from simulation.experiment_interface import run_episode
        record = load_series()[0]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.dict('os.environ', {'GEMINI_API_KEY': 'offline-test'}), \
                patch('agents.gemini_provider.load_dotenv'), \
                patch('agents.gemini_provider.time.sleep'), \
                patch('agents.gemini_provider.genai.Client') as client:
            client.return_value.models.generate_content.side_effect = ConnectionError('offline')
            self.assertEqual(runner.run_record(record, dataset='synthetic', method='no_communication',
                                              periods=2, output_root=directory), 1)
            folder = Path(directory) / runner.METHODS['no_communication']
            checkpoint_path = folder / f"{record['id']}_run_001.checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text())
            self.assertEqual(runner.next_run_path(folder, record['id']).name, f"{record['id']}_run_002.json")
            _, config = runner.prepare_config(record, 'synthetic', 2)
            config['init_inventories'][0] += 1
            provider = runner.make_provider('no_communication', runner.CoordinationMetrics())
            try:
                with self.assertRaises(ValueError):
                    run_episode(env_creator(config), provider, 'synthetic', config,
                                metrics=provider.metrics, provider_metadata=provider.metadata,
                                resume_checkpoint=checkpoint)
            finally:
                provider.close()
            self.assertEqual(json.loads(checkpoint_path.read_text()), checkpoint)
            _, current_config = runner.prepare_config(record, 'synthetic', 2)
            for changed in ('prompt', 'demand'):
                altered = json.loads(json.dumps(checkpoint))
                if changed == 'prompt':
                    altered['result']['run_metadata']['provider']['prompt_version'] = 'incompatible'
                else:
                    altered['demand_sha256'] = '0' * 64
                provider = runner.make_provider('no_communication', runner.CoordinationMetrics())
                try:
                    with self.subTest(changed=changed), self.assertRaises(ValueError):
                        run_episode(env_creator(current_config), provider, 'synthetic', current_config,
                                    metrics=provider.metrics, provider_metadata=provider.metadata,
                                    resume_checkpoint=altered)
                finally:
                    provider.close()

    def test_checkpoint_resume_preserves_partial_period_and_does_not_repeat_calls(self):
        record = load_series()[0]
        for method in ('no_communication', 'no_forecast', 'adjacent'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory, \
                    redirect_stdout(io.StringIO()), \
                    patch.dict('os.environ', {'GEMINI_API_KEY': 'offline-test'}), \
                    patch('agents.gemini_provider.load_dotenv'), \
                    patch('agents.gemini_provider.time.sleep'), \
                    patch('agents.gemini_provider.genai.Client') as client:
                api = client.return_value.models.generate_content
                calls = []
                failing = [True]

                def generate(*, model, contents, config):
                    period = len(calls) // 4 + 1
                    stage = next(i for i, name in enumerate(('Retailer', 'Wholesaler', 'Distributor', 'Manufacturer'))
                                 if f'You are the {name} in' in contents)
                    # Fail on week 2 Distributor, after six valid stage decisions.
                    if failing[0] and len(calls) == 6:
                        raise ConnectionError('offline disconnect')
                    calls.append((period, stage, contents))
                    output = '[[[1]]]'
                    if method == 'adjacent':
                        output += '\n' + json.dumps(None if stage == 3 else dict(
                            current_order_to_upstream=1, planning_insight='Maintain replenishment.'))
                    return SimpleNamespace(text=output, usage_metadata=None)

                api.side_effect = generate
                kwargs = dict(dataset='synthetic', method=method, periods=3, output_root=directory)
                self.assertEqual(runner.run_record(record, **kwargs), 1)
                folder = Path(directory) / runner.METHODS[method]
                checkpoint_path = folder / f"{record['id']}_run_001.checkpoint.json"
                checkpoint = json.loads(checkpoint_path.read_text())
                self.assertEqual(checkpoint['result']['completed_periods'], 1)
                self.assertEqual(len(checkpoint['decisions']), 6)
                self.assertEqual(checkpoint['result']['coordination_metrics']['decision_llm_call_count'], 9)
                before = list(calls)
                failing[0] = False
                self.assertEqual(runner.main(['--resume', str(checkpoint_path)]), 0)
                final = json.loads((folder / f"{record['id']}_run_001.json").read_text())
                self.assertEqual(final['completed_periods'], 3)
                self.assertEqual(final['run_status'], 'success')
                self.assertEqual(calls[:6], before)
                self.assertEqual([(p, s) for p, s, _ in calls],
                                 [(p, s) for p in range(1, 4) for s in range(4)])
                self.assertEqual(final['coordination_metrics']['decision_llm_call_count'], 15)
                self.assertEqual(len(final['provider_invocation_trace']), 15)
                if method == 'adjacent':
                    self.assertEqual(len(final['communication_trace']), 27)
                    self.assertEqual(final['coordination_metrics']['coordination_message_count'], 27)
                from simulation.env import env_creator
                from simulation.experiment_interface import run_episode
                prefix, config = runner.prepare_config(record, 'synthetic', 3)
                def reference_provider(*args):
                    return 1
                reference_provider.uses_forecast_input = method != 'no_forecast'
                reference = run_episode(env_creator(config), reference_provider, 'synthetic', config)
                for actual, expected in zip(final['action_trace'], reference['action_trace']):
                    for key in ('observation', 'operational_cost', 'reward', 'executed_order_quantity'):
                        self.assertEqual(actual[key], expected[key])
                self.assertEqual(final['episode_reward'], reference['episode_reward'])
                summary = json.loads((folder / 'summary.json').read_text())
                self.assertEqual(len(summary), 1)
                self.assertEqual(summary[0]['run_status'], 'success')
                with self.assertRaises(ValueError):
                    runner.main(['--resume', str(checkpoint_path)])

