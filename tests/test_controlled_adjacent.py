"""Offline selective routing, resource accounting and interrupted-run replay."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.gemini_controlled_adjacent import (
    GeminiControlledAdjacentProvider, build_sharing_prompt, parse_sharing_response,
)
from evaluation.metrics import CoordinationMetrics
from simulation.config import env_configs
from simulation.env import env_creator
from simulation.experiment_interface import run_episode
from scripts import run_experiments
from demand.synthetic_demand import load_series


class ControlledAdjacentTests(unittest.TestCase):
    def setUp(self):
        patches = [patch.dict(os.environ, {'GEMINI_API_KEY': 'offline-test'}),
                   patch('agents.gemini_provider.load_dotenv'),
                   patch('agents.gemini_provider.genai.Client'),
                   patch('agents.gemini_provider.time.sleep')]
        for item in patches:
            self.addCleanup(item.stop)
            value = item.start()
            if item is patches[2]:
                self.client = value.return_value
        self.config = dict(env_configs['m5'], num_periods=3,
                           prod_capacities=[7]*4, demand_fn=lambda t: t+1)

    def provider(self):
        return GeminiControlledAdjacentProvider(CoordinationMetrics())

    def generation(self, *, model, contents, config):
        if 'P6 Controlled adjacent coordination: communication selection' in contents:
            stage = next(i for i, name in enumerate(('Retailer', 'Wholesaler', 'Distributor', 'Manufacturer'))
                         if f'You are the {name} ' in contents)
            selected = dict(upstream='Supply risk.' if stage == 0 else None,
                            downstream='Limited availability.' if stage == 3 else None)
            text = json.dumps(selected)
        else:
            text = '[[[2]]]'
        return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(
            prompt_token_count=10, candidates_token_count=2, thoughts_token_count=3, total_token_count=15))

    def run_provider(self, provider, **kwargs):
        return run_episode(env_creator(self.config), provider, 'test', self.config,
                           metrics=provider.metrics, provider_metadata=provider.metadata, **kwargs)

    def test_strict_selection_and_boundaries(self):
        self.assertEqual(parse_sharing_response('{"upstream":null,"downstream":null}', 0),
                         {'upstream': None, 'downstream': None})
        for raw, stage in [('{"upstream":null,"downstream":"Hi"}', 0),
                           ('{"upstream":"Hi","downstream":null}', 3),
                           ('{"upstream":true,"downstream":null}', 1),
                           ('{"upstream":"","downstream":null}', 1),
                           ('{"upstream":null,"downstream":null,"extra":1}', 1),
                           ('{"upstream":null,"upstream":null,"downstream":null}', 1),
                           (json.dumps({'upstream': 'x'*201, 'downstream': None}), 1)]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_sharing_response(raw, stage)

    def test_only_selected_current_messages_and_correct_usage(self):
        self.client.models.generate_content.side_effect = self.generation
        provider = self.provider()
        result = self.run_provider(provider)
        self.assertEqual(result['run_status'], 'success', result['error_message'])
        trace = result['communication_trace']
        self.assertEqual(len(trace), 6)
        self.assertTrue(all((m['sender'], m['receiver']) in
                            [('retailer', 'wholesaler'), ('manufacturer', 'distributor')] for m in trace))
        metrics = result['coordination_metrics']
        self.assertEqual(metrics['decision_llm_call_count'], 12)
        self.assertEqual(metrics['coordination_llm_call_count'], 12)
        self.assertEqual(metrics['coordination_input_tokens'], 120)
        self.assertEqual(metrics['coordination_output_tokens'], 60)
        for x in provider.invocation_trace:
            if x['phase'] == 'ordering':
                m = x['incoming_messages']
                self.assertEqual(m['upstream'] is not None, x['stage_index'] == 2)
                self.assertEqual(m['downstream'] is not None, x['stage_index'] == 1)
                for message in m.values():
                    if message:
                        self.assertEqual(message['source_period'], x['decision_period'])
        for week in range(3):
            calls = provider.invocation_trace[week*8:(week+1)*8]
            self.assertEqual([c['call_kind'] for c in calls], ['coordination']*4 + ['decision']*4)
            if week == 1:
                self.assertEqual([c['observation']['latest_order'] for c in calls[:4]], [2, 2, 2, 2])

    def test_no_sharing_still_places_orders(self):
        self.client.models.generate_content.side_effect = lambda **kw: SimpleNamespace(
            text='{"upstream":null,"downstream":null}' if
            kw['config'].system_instruction.startswith('Decide independently') else '[[[2]]]',
            usage_metadata=None)
        result = self.run_provider(self.provider())
        self.assertEqual(result['run_status'], 'success', result['error_message'])
        self.assertEqual(result['communication_trace'], [])
        self.assertTrue(all(x['executed_order_quantity'] == 2 for x in result['action_trace']))

    def test_resume_during_sharing_and_ordering(self):
        self.client.models.generate_content.side_effect = self.generation
        reference = self.run_provider(self.provider())
        for fail_at in (3, 7, 11, 15):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'run.checkpoint.json'
                provider = self.provider()
                original_share = provider.share_information
                original_attempt = provider._attempt
                events = [0]
                def sharing(stage, observation, context):
                    events[0] += 1
                    if events[0] == fail_at:
                        raise RuntimeError('interruption')
                    return original_share(stage, observation, context)
                def attempt(stage, observation, context, number):
                    if context.get('phase') != 'controlled_information_sharing':
                        events[0] += 1
                        if events[0] == fail_at:
                            raise RuntimeError('interruption')
                    return original_attempt(stage, observation, context, number)
                provider.share_information = sharing
                provider._attempt = attempt
                failed = self.run_provider(provider, checkpoint_path=path)
                self.assertEqual(failed['run_status'], 'failed')
                checkpoint = json.loads(path.read_text())
                resumed = self.run_provider(self.provider(), checkpoint_path=path,
                                            resume_checkpoint=checkpoint)
                self.assertEqual(resumed['run_status'], 'success', resumed['error_message'])
                self.assertEqual(resumed['action_trace'], reference['action_trace'])
                self.assertEqual(resumed['communication_trace'], reference['communication_trace'])
                for key in ('coordination_message_count', 'llm_call_count', 'total_tokens'):
                    self.assertEqual(resumed['coordination_metrics'][key], reference['coordination_metrics'][key])

    def test_sharing_retries_are_coordination_calls(self):
        provider = self.provider()
        self.client.models.generate_content.side_effect = [
            SimpleNamespace(text='bad', usage_metadata=None),
            SimpleNamespace(text='{"upstream":null,"downstream":null}', usage_metadata=None)]
        obs = dict(inventory=5, backlog=0, upstream_backlog=0, latest_order=None,
                   deliveries=[0, 0], total_cost_so_far=0, recent_observations=[],
                   next_period_demand_forecast=None, prod_capacity=7)
        prompt = build_sharing_prompt(1, obs, {'period': 1})
        self.assertIn("Last week's realized downstream demand/order", prompt)
        self.assertNotIn('0 to 7', prompt)
        provider.share_information(1, obs, {'period': 1})
        self.assertEqual(provider.metrics.coordination_llm_call_count, 2)
        self.assertEqual(provider.metrics.decision_llm_call_count, 0)

    def test_runner_persists_separate_method_and_summary(self):
        self.client.models.generate_content.side_effect = self.generation
        with tempfile.TemporaryDirectory() as temp:
            status = run_experiments.run_record(
                load_series()[0], dataset='synthetic', method='controlled_adjacent',
                periods=3, output_root=temp)
            self.assertEqual(status, 0)
            folder = Path(temp) / 'gemini_controlled_adjacent_communication'
            row = json.loads((folder / 'summary.json').read_text())[0]
            self.assertEqual(row['architecture'], 'controlled_adjacent_coordination')
            self.assertEqual(row['coordination_message_count'], 6)
            self.assertEqual(row['completed_periods'], 3)
            self.assertEqual(row['coordination_llm_call_count'], 12)
            self.assertEqual(row['decision_llm_call_count'], 12)
            self.assertTrue((folder / 'smooth_001_run_001.checkpoint.json').exists())


if __name__ == '__main__':
    unittest.main()
