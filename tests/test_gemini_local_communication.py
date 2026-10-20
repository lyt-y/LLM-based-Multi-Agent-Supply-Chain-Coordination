"""Offline tests for the adjacent Gemini architecture; no live API calls."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from evaluation.metrics import CoordinationMetrics
from simulation.config import make_m5_config
from simulation.env import env_creator
from simulation.experiment_interface import DecisionResult, run_episode
from agents.gemini_local_communication import (
    GeminiLocalCommunicationProvider, build_prompt, parse_response, preflight, EXPERIMENT_ID,
)
from demand.m5_demand import get_representative
from agents.gemini_provider import REPRESENTATIVE_ID, build_prompt as build_no_communication_prompt
import agents.gemini_local_communication as local
def observation():
    current = dict(inventory_level=4, backlog=1, upstream_backlog=6, latest_order=None,
                   in_transit_orders=[2, 0], total_cost_so_far=3)
    return dict(inventory=4, backlog=1, upstream_backlog=6, latest_order=None, deliveries=[2, 0],
                total_cost_so_far=3, recent_observations=[current],
                next_period_demand_forecast=None, downstream_demand_history=[], prod_capacity=7)


def context(stage=0, order=2):
    snapshot = dict(inventory_level=20, backlog=3, in_transit_orders=[4, 5])
    return dict(period=1,
                downstream_opening_message=None if stage == 0 else dict(snapshot),
                upstream_message=None if stage == 3 else dict(snapshot),
                downstream_message=None if stage == 0 else dict(
                    current_order_to_upstream=order, planning_insight='Maintain replenishment.'))


def reply(local_state=None, order=2, stage=0, insight='Maintain replenishment.'):
    message = None if stage == 3 else dict(
        current_order_to_upstream=order, planning_insight=insight)
    return f'[[[{order}]]]\n' + json.dumps(message)


def offline_generation(*, model, contents, config):
    current = json.JSONDecoder().raw_decode(
        contents.split('Your current observations are:\n\n')[1])[0]
    if 'at your Wholesaler stage.' in contents:
        stage = 1
    elif 'at your Distributor stage.' in contents:
        stage = 2
    elif 'at your Manufacturer stage.' in contents:
        stage = 3
    else:
        stage = 0
    return SimpleNamespace(text=reply(current, order=0, stage=stage), usage_metadata=None)


class LocalCommunicationTests(unittest.TestCase):
    def setUp(self):
        key = patch.dict(os.environ, {'GEMINI_API_KEY': 'offline-test-key'})
        self.addCleanup(key.stop)
        key.start()
        dotenv = patch('agents.gemini_provider.load_dotenv')
        self.addCleanup(dotenv.stop)
        dotenv.start()
        client = patch('agents.gemini_provider.genai.Client')
        self.addCleanup(client.stop)
        self.client_factory = client.start()
        sleep = patch('agents.gemini_provider.time.sleep')
        self.addCleanup(sleep.stop)
        sleep.start()
        self.metrics = CoordinationMetrics()
        self.provider = GeminiLocalCommunicationProvider(self.metrics)

    def test_p1_to_p5_are_identical_and_message_is_adjacent(self):
        obs = observation()
        baseline = build_no_communication_prompt(0, obs, {'period': 1})
        prompt = build_prompt(0, obs, context())
        self.assertEqual(prompt.split('P6 ')[0], baseline.split('P6 ')[0])
        self.assertIn('P6 Information and communication rule: adjacent agents', prompt)
        self.assertIn('null', prompt)
        with self.assertRaises(ValueError):
            build_prompt(1, obs, {'period': 1, 'downstream_message': None})

    def test_strict_order_and_insight_validation(self):
        obs = observation()
        self.assertEqual(parse_response(reply(), 0, obs)[0], 2)
        invalid = (
            '[[2]]', '[[[2]]]', '[[[2]]]\nnull', '[[[2]]]\n{}', None,
            reply(order=2).replace('"current_order_to_upstream": 2', '"current_order_to_upstream": 3'),
            '[[[2]]]\n{"current_order_to_upstream":true,"planning_insight":"Good."}',
            '[[[2]]]\n{"current_order_to_upstream":2,"planning_insight":"Good.","extra":1}',
            '[[[2]]]\n{"current_order_to_upstream":2,"planning_insight":"Good.","planning_insight":"Bad."}',
            reply(insight=''), reply(insight='x' * 201), reply(insight='Two\nlines'),
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_response(raw, 0, obs)
        self.assertEqual(parse_response(reply(stage=3), 3, obs), (2, None))
        with self.assertRaises(ValueError):
            parse_response(reply(), 3, obs)

    def test_malformed_message_retries_same_prompt(self):
        obs = observation()
        good = reply(obs['recent_observations'][0])
        self.client_factory.return_value.models.generate_content.side_effect = [
            SimpleNamespace(text='[[2]]\n{}', usage_metadata=None),
            SimpleNamespace(text=good, usage_metadata=None),
        ]
        decision = self.provider(0, obs, context())
        self.assertEqual(decision.proposed_order_quantity, 2)
        self.assertEqual(self.metrics.decision_llm_call_count, 2)
        self.assertEqual(self.provider.metadata['malformed_output_count'], 1)
        calls = self.client_factory.return_value.models.generate_content.call_args_list
        self.assertEqual(calls[0], calls[1])

    def test_full_offline_episode_routes_only_neighbor_messages(self):
        record = get_representative(REPRESENTATIVE_ID)
        config = make_m5_config(record)
        self.client_factory.return_value.models.generate_content.side_effect = offline_generation
        result = run_episode(env_creator(config), self.provider, 'm5', config,
                             metrics=self.metrics, provider_metadata=self.provider.metadata)
        self.assertEqual(result['run_status'], 'success')
        self.assertEqual(result['completed_periods'], 277)
        self.assertEqual(len(result['action_trace']), 1108)
        self.assertEqual(len(result['communication_trace']), 2493)
        counts = result['coordination_metrics']
        self.assertEqual(counts['decision_llm_call_count'], 1108)
        self.assertEqual(counts['coordination_llm_call_count'], 0)
        self.assertEqual(counts['llm_call_count'], 1108)
        self.assertEqual(counts['coordination_message_count'], 2493)
        self.assertEqual(counts['coordination_round_count'], 277)
        self.assertEqual(counts['active_communication_links'], 6)
        calls = result['provider_invocation_trace']
        self.assertTrue(all(call['phase'] == 'ordering_and_insight' for call in calls))
        self.assertIsNone(calls[0]['incoming_message'])
        self.assertEqual(calls[1]['incoming_message']['planning_insight'], 'Maintain replenishment.')
        self.assertEqual(calls[1]['incoming_message']['current_order_to_upstream'], 0)


    def test_single_call_sequential_advice_and_frozen_opening_ledger(self):
        config = make_m5_config(get_representative(REPRESENTATIVE_ID))
        config['num_periods'] = 4
        config['init_inventories'] = [1, 2, 3, 4]
        config['prod_capacities'] = [3] * 4
        env = env_creator(config)
        events = []

        def generate(*, model, contents, config):
            stage = next(index for index, name in enumerate(local.STAGES)
                         if f'You are the {name} in' in contents)
            events.append(stage)
            week = (len(events) - 1) // 4 + 1
            self.assertEqual(self.metrics.coordination_message_count,
                             (week - 1) * 9 + 6 + stage)
            self.assertIn('It is the SAME order shown as latest_order in P3', contents)
            self.assertIn('There is no\nseparate insight-generation LLM call.', contents)
            return SimpleNamespace(
                text=reply(order=2, stage=stage, insight=f'Stage {stage} week {week} needs replenishment.'),
                usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=3,
                                               thoughts_token_count=2, total_token_count=15))

        self.client_factory.return_value.models.generate_content.side_effect = generate
        result = run_episode(env, self.provider, 'm5', config, metrics=self.metrics)
        self.assertEqual(result['run_status'], 'success', result['error_message'])
        self.assertEqual(events, list(range(4)) * 4)
        calls = result['provider_invocation_trace']
        for week in range(1, 5):
            decisions = calls[(week - 1) * 4:week * 4]
            for stage, call in enumerate(decisions):
                obs = call['observation']
                self.assertEqual(obs['latest_order'], 2 if stage else
                                 None if week == 1 else int(env.demands[week - 1]))
                due = int(env.arriving_orders[stage, week - 2]) if week >= 3 else 0
                self.assertEqual(obs['inventory'], int(env.inventories[stage, week - 1]) + due)
                self.assertEqual(obs['backlog'], int(env.backlogs[stage, week - 1]))
                self.assertEqual(len(obs['recent_observations']), week - 1)
                self.assertEqual(len(obs['downstream_demand_history']), week - 1)
                for key, neighbor in (('incoming_downstream_opening_message', stage - 1),
                                      ('incoming_upstream_message', stage + 1)):
                    message = call[key]
                    if not 0 <= neighbor < 4:
                        self.assertIsNone(message)
                        continue
                    neighbor_obs = decisions[neighbor]['observation']
                    self.assertEqual(message, dict(
                        inventory_level=neighbor_obs['inventory'], backlog=neighbor_obs['backlog'],
                        in_transit_orders=neighbor_obs['deliveries']))
                if stage == 0:
                    self.assertIsNone(call['incoming_message'])
                else:
                    self.assertEqual(call['incoming_message'], dict(
                        current_order_to_upstream=2,
                        planning_insight=f'Stage {stage - 1} week {week} needs replenishment.'))
            messages = result['communication_trace'][(week - 1) * 9:week * 9]
            self.assertEqual([msg['phase'] for msg in messages],
                             ['opening_information_sharing'] * 6 + ['sequential_order_and_insight'] * 3)
            for msg in messages:
                self.assertEqual(msg['outgoing_message'], msg['delivered_message'])
                expected = local.OPENING_MESSAGE_FIELDS if msg['phase'] == 'opening_information_sharing' else local.MESSAGE_FIELDS
                self.assertEqual(set(msg['delivered_message']), expected)
        counts = result['coordination_metrics']
        self.assertEqual(counts['coordination_llm_call_count'], 0)
        self.assertEqual(counts['decision_llm_call_count'], 16)
        self.assertEqual(counts['decision_input_tokens'], 160)
        self.assertEqual(counts['decision_output_tokens'], 80)
        self.assertEqual(counts['coordination_total_tokens'], 0)
        reference = run_episode(env_creator(config), lambda *args: 2, 'm5', config)
        self.assertEqual(result['episode_reward'], reference['episode_reward'])
        for actual, expected in zip(result['action_trace'], reference['action_trace']):
            for key in ('operational_cost', 'holding_cost', 'backlog_cost',
                        'fixed_ordering_cost', 'executed_order_quantity', 'observation'):
                self.assertEqual(actual[key], expected[key])

    def test_prompt_boundaries_and_downstream_order_consistency(self):
        obs = observation()
        for stage in range(4):
            current = dict(obs, latest_order=None if stage == 0 else 2)
            ctx = context(stage)
            prompt = build_prompt(stage, current, ctx)
            self.assertIn('Opening snapshots contain no insight', prompt)
            self.assertIn('You do not receive an upstream insight', prompt)
            self.assertIn('Return exactly two lines.', prompt)
            for field in ('upstream_message', 'downstream_opening_message'):
                if ctx[field] is not None:
                    altered = dict(ctx, **{field: dict(ctx[field], planning_insight='Hidden advice.')})
                    with self.assertRaises(ValueError):
                        build_prompt(stage, current, altered)
            if stage > 0:
                altered = dict(ctx, downstream_message=dict(ctx['downstream_message'],
                                                           current_order_to_upstream=3))
                with self.assertRaises(ValueError):
                    build_prompt(stage, current, altered)
            else:
                altered = dict(ctx, downstream_message=dict(
                    current_order_to_upstream=2, planning_insight='Invalid customer message.'))
                with self.assertRaises(ValueError):
                    build_prompt(stage, current, altered)

    def test_ordering_failure_retains_opening_sharing_without_executing_period(self):
        self.client_factory.return_value.models.generate_content.return_value = SimpleNamespace(
            text='malformed order', usage_metadata=None)
        config = make_m5_config(get_representative(REPRESENTATIVE_ID))
        config['num_periods'] = 2
        env = env_creator(config)
        with patch.object(env, 'step', wraps=env.step) as step:
            result = run_episode(env, self.provider, 'm5', config, metrics=self.metrics)
            step.assert_not_called()
        self.assertEqual(result['run_status'], 'failed')
        self.assertEqual(result['completed_periods'], 0)
        self.assertEqual(result['action_trace'], [])
        self.assertEqual(len(result['communication_trace']), 6)
        self.assertEqual(self.metrics.coordination_llm_call_count, 0)
        self.assertEqual(self.metrics.decision_llm_call_count,
                         1 + self.provider.metadata['retry_policy']['adapter_retries'])
        self.assertTrue(all(call['phase'] == 'ordering_and_insight'
                            for call in result['provider_invocation_trace']))

    def test_later_failure_delivers_only_validated_preceding_advice(self):
        config = make_m5_config(get_representative(REPRESENTATIVE_ID))
        config['num_periods'] = 2
        def generate(*, model, contents, config):
            return offline_generation(model=model, contents=contents, config=config) if 'You are the Retailer in' in contents else SimpleNamespace(
                text='[[[0]]]\n{"current_order_to_upstream":1,"planning_insight":"Mismatch."}',
                usage_metadata=None)
        self.client_factory.return_value.models.generate_content.side_effect = generate
        env = env_creator(config)
        with patch.object(env, 'step', wraps=env.step) as step:
            result = run_episode(env, self.provider, 'm5', config, metrics=self.metrics)
            step.assert_not_called()
        self.assertEqual(result['run_status'], 'failed')
        self.assertEqual(result['completed_periods'], 0)
        self.assertEqual(len(result['communication_trace']), 7)
        self.assertEqual(result['communication_trace'][-1]['sender'], 'retailer')
        self.assertEqual(result['communication_trace'][-1]['receiver'], 'wholesaler')
        self.assertEqual(result['communication_trace'][-1]['phase'], 'sequential_order_and_insight')

    def test_recipient_mutation_does_not_change_frozen_snapshots(self):
        class Provider:
            uses_local_communication = True

            def __call__(self, stage, observation, context):
                for key in ('upstream_message', 'downstream_opening_message'):
                    if context[key] is not None:
                        context[key]['inventory_level'] = 987654321
                        context[key]['in_transit_orders'][0] = 987654321
                if context['downstream_message'] is not None:
                    context['downstream_message']['planning_insight'] = '987654321'
                return DecisionResult(raw_output='offline', proposed_order_quantity=0,
                                      outgoing_message=None if stage == 3 else dict(
                                          current_order_to_upstream=0,
                                          planning_insight='Maintain replenishment.'))

        config = make_m5_config(get_representative(REPRESENTATIVE_ID))
        config['num_periods'] = 2
        result = run_episode(env_creator(config), Provider(), 'm5', config)
        self.assertEqual(result['run_status'], 'success')
        self.assertNotIn('987654321', json.dumps(result))

    def test_entrypoint_persists_separate_detail_and_summary(self):
        record = get_representative(REPRESENTATIVE_ID)
        config = make_m5_config(record)
        config['num_periods'] = 3
        config['_provenance']['demand_window'] = 'weeks_1_to_3'
        config['_provenance']['calibration_window'] = 'weeks_1_to_130'
        self.client_factory.return_value.models.generate_content.side_effect = offline_generation
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / EXPERIMENT_ID / (REPRESENTATIVE_ID + '_run_001.json')
            destination.parent.mkdir()
            with patch.object(local, 'preflight', return_value=(self.provider, config, destination)):
                self.assertEqual(local.main([]), 0)
            saved = json.loads(destination.read_text())
            summary = json.loads((destination.parent / 'summary.json').read_text())[0]
            self.assertEqual(saved['run_status'], 'success')
            self.assertEqual(saved['run_metadata']['provider']['architecture'], 'local_communication')
            self.assertEqual(saved['completed_periods'], 3)
            self.assertEqual(len(saved['communication_trace']), 27)
            self.assertEqual(summary['coordination_message_count'], 27)
            self.assertEqual(summary['coordination_llm_call_count'], 0)
            self.assertEqual(summary['decision_llm_call_count'], 12)
            self.assertEqual(summary['api_invocation_count'], 12)
            self.assertEqual(summary['run_id'], 'run_001')
            self.assertEqual(summary['num_periods'], 3)
            self.assertEqual(summary['demand_window'], 'weeks_1_to_3')

    def test_preflight_uses_separate_result_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            provider, config, path = preflight(output_root=directory)
            try:
                self.assertEqual(path.parent.name, EXPERIMENT_ID)
                self.assertEqual(path.name, REPRESENTATIVE_ID + '_run_001.json')
                self.assertEqual(config['num_periods'], 104)
                self.assertEqual(config['_provenance']['demand_window'], 'weeks_1_to_104')
                self.assertEqual(config['_provenance']['calibration_window'], 'weeks_1_to_130')
                self.client_factory.return_value.models.generate_content.assert_not_called()
            finally:
                provider.close()





if __name__ == '__main__':
    unittest.main()
