import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluation.metrics import CoordinationMetrics
from simulation.config import env_configs
from simulation.env import env_creator
from simulation.experiment_interface import (ActionValidationError, DecisionResult,
                                  json_safe, run_episode, validate_action)


class InterfaceTests(unittest.TestCase):
    def test_completed_histories_are_local_but_current_downstream_orders_are_visible(self):
        config = dict(env_configs['m5'], num_periods=30, prod_capacities=[100]*4,
                      demand_fn=lambda period: period * 3)
        snapshots = [[] for _ in range(4)]
        orders = [[] for _ in range(4)]

        def provider(stage, observation, context):
            period = context['period']
            self.assertEqual(observation['recent_observations'], snapshots[stage][-10:])
            expected_history = ([week*3 for week in range(max(1, period-26), period)]
                                if stage == 0 else orders[stage-1][max(0, period-27):period-1])
            self.assertEqual(observation['downstream_demand_history'], expected_history)
            expected_forecast = sum(expected_history)/len(expected_history) if expected_history else None
            self.assertEqual(observation['next_period_demand_forecast'], expected_forecast)
            self.assertEqual(observation['forecast_target_period'], period+1)
            current_order = ((period-1)*3 if period > 1 else None) if stage == 0 else orders[stage-1][-1]
            self.assertEqual(observation['latest_order'], current_order)
            snapshots[stage].append(dict(
                inventory_level=observation['inventory'], backlog=observation['backlog'],
                upstream_backlog=observation['upstream_backlog'],
                latest_order=observation['latest_order'], in_transit_orders=observation['deliveries'],
                total_cost_so_far=observation['total_cost_so_far']))
            quantity = (period + stage*11) % 7
            orders[stage].append(quantity)
            return quantity

        result = run_episode(env_creator(config), provider, 'test', config)
        self.assertEqual(result['run_status'], 'success', result['error_message'])
        self.assertEqual(result['completed_periods'], 30)

    def test_action_contract_rejects_without_clipping(self):
        valid = validate_action(np.int64(3), stage=1, period=2, capacity=5)
        self.assertEqual(valid.proposed_order_quantity, 3)
        self.assertEqual(valid.executed_order_quantity, 3)
        self.assertEqual(valid.validation_status, 'valid')
        for value in (-1, 6, 3.0):
            with self.assertRaises(ActionValidationError) as caught:
                validate_action(value, stage=1, period=2, capacity=5)
            self.assertEqual(caught.exception.result.proposed_order_quantity, value)
            self.assertEqual(caught.exception.result.executed_order_quantity, None)
            self.assertEqual(caught.exception.stage, 1)
            self.assertEqual(caught.exception.period, 2)

    def test_json_safe_and_provenance(self):
        result = run_episode(
            env_creator(env_configs['m5']),
            lambda stage, observation, context: 0,
            'm5',
            env_configs['m5'],
            provider_metadata={'kind': 'mock'},
        )
        encoded = json.dumps(result)
        decoded = json.loads(encoded)
        self.assertEqual(decoded['run_status'], 'success')
        self.assertEqual(decoded['completed_periods'], 277)
        self.assertEqual(decoded['run_metadata']['environment_config'], 'm5')
        self.assertIn('representative_record_id', decoded['run_metadata'])
        self.assertEqual(decoded['run_metadata']['q'], 0.95)
        self.assertEqual(decoded['run_metadata']['capacity_quantile'], 0.95)
        self.assertEqual(decoded['run_metadata']['capacity_scope'], 'new_order_quantity_only')
        self.assertEqual(decoded['run_metadata']['provider']['kind'], 'mock')
        self.assertEqual(len(decoded['action_trace']), 277 * 4)
        self.assertTrue(all(
            entry['proposed_order_quantity'] == entry['executed_order_quantity'] == 0
            for entry in decoded['action_trace']))
        self.assertTrue(all(e['attempt_count'] == 1 for e in decoded['action_trace']))
        self.assertEqual(json_safe(np.bool_(True)), True)

    def test_failure_preserves_partial_trace_and_metrics(self):
        calls = []

        def provider(stage, observation, context):
            calls.append(context['period'])
            if context['period'] == 2 and stage == 1:
                raise RuntimeError('mock provider failure')
            return 0

        result = run_episode(env_creator(env_configs['m5']), provider, 'm5', env_configs['m5'])
        self.assertEqual(result['run_status'], 'failed')
        self.assertEqual(result['completed_periods'], 1)
        self.assertEqual(result['error_type'], 'RuntimeError')
        self.assertIn('mock provider failure', result['error_message'])
        self.assertEqual(len(result['action_trace']), 4)
        self.assertTrue(all(e['period'] == 1 and e['validation_status'] == 'valid'
                            for e in result['action_trace']))
        self.assertEqual(result['coordination_metrics']['llm_call_count'], 0)

    def test_final_fallback_action_only_and_all_client_usage_counted(self):
        metrics = CoordinationMetrics()

        class Client:
            def create(self, quantity):
                return {'quantity': quantity,
                        'usage': {'prompt_tokens': 7, 'completion_tokens': 3}}

        client = Client()

        def decision(quantity):
            response = client.create(quantity)
            return DecisionResult(response, response['quantity'], decision_summary='external')

        with metrics.track_client(client):
            result = run_episode(
                env_creator(env_configs['m5']),
                lambda stage, observation, context: decision(-1),
                'm5', env_configs['m5'], metrics=metrics,
                fallback_handler=lambda error: decision(0),
            )
        self.assertEqual(result['run_status'], 'success')
        self.assertEqual(result['completed_periods'], 277)
        trace = result['action_trace']
        self.assertEqual(len(trace), 1108)
        self.assertEqual(len({(e['period'], e['stage']) for e in trace}), 1108)
        for entry in trace:
            self.assertEqual(entry['validation_status'], 'valid')
            self.assertEqual(entry['proposed_order_quantity'], 0)
            self.assertEqual(entry['executed_order_quantity'], 0)
            self.assertEqual(entry['raw_output']['quantity'], 0)
            self.assertEqual(entry['decision_summary'], 'external')
            self.assertEqual(entry['attempt_count'], 2)
            self.assertIsNotNone(entry['reward'])
        accounting = result['coordination_metrics']
        for prefix in ('', 'decision_'):
            self.assertEqual(accounting[prefix+'llm_call_count'], 2216)
            self.assertEqual(accounting[prefix+'input_tokens'], 2216 * 7)
            self.assertEqual(accounting[prefix+'output_tokens'], 2216 * 3)
            self.assertEqual(accounting[prefix+'total_tokens'], 2216 * 10)
        self.assertEqual(accounting['coordination_llm_call_count'], 0)
        json.dumps(result, allow_nan=False)

    def test_invalid_action_without_valid_fallback_fails_without_trace(self):
        for fallback in (None, lambda error: -2):
            with self.subTest(fallback=fallback):
                result = run_episode(
                    env_creator(env_configs['m5']), lambda *args: -1,
                    'm5', env_configs['m5'], fallback_handler=fallback,
                )
                self.assertEqual(result['run_status'], 'failed')
                self.assertEqual(result['completed_periods'], 0)
                self.assertEqual(result['error_type'], 'ActionValidationError')
                self.assertEqual(result['action_trace'], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
