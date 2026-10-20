import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from types import SimpleNamespace
import re

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulation.config import env_configs
from simulation.env import env_creator
from simulation.experiment_interface import run_episode
from evaluation.experiment_results import (
    bullwhip, bullwhip_cv_ratio, persist_run, COST_FIELDS, build_evaluation_summary,
)
from evaluation.metrics import CoordinationMetrics


class ResultTests(unittest.TestCase):
    def test_fulfillment_can_exceed_order_capacity_at_every_stage(self):
        config = dict(env_configs['m5'], num_periods=3, init_inventories=[20] * 4,
                      prod_capacities=[2] * 4, demand_fn=lambda period: 10)
        env = env_creator(config)
        env.reset()
        env.backlogs[:, 0] = [5, 6, 7, 8]
        env.update_state()
        _, _, _, _, infos = env.step({f'stage_{m}': 2 for m in range(4)})
        np.testing.assert_array_equal(env.sales[:, 1], [15, 8, 9, 10])
        np.testing.assert_array_equal(env.backlogs[:, 1], [0] * 4)
        np.testing.assert_array_equal(env.inventories[:, 1], [5, 12, 11, 10])
        np.testing.assert_array_equal(env.arriving_orders[:, 1], [8, 9, 10, 2])
        self.assertTrue(env.observation_space.contains(env.state_dict))
        for m in range(4):
            self.assertEqual(infos[f'stage_{m}']['operational_cost'], env.inventories[m, 1] + 1)
        for quantity in (3, -1, 1.5, True):
            invalid = env_creator(config)
            invalid.reset()
            with self.assertRaises(AssertionError):
                invalid.step({f'stage_{m}': quantity for m in range(4)})
            self.assertEqual(invalid.period, 0)

    def test_order_saturation_and_inventory_shortage_diagnostics(self):
        from evaluation.experiment_results import add_service_inventory_summary
        config = dict(env_configs['m5'], num_periods=1, init_inventories=[10] * 4,
                      prod_capacities=[2] * 4, demand_fn=lambda period: 12)
        env = env_creator(config)
        result = run_episode(env, lambda *args: 2, 'm5', config)
        add_service_inventory_summary(result, env, config)
        for stage in result['evaluation_summary']['stage_metrics'].values():
            self.assertEqual(stage['capacity_saturation_pct'], 100.)
            self.assertIsNone(stage['strict_capacity_binding_pct'])
        self.assertEqual(result['evaluation_summary']['stage_metrics']['stage_0']['supply_shortage_pct'], 100.)

    def test_default_env_creator_uses_current_m5_configuration(self):
        expected = env_configs['m5']
        env = env_creator(None)
        self.assertEqual(env.num_stages, expected['num_stages'])
        self.assertEqual(env.num_periods, expected['num_periods'])
        np.testing.assert_array_equal(env.init_inventories, expected['init_inventories'])
        np.testing.assert_array_equal(env.prod_capacities, expected['prod_capacities'])
        env.reset()
        env.step({f'stage_{m}': 0 for m in range(env.num_stages)})
        self.assertEqual(env.demands[1], expected['demand_fn'](1))

    def test_global_bullwhip_product_preserves_components(self):
        names = env_configs['m5']['stage_names']
        metrics = CoordinationMetrics().to_dict()
        for components, expected in (([2., 3., 4., 5.], 120.), ([2., 3., 4., 10.], 240.)):
            with patch('evaluation.experiment_results.bullwhip', side_effect=components):
                summary = build_evaluation_summary([], names, [], metrics)
            self.assertEqual(summary['global_bullwhip'], expected)
            self.assertEqual(list(summary['bullwhip'].values()), components)
            self.assertEqual([summary['stage_metrics'][f'stage_{m}']['bullwhip']
                              for m in range(4)], components)
            self.assertEqual(set(summary), {'operational', 'llm_resource', 'time',
                                           'coordination', 'bullwhip', 'bullwhip_cv',
                                           'stage_metrics', 'global_bullwhip'})

    def test_global_bullwhip_undefined_nonfinite_and_stage_count(self):
        names = env_configs['m5']['stage_names']
        metrics = CoordinationMetrics().to_dict()
        for bad in (None, float('nan'), float('inf'), -float('inf')):
            for stage in range(4):
                components = [0., 2., 3., 4.]
                components[stage] = bad
                with patch('evaluation.experiment_results.bullwhip', side_effect=components):
                    summary = build_evaluation_summary([], names, [], metrics)
                self.assertIsNone(summary['global_bullwhip'])
                self.assertIs(summary['stage_metrics'][f'stage_{stage}']['bullwhip'], bad)
        for count in (0, 3, 5):
            with self.assertRaisesRegex(ValueError, 'exactly four'):
                build_evaluation_summary([], ['stage']*count, [], metrics)
        with patch('evaluation.experiment_results.bullwhip', side_effect=[1e308]*4):
            self.assertIsNone(build_evaluation_summary([], names, [], metrics)['global_bullwhip'])
        with patch('evaluation.experiment_results.bullwhip', side_effect=[0., 2., 3., 4.]):
            self.assertEqual(build_evaluation_summary([], names, [], metrics)['global_bullwhip'], 0.)

    def test_config_observation_and_fixed_cost_independent_of_fulfilment(self):
        config = env_configs['m5']
        for key in ('sale_prices', 'order_costs', 'backlog_to_holding_ratio'):
            self.assertNotIn(key, config)
        for key in ('holding_costs', 'backlog_costs', 'fixed_order_costs'):
            self.assertEqual(config[key], [1]*4)
        for quantity in (0, 1, config['prod_capacities'][0]):
            env = env_creator(config)
            env.reset()
            # No upstream inventory: positive orders can have zero fulfilment.
            env.inventories[:, 0] = 0
            env.update_state()
            observation = env.parse_state()['stage_0']
            self.assertNotIn('sale_price', observation)
            self.assertNotIn('order_cost', observation)
            self.assertEqual(observation['fixed_order_cost'], 1)
            _, rewards, _, _, infos = env.step({f'stage_{m}': quantity for m in range(4)})
            self.assertEqual(env.arriving_orders[0, 1], 0)
            for m in range(4):
                costs = infos[f'stage_{m}']
                self.assertEqual(costs['fixed_ordering_cost'], int(quantity > 0))
                self.assertEqual(costs['holding_cost'], env.inventories[m, 1])
                self.assertEqual(costs['backlog_cost'], env.backlogs[m, 1])
                self.assertEqual(costs['operational_cost'], sum(costs[k] for k in COST_FIELDS[:3]))
                self.assertEqual(rewards[f'stage_{m}'], -costs['operational_cost'])

    def test_cost_history_is_not_mutated_by_later_deliveries(self):
        env = env_creator(env_configs['m5'])
        env.reset()
        actions = {f'stage_{m}': 4 for m in range(4)}
        env.step(actions)
        _, _, _, _, infos = env.step(actions)
        end_inventory = env.inventories[:, 2].copy()
        saved_costs = {key: values[:, :3].copy() for key, values in env.period_costs.items()}
        env.step(actions)
        np.testing.assert_array_equal(env.inventories[:, 2], end_inventory)
        for key, expected in saved_costs.items():
            np.testing.assert_array_equal(env.period_costs[key][:, :3], expected)
        for m in range(4):
            self.assertEqual(infos[f'stage_{m}']['holding_cost'], end_inventory[m])

    def test_opening_receipts_are_observed_once_and_excluded_from_pipeline(self):
        config = dict(env_configs['m5'], num_periods=6, init_inventories=[10]*4,
                      prod_capacities=[10]*4, demand_fn=lambda period: 0)
        env = env_creator(config)
        env.reset()
        env.step({f'stage_{m}': 4 for m in range(4)})
        env.step({f'stage_{m}': 3 for m in range(4)})
        end_stock = env.inventories[:, 2].copy()
        observed = env.parse_state()
        for stage in range(4):
            row = observed[f'stage_{stage}']
            self.assertEqual(row['inventory'], end_stock[stage] + 4)
            self.assertEqual(row['deliveries'], [3, 0])
        env.update_state()
        self.assertEqual(env.parse_state(), observed)
        np.testing.assert_array_equal(env.inventories[:, 2], end_stock)
        env.step({f'stage_{m}': 0 for m in range(4)})
        np.testing.assert_array_equal(env.inventories[:, 3], end_stock + 4)
        for stage in range(4):
            self.assertEqual(env.parse_state()[f'stage_{stage}']['inventory'], end_stock[stage] + 7)

    def test_opening_inventory_and_future_pipeline_for_different_lead_times(self):
        config = dict(env_configs['m5'], num_periods=6, lead_times=[1, 2, 3, 2])
        env = env_creator(config)
        env.reset()
        env.period = 3
        env.inventories[:, 3] = 10
        env.arriving_orders[:, 1:4] = [1, 2, 3]
        env.update_state()
        observed = env.parse_state()
        for stage, (inventory, pipeline) in enumerate(((13, [0, 0, 0]),
                                                       (12, [3, 0, 0]),
                                                       (11, [2, 3, 0]),
                                                       (12, [3, 0, 0]))):
            self.assertEqual(observed[f'stage_{stage}']['inventory'], inventory)
            self.assertEqual(observed[f'stage_{stage}']['deliveries'], pipeline)
        np.testing.assert_array_equal(env.inventories[:, 3], [10]*4)

    def test_full_run_aggregation_timing_and_persistence(self):
        env = env_creator(env_configs['m5'])
        def provider(stage, observation, context):
            return (context['period'] + 3 * stage) % (int(observation['prod_capacity']) + 1)
        with patch('evaluation.metrics.perf_counter', side_effect=[10., 12.5]):
            result = run_episode(env, provider, 'm5', env_configs['m5'])
        self.assertEqual(result['run_status'], 'success')
        self.assertEqual(result['completed_periods'], 277)
        self.assertEqual(len(result['action_trace']), 1108)
        self.assertEqual(result['coordination_metrics']['wall_clock_seconds'], 2.5)
        summary = result['evaluation_summary']
        self.assertEqual(summary['time']['wall_clock_seconds'], 2.5)
        for entry in result['action_trace']:
            self.assertEqual(entry['reward'], -entry['operational_cost'])
            self.assertEqual(entry['proposed_order_quantity'], entry['executed_order_quantity'])
            self.assertEqual(entry['attempt_count'], 1)
        for m in range(4):
            entries = [e for e in result['action_trace'] if e['stage'] == m]
            stage = summary['stage_metrics'][f'stage_{m}']
            for key in COST_FIELDS:
                self.assertEqual(stage[key], sum(e[key] for e in entries))
            self.assertEqual(stage['operational_cost'], sum(stage[k] for k in COST_FIELDS[:3]))
            self.assertEqual(stage['positive_order_periods'], stage['fixed_ordering_cost'])
            self.assertEqual(stage['total_order_quantity'], int(env.orders[m, 1:].sum()))
            expected = float(np.var(env.orders[m, 1:], ddof=0) / np.var(env.demands[1:], ddof=0))
            received_demand = env.demands[1:] if m == 0 else env.orders[m - 1, 1:]
            expected_cv = bullwhip_cv_ratio(env.orders[m, 1:].tolist(), received_demand.tolist())
            self.assertEqual(stage['bullwhip'], expected)
            self.assertEqual(stage['bullwhip_cv'], expected_cv)
            self.assertEqual(summary['bullwhip'][f'stage_{m}'], expected)
            self.assertEqual(summary['bullwhip_cv'][f'stage_{m}'], expected_cv)
        operational = summary['operational']
        self.assertEqual(summary['global_bullwhip'], math.prod(summary['bullwhip'].values()))
        self.assertEqual(operational['global_operational_cost'], -result['episode_reward'])
        self.assertEqual(operational['global_operational_cost'], sum(operational[k] for k in COST_FIELDS[:3]))
        self.assertEqual(operational['global_operational_cost'], sum(s['operational_cost'] for s in summary['stage_metrics'].values()))
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        with tempfile.TemporaryDirectory() as directory:
            path = persist_run(result, 'offline', 'run_1', output_root=directory)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), result)
            self.assertEqual(json.loads(path.read_text())['evaluation_summary']['global_bullwhip'],
                             math.prod(summary['bullwhip'].values()))
            with self.assertRaises(FileExistsError):
                persist_run(result, 'offline', 'run_1', output_root=directory)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), result)
            self.assertEqual(list(path.parent.iterdir()), [path])
            with self.assertRaises(ValueError):
                persist_run(result, '../escape', 'run', output_root=directory)

    def test_failed_run_partial_costs_and_persistence(self):
        def provider(stage, observation, context):
            return -1 if context['period'] == 2 and stage == 1 else 0
        result = run_episode(env_creator(env_configs['m5']), provider, 'm5', env_configs['m5'])
        self.assertEqual(result['run_status'], 'failed')
        self.assertEqual(result['completed_periods'], 1)
        self.assertEqual(len(result['action_trace']), 4)
        self.assertEqual(result['evaluation_summary']['operational']['global_operational_cost'], -result['episode_reward'])
        self.assertTrue(all(v is None for v in result['evaluation_summary']['bullwhip'].values()))
        self.assertIsNone(result['evaluation_summary']['global_bullwhip'])
        with tempfile.TemporaryDirectory() as directory:
            path = persist_run(result, 'offline', 'failed', output_root=directory)
            self.assertEqual(json.loads(path.read_text()), result)

    def test_bullwhip_population_ratio_and_undefined_cases(self):
        self.assertEqual(bullwhip([0, 2, 4], [0, 1, 2]), 4.)
        self.assertEqual(bullwhip([3, 3, 3], [0, 1, 2]), 0.)
        self.assertIsNone(bullwhip([0, 2, 4], [1, 1, 1]))
        self.assertIsNone(bullwhip([], []))
        with self.assertRaises(ValueError):
            bullwhip([1], [])

    def test_bullwhip_cv_ratio_and_downstream_demand(self):
        self.assertEqual(bullwhip_cv_ratio([0, 2, 4], [0, 1, 2]), 1.)
        self.assertEqual(bullwhip_cv_ratio([3, 3, 3], [1, 2, 3]), 0.)
        self.assertIsNone(bullwhip_cv_ratio([1, 2, 3], [2, 2, 2]))
        self.assertIsNone(bullwhip_cv_ratio([0, 0, 0], [1, 2, 3]))
        self.assertIsNone(bullwhip_cv_ratio([], []))
        with self.assertRaises(ValueError):
            bullwhip_cv_ratio([1], [])

        names = env_configs['m5']['stage_names']
        metrics = CoordinationMetrics().to_dict()
        trace = []
        stage_orders = {
            0: [2, 4, 6],
            1: [1, 3, 9],
            2: [4, 4, 4],
            3: [0, 3, 6],
        }
        for period in range(1, 4):
            for stage in range(4):
                trace.append({
                    'period': period, 'stage': stage,
                    'executed_order_quantity': stage_orders[stage][period - 1],
                    **{field: 0 for field in COST_FIELDS},
                })
        summary = build_evaluation_summary(trace, names, [1, 2, 4], metrics)
        self.assertEqual(summary['stage_metrics']['stage_0']['bullwhip_cv'],
                         bullwhip_cv_ratio(stage_orders[0], [1, 2, 4]))
        for stage in range(1, 4):
            self.assertEqual(summary['stage_metrics'][f'stage_{stage}']['bullwhip_cv'],
                             bullwhip_cv_ratio(stage_orders[stage], stage_orders[stage - 1]))



class SSBaselineTests(unittest.TestCase):
    def test_fixed_26_week_minimum_history(self):
        from policies.ss import compute_s_S_from_training_window
        for length in (0, 1, 25):
            with self.assertRaisesRegex(ValueError, 'Insufficient training observations'):
                compute_s_S_from_training_window([1]*length, 2)
        fitted = compute_s_S_from_training_window(list(range(26)), 2)
        self.assertEqual(fitted['training_weeks'], 26)
        self.assertEqual(fitted, compute_s_S_from_training_window(list(range(26)) + [99999]*100, 2))

    def test_training_prefix_formulas_and_validation(self):
        from policies.ss import compute_s_S_from_training_window
        from scipy.stats import norm
        training = list(range(26))
        left = compute_s_S_from_training_window(training + [0]*251, 2)
        right = compute_s_S_from_training_window(training + [999999]*251, 2)
        self.assertEqual(left, right)
        self.assertEqual(left['mu'], np.mean(training))
        self.assertEqual(left['sigma'], np.std(training, ddof=1))
        self.assertEqual(left['z'], norm.ppf(.95))
        self.assertEqual(left['s'], int(np.ceil(np.mean(training)*2 + norm.ppf(.95)*np.std(training, ddof=1)*np.sqrt(2))))
        self.assertEqual(left['eoq_increment'], np.sqrt(2*np.mean(training)))
        self.assertEqual(left['S'], int(np.ceil(left['s'] + left['eoq_increment'])))
        self.assertIs(type(left['s']), int)
        self.assertIs(type(left['S']), int)
        self.assertGreaterEqual(left['S'], left['s'])
        zero = compute_s_S_from_training_window([0]*26, 2)
        self.assertEqual((zero['mu'], zero['sigma'], zero['s'], zero['S']), (0, 0, 0, 1))
        self.assertTrue(zero['zero_demand_fallback_applied'])
        self.assertFalse(left['zero_demand_fallback_applied'])
        self.assertEqual(zero, compute_s_S_from_training_window([0]*26 + [999999]*251, 2))
        for kwargs in ({'lead_time':0}, {'fixed_order_cost':-1}, {'holding_cost':0},
                       {'service_level':0}, {'service_level':1}, {'training_weeks':0},
                       {'training_weeks':1},
                       {'training_weeks':27}, {'lead_time':float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compute_s_S_from_training_window(training, **{'lead_time':2, **kwargs})

    def test_zero_demand_fallback_actions(self):
        from policies.ss import RuleBasedSS, compute_s_S_from_training_window
        policy = RuleBasedSS({'stage_0': compute_s_S_from_training_window([0]*26, 2)})
        for position, capacity, expected in [(1, 7, 0), (0, 7, 1), (0, 0, 0), (-2, 7, 3), (-9, 7, 7)]:
            with self.subTest(position=position, capacity=capacity):
                observation = dict(inventory=max(0, position), backlog=max(0, -position),
                                   upstream_backlog=0, deliveries=[0, 0], prod_capacity=capacity)
                self.assertEqual(policy(0, observation, None), expected)

    def test_inventory_position_boundaries_capacity_and_no_hidden_inputs(self):
        from policies.ss import RuleBasedSS
        params = {'stage_0': {'s':10, 'S':15}}
        policy = RuleBasedSS(params)
        params['stage_0']['S'] = 999
        exposed = policy.stage_parameters
        exposed['stage_0']['s'] = 999
        # IP = 8 + 3 + (2+1) - 4 = 10, exactly at threshold.
        obs = dict(inventory=8, upstream_backlog=3, deliveries=[2,1], backlog=4, prod_capacity=7)
        class ForbiddenContext:
            def __getitem__(self, key):
                raise AssertionError('The policy must not inspect demand/context')
        self.assertEqual(policy(0, obs, ForbiddenContext()), 5)
        self.assertEqual(policy(0, dict(obs, inventory=9), ForbiddenContext()), 0)
        self.assertEqual(policy(0, dict(obs, inventory=0), ForbiddenContext()), 7)
        self.assertEqual(policy.stage_parameters, {'stage_0': {'s':10, 'S':15}})

    def test_four_frozen_episodes_kpis_policy_and_persistence(self):
        from policies.ss import run_sS_baseline, compute_s_S_from_training_window, RuleBasedSS
        from demand.m5_demand import REGIMES, get_representative
        with tempfile.TemporaryDirectory() as directory:
            for regime in REGIMES:
                with self.subTest(regime=regime):
                    record = get_representative(regime=regime)
                    with patch('policies.ss.compute_s_S_from_training_window', wraps=compute_s_S_from_training_window) as estimator:
                        result = run_sS_baseline(record)
                    self.assertEqual(estimator.call_count, 4)
                    self.assertEqual(result['policy']['training_weeks'], 26)
                    self.assertEqual(result['policy']['standard_deviation_ddof'], 1)
                    self.assertEqual(result['run_status'], 'success')
                    self.assertEqual(result['completed_periods'], 277)
                    self.assertEqual(len(result['action_trace']), 1108)
                    self.assertEqual(result['policy']['training_demand'], record['weekly_demand'][:26])
                    parameters = result['policy']['stage_parameters']
                    for parameter in parameters.values():
                        self.assertEqual(parameter['zero_demand_fallback_applied'], regime == 'intermittent')
                        if regime == 'intermittent':
                            self.assertEqual((parameter['s'], parameter['S']), (0, 1))
                    policy = RuleBasedSS(parameters)
                    for entry in result['action_trace']:
                        self.assertEqual(entry['executed_order_quantity'], policy(entry['stage'], entry['observation'], None))
                        self.assertEqual(entry['proposed_order_quantity'], entry['executed_order_quantity'])
                        self.assertEqual(entry['attempt_count'], 1)
                    self.assertEqual(policy.stage_parameters, parameters)
                    summary = result['evaluation_summary']
                    op = summary['operational']
                    self.assertEqual(op['global_operational_cost'], sum(op[k] for k in COST_FIELDS[:3]))
                    self.assertEqual(op['global_operational_cost'], -result['episode_reward'])
                    for m in range(4):
                        entries = [e for e in result['action_trace'] if e['stage']==m]
                        stage = summary['stage_metrics'][f'stage_{m}']
                        self.assertEqual(stage['bullwhip'], bullwhip([e['executed_order_quantity'] for e in entries], record['weekly_demand']))
                        received_demand = record['weekly_demand'] if m == 0 else [
                            e['executed_order_quantity']
                            for e in result['action_trace'] if e['stage'] == m - 1
                        ]
                        self.assertEqual(stage['bullwhip_cv'], bullwhip_cv_ratio(
                            [e['executed_order_quantity'] for e in entries], received_demand))
                        self.assertEqual(stage['average_inventory'], np.mean([e['holding_cost'] for e in entries]))
                        self.assertEqual(stage['average_backlog'], np.mean([e['backlog_cost'] for e in entries]))
                        self.assertEqual(stage['backlog_frequency_pct'], 100*np.mean([e['backlog_cost']>0 for e in entries]))
                        self.assertEqual(stage['operational_cost'], sum(e['operational_cost'] for e in entries))
                    # Backlog conservation: total sales = total demand - terminal retail backlog.
                    terminal_backlog = result['action_trace'][-4]['backlog_cost']
                    self.assertEqual(summary['service_inventory']['cumulative_fulfilled_fraction'],
                                     (sum(record['weekly_demand'])-terminal_backlog)/sum(record['weekly_demand']))
                    for key, value in result['coordination_metrics'].items():
                        if key == 'wall_clock_seconds':
                            self.assertGreater(value, 0)
                        elif key == 'per_link_message_counts':
                            self.assertEqual(value, [])
                        else:
                            self.assertEqual(value, 0)
                    path = persist_run(result, 'ss_test', regime, output_root=directory)
                    self.assertEqual(json.loads(path.read_text()), result)


if __name__ == '__main__':
    unittest.main(verbosity=2)


class MetricsTests(unittest.TestCase):
    def test_empty_and_no_coordination(self):
        m = CoordinationMetrics()
        self.assertEqual(m.to_dict()['llm_call_count'], 0)
        m.record_call('decision', {'usage': {'prompt_tokens': 7, 'completion_tokens': 2}})
        d = m.to_dict()
        for key in ('coordination_llm_call_count', 'coordination_input_tokens', 'coordination_output_tokens',
                    'coordination_total_tokens', 'coordination_message_count', 'coordination_round_count', 'active_communication_links'):
            self.assertEqual(d[key], 0)

    def test_accounting_and_separation(self):
        m = CoordinationMetrics()
        response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=13, completion_tokens=5, total_tokens=18))
        self.assertIs(m.call('decision', lambda: response), response)
        self.assertEqual(m.to_dict()['coordination_llm_call_count'], 0)
        m.call('coordination', lambda: {'usage': {'input_tokens': 7, 'output_tokens': 3}})
        d = m.to_dict()
        self.assertEqual((d['decision_llm_call_count'], d['coordination_llm_call_count']), (1, 1))
        self.assertEqual((d['input_tokens'], d['output_tokens'], d['total_tokens']), (20, 8, 28))
        for name in ('llm_call_count', 'input_tokens', 'output_tokens', 'total_tokens'):
            self.assertEqual(d[name], d['decision_'+name] + d['coordination_'+name])
        for prefix in ('', 'decision_', 'coordination_'):
            self.assertEqual(d[prefix+'total_tokens'], d[prefix+'input_tokens']+d[prefix+'output_tokens'])

    def test_missing_usage_and_exceptions(self):
        m = CoordinationMetrics()
        m.call('decision', lambda: {'summary': 'offline'})
        failure = RuntimeError('original failure')
        def fail():
            raise failure
        with self.assertRaises(RuntimeError) as caught:
            m.call('coordination', fail)
        self.assertIs(caught.exception, failure)
        self.assertEqual(m.to_dict()['usage_unavailable_call_count'], 2)
        self.assertEqual(m.to_dict()['total_tokens'], 0)

    def test_messages_links_and_explicit_rounds(self):
        m = CoordinationMetrics()
        m.record_message('a', 'b', round_id='episode/round1')
        m.record_message('a', 'b', round_id='episode/round1')
        m.record_message('b', 'a', round_id='episode/round2')
        m.record_message('b', 'c')
        m.record_round('episode/round2')
        d = m.to_dict()
        self.assertEqual(d['coordination_message_count'], 4)
        self.assertEqual(d['active_communication_links'], 3)
        self.assertEqual(d['coordination_round_count'], 2)
        self.assertEqual(d['per_link_message_counts'][0]['message_count'], 2)
        self.assertEqual(d['llm_call_count'], 0)

