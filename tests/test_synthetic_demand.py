import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from policies.ss import run_sS_baseline, run_synthetic_sS_baseline, compute_s_S_from_training_window
from simulation.config import make_synthetic_config
from demand.demand_data import demand_fn_from_sequence, demand_capacity
from project_paths import M5_RESULTS, SYNTHETIC_RESULTS
from demand.synthetic_demand import get_series, load_series


class SyntheticDemandTests(unittest.TestCase):
    def assert_baseline_ledger(self, record, result):
        """Reconstruct the physical ledger without calling env.step or its helpers."""
        count = result['completed_periods']
        config = make_synthetic_config(record)
        self.assertEqual(result['run_status'], 'success', result['error_message'])
        self.assertEqual(count, result['run_metadata']['num_periods'])
        self.assertEqual(len(result['action_trace']), count * 4)
        inventory = list(config['init_inventories'])
        backlog = [0] * 4
        shipments = [[0] * 4 for _ in range(count + 1)]
        orders = [[0] * 4 for _ in range(count + 1)]
        costs = [0] * 4
        stage_entries = [[] for _ in range(4)]
        fulfilled = [[] for _ in range(4)]
        short = [[] for _ in range(4)]
        demand = record['weekly_demand'][:count]
        snapshots = [[] for _ in range(4)]
        for week in range(1, count + 1):
            entries = result['action_trace'][(week - 1) * 4:week * 4]
            opening = [inventory[m] + (shipments[week - config['lead_times'][m]][m]
                       if week > config['lead_times'][m] else 0) for m in range(4)]
            for m, entry in enumerate(entries):
                self.assertEqual((entry['period'], entry['stage']), (week, m))
                obs = entry['observation']
                future = [shipments[dispatch][m] if 1 <= dispatch < week else 0
                          for dispatch in (week + offset - config['lead_times'][m]
                                           for offset in range(1, max(config['lead_times']) + 1))]
                self.assertEqual(obs['inventory'], opening[m])
                self.assertEqual(obs['deliveries'], future)
                self.assertEqual(obs['backlog'], backlog[m])
                self.assertEqual(obs['upstream_backlog'], backlog[m + 1] if m < 3 else 0)
                self.assertEqual(obs['total_cost_so_far'], costs[m])
                self.assertEqual(obs['recent_observations'], snapshots[m][-10:])
                latest = (demand[week - 2] if week > 1 else None) if m == 0 else orders[week][m - 1]
                self.assertEqual(obs['latest_order'], latest)
                history = demand[max(0, week - 27):week - 1] if m == 0 else [
                    orders[w][m - 1] for w in range(max(1, week - 26), week)]
                self.assertEqual(obs['downstream_demand_history'], history)
                self.assertEqual(obs['next_period_demand_forecast'], float(np.mean(history)) if history else None)
                self.assertEqual(obs['forecast_target_period'], week + 1)
                params = result['policy']['stage_parameters'][f'stage_{m}']
                self.assertEqual(params, compute_s_S_from_training_window(record['weekly_demand'], 2))
                position = opening[m] + (backlog[m + 1] if m < 3 else 0) + sum(future) - backlog[m]
                wanted = max(0, params['S'] - position) if position <= params['s'] else 0
                quantity = min(config['prod_capacities'][m], wanted)
                self.assertEqual(entry['executed_order_quantity'], quantity)
                self.assertEqual(entry['proposed_order_quantity'], quantity)
                self.assertEqual(entry['validation_status'], 'valid')
                orders[week][m] = quantity
                snapshots[m].append({
                    'inventory_level': opening[m], 'backlog': backlog[m],
                    'upstream_backlog': backlog[m + 1] if m < 3 else 0,
                    'latest_order': latest, 'in_transit_orders': future,
                    'total_cost_so_far': costs[m],
                })
            received = [demand[week - 1], *orders[week][:3]]
            sales = [min(backlog[m] + received[m], opening[m])
                     for m in range(4)]
            shipments[week] = [*sales[1:], orders[week][3]]
            for m, entry in enumerate(entries):
                pending = backlog[m] + received[m]
                short[m].append(opening[m] < pending and pending > 0)
                inventory[m] = opening[m] - sales[m]
                backlog[m] = pending - sales[m]
                self.assertGreaterEqual(inventory[m], 0)
                self.assertGreaterEqual(backlog[m], 0)
                holding = inventory[m] * config['holding_costs'][m]
                backorder = backlog[m] * config['backlog_costs'][m]
                ordering = config['fixed_order_costs'][m] * (orders[week][m] > 0)
                total = holding + backorder + ordering
                self.assertEqual(entry['holding_cost'], holding)
                self.assertEqual(entry['backlog_cost'], backorder)
                self.assertEqual(entry['fixed_ordering_cost'], ordering)
                self.assertEqual(entry['operational_cost'], total)
                self.assertEqual(entry['reward'], -total)
                costs[m] += total
                stage_entries[m].append(entry)
                fulfilled[m].append(sales[m])
        self.assertEqual(result['episode_reward'], -sum(costs))
        evaluation = result['evaluation_summary']
        self.assertEqual(evaluation['operational']['global_operational_cost'], sum(costs))
        for m in range(4):
            outgoing = [o[m] for o in orders[1:]]
            incoming = demand if m == 0 else [o[m - 1] for o in orders[1:]]
            cv = (float(np.std(outgoing) / np.mean(outgoing)) /
                  float(np.std(incoming) / np.mean(incoming)))
            metrics = evaluation['stage_metrics'][f'stage_{m}']
            self.assertAlmostEqual(metrics['bullwhip_cv'], cv)
            self.assertAlmostEqual(metrics['bullwhip'], float(np.var(outgoing) / np.var(demand)))
            self.assertAlmostEqual(metrics['average_inventory'], np.mean([e['holding_cost'] for e in stage_entries[m]]))
            self.assertAlmostEqual(metrics['average_backlog'], np.mean([e['backlog_cost'] for e in stage_entries[m]]))
            self.assertAlmostEqual(metrics['capacity_saturation_pct'], 100 * np.mean(np.array(outgoing) == config['prod_capacities'][m]))
            self.assertAlmostEqual(metrics['supply_shortage_pct'], 100 * np.mean(short[m]))
            self.assertIsNone(metrics['strict_capacity_binding_pct'])
        self.assertAlmostEqual(evaluation['service_inventory']['cumulative_fulfilled_fraction'], sum(fulfilled[0]) / sum(demand))
        self.assertEqual(result['coordination_metrics']['llm_call_count'], 0)

    def test_four_regime_baselines_match_independent_physical_ledger(self):
        for record in load_series():
            with self.subTest(regime=record['regime']):
                self.assert_baseline_ledger(record, run_synthetic_sS_baseline(record, 130))

    def test_order_capacity_is_full_sequence_quantile_including_zeros(self):
        self.assertEqual(demand_capacity([0] * 20), 1)
        self.assertEqual(demand_capacity(list(range(20))), 19)
        self.assertEqual(demand_capacity([0] * 19 + [10]), 1)
        for quantile in (-0.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                demand_capacity([0, 1], quantile)
        for record in load_series():
            config = make_synthetic_config(record)
            expected = max(1, int(np.ceil(np.quantile(record['weekly_demand'][:130], .95, method='linear'))))
            self.assertEqual(config['prod_capacities'], [expected] * 4)
            self.assertEqual(config['_provenance']['capacity_quantile'], .95)
            self.assertEqual(config['_provenance']['capacity_calibration_weeks'], 130)
            self.assertNotIn('rho', config['_provenance'])

    def test_130_week_prefix_calibrates_environment_and_preserves_training_window(self):
        for record in load_series():
            with self.subTest(regime=record['regime']):
                original = list(record['weekly_demand'])
                config = make_synthetic_config({**record, 'weekly_demand': original[:130]})
                result = run_synthetic_sS_baseline(record, 130)
                self.assertEqual(result['run_status'], 'success')
                self.assertEqual(result['completed_periods'], 130)
                self.assertEqual(len(result['action_trace']), 130 * 4)
                self.assertEqual(record['weekly_demand'], original)
                self.assertEqual(result['policy']['training_demand'], original[:26])
                expected = compute_s_S_from_training_window(original, 2)
                for stage in range(4):
                    self.assertEqual(result['policy']['stage_parameters'][f'stage_{stage}'], expected)
                    observation = result['action_trace'][stage]['observation']
                    self.assertEqual(observation['prod_capacity'], config['prod_capacities'][stage])
                    self.assertEqual(observation['inventory'], config['init_inventories'][stage])
                metadata = result['run_metadata']
                self.assertEqual(metadata['evaluation_window'], {'first_week': 1, 'last_week': 130})
                self.assertEqual(metadata['environment_calibration_weeks'], 130)
                self.assertEqual(metadata['initial_inventory_calibration_weeks'], 130)
                self.assertEqual(metadata['capacity_calibration_weeks'], 130)
                self.assertEqual(metadata['source_num_periods'], 277)
                changed_tail = {**record, 'weekly_demand': original[:130] + [1000000] * 147}
                repeated = run_synthetic_sS_baseline(changed_tail, 130)
                self.assertEqual(repeated['action_trace'], result['action_trace'])
                self.assertEqual(repeated['episode_reward'], result['episode_reward'])
        for horizon in (25, 278, True, 130.5):
            with self.assertRaises(ValueError):
                run_synthetic_sS_baseline(load_series()[0], horizon)


    def test_frozen_series_identity_variable_horizon_and_baseline(self):
        record = dict(id='fixture_001', regime='intermittent', seed=42,
                      weekly_demand=[0, 1, 0, 3]*8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'demand_series.json'
            path.write_text(json.dumps([record]), encoding='utf-8')
            loaded = get_series('fixture_001', path)
            self.assertEqual(loaded['input_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            config = make_synthetic_config(loaded)
            self.assertEqual(config['num_periods'], 32)
            self.assertEqual([config['demand_fn'](i) for i in range(1, 33)], record['weekly_demand'])
            self.assertEqual(config['_provenance']['seed'], 42)
            result = run_sS_baseline(loaded, config=config)
            self.assertEqual(result['run_status'], 'success')
            self.assertEqual(result['completed_periods'], 32)
            self.assertEqual(result['run_metadata']['dataset_source'], 'synthetic')
            self.assertEqual(result['run_metadata']['dataset_id'], 'bernoulli_gamma_v1')
            self.assertEqual(result['policy']['stage_parameters']['stage_0']['sigma'],
                             float(np.std(record['weekly_demand'][:26], ddof=1)))
            self.assertNotEqual(M5_RESULTS, SYNTHETIC_RESULTS)

    def test_invalid_sequences_and_duplicate_ids(self):
        for demand in ([], [1, -1], [1.5], [True]):
            with self.assertRaises(ValueError):
                demand_fn_from_sequence(demand)
        demand_fn = demand_fn_from_sequence([1, 2])
        for period in (0, 3, True):
            with self.assertRaises(IndexError):
                demand_fn(period)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'demand_series.json'
            path.write_text(json.dumps([dict(id='duplicate', weekly_demand=[1, 2])]*2))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_series(path)


if __name__ == '__main__':
    unittest.main()
