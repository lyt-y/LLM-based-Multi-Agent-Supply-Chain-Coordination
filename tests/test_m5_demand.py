import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from demand.m5_demand import (DATA_PATH, ARTIFACT_PATH, REGIMES, aggregate_weekly,
                       characterize, classify_regime, robust_scale, alternating_kmedoids,
                       read_m5_records, load_representatives, get_representative, m5_demand_fn)


class M5DemandTests(unittest.TestCase):
    def test_forecast_completed_history_alignment_and_no_leakage(self):
        from demand.m5_demand import build_demand_forecasts
        demand = list(range(100))
        forecasts = build_demand_forecasts(demand)
        self.assertEqual(len(forecasts), len(demand))
        self.assertIsNone(forecasts[0])
        for t in (1, 5, 24, 25, 26, 27, 60, 99):
            self.assertEqual(forecasts[t], np.mean(demand[max(0, t-26):t]))
            changed = demand[:t] + [999999] * (len(demand)-t)
            self.assertEqual(forecasts[t], build_demand_forecasts(changed)[t])
        self.assertEqual(forecasts[1], demand[0])
        self.assertEqual(forecasts[26], np.mean(demand[:26]))
        self.assertEqual(forecasts[27], np.mean(demand[1:27]))
        changed = demand.copy()
        changed[0] = 999999
        self.assertEqual(forecasts[27], build_demand_forecasts(changed)[27])

    def test_forecast_artifact_and_manifest(self):
        from demand.m5_demand import persist_demand_forecasts, build_demand_forecasts, FORECAST_METADATA
        with tempfile.TemporaryDirectory() as directory:
            artifact, manifest = persist_demand_forecasts(directory)
            self.assertEqual(json.loads((Path(directory)/'demand_forecasts_26.json').read_text()), artifact)
            self.assertEqual(json.loads((Path(directory)/'forecast_manifest.json').read_text()), manifest)
            self.assertEqual(artifact['metadata'], FORECAST_METADATA)
            self.assertEqual(manifest['window'], 26)
            self.assertEqual(manifest['forecast_horizon'], 1)
            self.assertEqual(manifest['target'], 'next_period_demand')
            for group in load_representatives().values():
                for record in group:
                    stored = artifact['forecasts'][record['id']]
                    self.assertEqual(len(stored), 277)
                    self.assertEqual(stored, build_demand_forecasts(record['weekly_demand']))
            self.assertEqual(json.loads((ARTIFACT_PATH/'demand_forecasts_26.json').read_text()), artifact)
            self.assertEqual(json.loads((ARTIFACT_PATH/'forecast_manifest.json').read_text()), manifest)

    def test_aggregation_and_discarded_tail(self):
        row = {f'd_{i}': str(i) for i in range(1, 1942)}
        weekly = aggregate_weekly(row)
        self.assertEqual(weekly, [sum(range(s, s + 7)) for s in range(1, 1940, 7)])
        row['d_1940'], row['d_1941'] = '999999', '888888'
        self.assertEqual(aggregate_weekly(row), weekly)

    def test_statistics_include_full_sequence_and_zeros(self):
        x = np.array(([0, 1, 3, 0, 2] * 56)[:277])
        stats = characterize(x)
        positive = x[x > 0]
        self.assertTrue(stats['eligible'])
        self.assertEqual(stats['adi'], 277 / len(positive))
        self.assertEqual(stats['mean_weekly_demand'], x.mean())
        self.assertAlmostEqual(stats['positive_cv2'], positive.var(ddof=1) / positive.mean() ** 2)
        self.assertAlmostEqual(stats['lag1_autocorrelation'], np.corrcoef(x[:-1], x[1:])[0, 1])
        changed = x.copy()
        changed[-1] += 100
        self.assertNotEqual(characterize(changed)['mean_weekly_demand'], stats['mean_weekly_demand'])

    def test_frozen_boundaries(self):
        for adi, cv2, expected in [(1.32, .49, 'smooth'), (1.32, .490001, 'erratic'),
                                    (1.320001, .49, 'intermittent'), (1.320001, .490001, 'lumpy')]:
            self.assertEqual(classify_regime(adi, cv2), expected)

    def test_undefined_statistics(self):
        for x, reason in [([0]*277, 'fewer_than_two_positive_weeks'),
                           ([0]*276+[1], 'fewer_than_two_positive_weeks'),
                           ([2]*277, 'undefined_lag1_autocorrelation'),
                           ([2]*276+[3], 'undefined_lag1_autocorrelation')]:
            self.assertEqual(characterize(x)['reason'], reason)

    def test_robust_scaling(self):
        x = np.array([[1., 4.], [2., 4.], [3., 4.], [100., 4.]])
        z, median, scale = robust_scale(x)
        np.testing.assert_array_equal(median, [2.5, 4.])
        np.testing.assert_array_equal(scale, [25.5, 1.])
        np.testing.assert_allclose(z, (x - median) / scale)

    def test_kmedoids_actual_members_and_fixed_point(self):
        z = np.random.default_rng(10).normal(size=(45, 4))
        medoids, info = alternating_kmedoids(z, k=4)
        repeated, repeated_info = alternating_kmedoids(z, k=4, block_size=7)
        np.testing.assert_array_equal(medoids, repeated)
        self.assertEqual(info, repeated_info)
        self.assertEqual(len(set(medoids)), 4)
        self.assertTrue(info['converged'])
        self.assertTrue(np.all(np.diff(info['objective_history']) <= 1e-10))
        distances = cdist(z, z[medoids])
        labels = distances.argmin(axis=1)
        labels[medoids] = np.arange(4)
        for cluster, medoid in enumerate(medoids):
            members = np.flatnonzero(labels == cluster)
            costs = cdist(z[members], z[members]).sum(axis=1)
            self.assertAlmostEqual(costs.min(), costs[np.searchsorted(members, medoid)])
        with self.assertRaises(ValueError):
            alternating_kmedoids(z, k=46)
        with self.assertRaises(RuntimeError):
            alternating_kmedoids(z, k=4, max_iter=0)

    def test_duplicate_features_keep_distinct_medoids(self):
        medoids, info = alternating_kmedoids(np.zeros((40, 4)), k=30)
        np.testing.assert_array_equal(medoids, np.arange(30))
        self.assertEqual(info['iterations'], 1)

    def test_one_based_adapter_and_snapshot(self):
        x = list(range(277))
        demand = m5_demand_fn(x)
        x[0] = 999
        self.assertEqual([demand(t) for t in range(1, 278)], list(range(277)))
        for t in [0, -1, 278, 1.5, True]:
            with self.assertRaises(IndexError):
                demand(t)
        for bad in [[0]*276, [0.5]*277, [-1]*277]:
            with self.assertRaises(ValueError):
                m5_demand_fn(bad)

    def test_exported_records_match_all_source_weeks(self):
        reps = load_representatives()
        manifest = json.loads((ARTIFACT_PATH / 'selection_manifest.json').read_text())
        selected = {r['id']: r for group in reps.values() for r in group}
        self.assertEqual(len(selected), 120)
        for regime in REGIMES:
            self.assertEqual(len(reps[regime]), 30)
            self.assertEqual([r['id'] for r in reps[regime]], manifest['selected_record_ids'][regime])
        found = set()
        import os
        data_path = Path(os.environ.get("M5_DATA_PATH", str(DATA_PATH)))
        if not data_path.is_file():
            self.skipTest("Set M5_DATA_PATH to the external sales_train_evaluation.csv")
        for row in read_m5_records(data_path):
            if row['id'] not in selected:
                continue
            record = selected[row['id']]
            weekly = record['weekly_demand']
            expected = [sum(int(row[f'd_{i}']) for i in range(start, start+7)) for start in range(1, 1940, 7)]
            self.assertEqual(weekly, expected)
            self.assertEqual(len(weekly), 277)
            self.assertTrue(all(type(x) is int for x in weekly))
            stats = characterize(weekly)
            self.assertTrue(stats.pop('eligible'))
            for key, value in stats.items():
                if isinstance(value, str):
                    self.assertEqual(record[key], value)
                else:
                    self.assertAlmostEqual(record[key], value, places=12)
            found.add(row['id'])
        self.assertEqual(found, set(selected))

    def test_config_and_four_full_simulator_runs(self):
        from simulation.config import env_configs, make_m5_config
        from simulation.env import env_creator
        from demand.demand_data import initial_inventory, demand_capacity
        self.assertEqual(list(env_configs), ['m5'])
        for regime in REGIMES:
            record = get_representative(regime=regime)
            config = make_m5_config(record['id'])
            self.assertEqual(config['num_periods'], 277)
            self.assertEqual(config['num_stages'], 4)
            weekly = record['weekly_demand']
            self.assertEqual(config['_provenance']['q'], 0.95)
            self.assertEqual(config['_provenance']['capacity_quantile'], 0.95)
            for key, expected in [
                    ('init_inventories', [initial_inventory(weekly[:130], q=0.95)]*4),
                    ('prod_capacities', [demand_capacity(weekly[:130], capacity_quantile=0.95)]*4),
                    ('lead_times', [2]*4), ('fixed_order_costs', [1]*4),
                    ('backlog_costs', [1]*4), ('holding_costs', [1]*4)]:
                self.assertEqual(config[key], expected)
            self.assertNotIn('sale_prices', config)
            self.assertNotIn('order_costs', config)
            self.assertEqual([config['demand_fn'](t) for t in range(1, 278)], weekly)
            env = env_creator(config)
            env.reset(seed=0)
            for t in range(1, 278):
                observations, rewards, terminated, truncated, _ = env.step({
                    f'stage_{m}': min(4, config['prod_capacities'][m]) for m in range(4)})
                self.assertEqual(env.demands[t], record['weekly_demand'][t-1])
                self.assertEqual(terminated['__all__'], t == 277)
                self.assertFalse(truncated['__all__'])
                self.assertTrue(all(np.isfinite(v) for v in rewards.values()))
            np.testing.assert_array_equal(env.demands[1:], record['weekly_demand'])
            print(f'Smoke {regime}: {record["id"]}, {env.period} periods, total demand {int(env.demands.sum())}')



if __name__ == '__main__':
    unittest.main(verbosity=2)
