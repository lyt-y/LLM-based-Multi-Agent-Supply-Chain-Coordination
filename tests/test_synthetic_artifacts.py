"""Check frozen demand inputs without simulation or provider dependencies."""

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from simulation.config import make_synthetic_config


class SyntheticArtifactTests(unittest.TestCase):
    def setUp(self):
        self.folder = ROOT / 'artifacts/synthetic/bernoulli_gamma_v1'
        self.records = self.read('demand_series.json')

    def read(self, name):
        return json.loads((self.folder / name).read_text(encoding='utf-8'))

    def test_config_reproduces_every_record_and_statistics(self):
        generation = self.read('generation_config.json')
        self.assertEqual(len(self.records), len(generation['regimes']))
        frozen_regimes = [('smooth', 1.0, 25), ('erratic', .95, 1), ('intermittent', .40, 5), ('lumpy', .40, 1)]
        for index, (record, parameters) in enumerate(zip(self.records, generation['regimes'])):
            with self.subTest(series=record['id']):
                regime, probability, shape = frozen_regimes[index]
                self.assertEqual((parameters['regime'], parameters['p'], parameters['k']), (regime, probability, shape))
                self.assertEqual(parameters['theta'], 20 / (probability * shape))
                self.assertEqual(record['seed'], [42, index, 0])
                self.assertEqual(parameters['num_periods'], 277)
                self.assertTrue(all(type(x) is int and x >= 0 for x in record['weekly_demand']))
                self.assertEqual(record['statistics']['observed_regime'], regime)
                self.assertEqual(record['regime'], parameters['regime'])
                self.assertEqual(record['seed'], parameters['seed'])
                for key, value in record['generation_parameters'].items():
                    self.assertEqual(value, parameters[key])
                rng = np.random.default_rng(np.random.SeedSequence(parameters['seed']))
                n = parameters['num_periods']
                occurrences = rng.binomial(1, parameters['p'], size=n)
                sizes = rng.gamma(parameters['k'], parameters['theta'], size=n)
                expected = occurrences * np.ceil(sizes).astype(np.int64)
                np.testing.assert_array_equal(record['weekly_demand'], expected)
                positive = expected[expected > 0]
                stats = record['statistics']
                self.assertAlmostEqual(stats['mean'], float(expected.mean()))
                self.assertEqual(stats['total'], int(expected.sum()))
                self.assertEqual(stats['zero_weeks'], int((expected == 0).sum()))
                self.assertEqual(stats['positive_weeks'], len(positive))
                self.assertAlmostEqual(stats['adi'], n / len(positive))
                cv2 = float(positive.var(ddof=1) / positive.mean() ** 2)
                self.assertAlmostEqual(stats['positive_cv2'], cv2)
                observed = ('smooth' if cv2 <= .49 else 'erratic') if stats['adi'] <= 1.32 else (
                    'intermittent' if cv2 <= .49 else 'lumpy')
                self.assertEqual(stats['observed_regime'], observed)

    def test_manifest_matches_current_files(self):
        manifest = self.read('generation_manifest.json')
        self.assertEqual(manifest['record_count'], len(self.records))
        for name, digest in manifest['files_sha256'].items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((self.folder / name).read_bytes()).hexdigest(), digest)

    def test_saved_calibration_matches_runtime(self):
        calibration = self.read('environment_calibration.json')
        self.assertEqual(calibration['calibration_window'], 'weeks_1_to_130')
        self.assertNotIn('rho', calibration)
        self.assertEqual(len(calibration['records']), len(self.records))
        for record in self.records:
            with self.subTest(series=record['id']):
                saved = next(r for r in calibration['records'] if r['series_id'] == record['id'])
                config = make_synthetic_config(record, q=calibration['q'],
                                               capacity_quantile=calibration['capacity_quantile'])
                self.assertEqual(config['num_periods'], 277)
                self.assertEqual(saved['init_inventories'], config['init_inventories'])
                self.assertEqual(saved['prod_capacities'], config['prod_capacities'])
                self.assertEqual(calibration['lead_times'], config['lead_times'])


if __name__ == '__main__':
    unittest.main()
