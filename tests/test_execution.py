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


class ExecutionTests(unittest.TestCase):
    def test_package_imports_and_paths(self):
        for name in ('simulation.env', 'simulation.config', 'policies.ss', 
                     'demand.forecast', 'demand.m5_demand', 'demand.synthetic_demand',
                     'evaluation.metrics', 'evaluation.summary', 'evaluation.experiment_results',
                     'agents.gemini_provider', 'agents.gemini_local_communication',
                     'agents.gemini_no_forecast', 'scripts.plot_four_demand_series',
                     'scripts.plot_four_regime_experiment_comparison'):
            importlib.import_module(name)
        self.assertEqual(PROJECT_ROOT, ROOT)

    def test_synthetic_config_does_not_read_m5(self):
        record = load_series()[0]
        with patch('simulation.config.get_representative', side_effect=AssertionError('M5 read')):
            prefix, config = runner.prepare_config(record, 'synthetic', 130)
        self.assertEqual(len(prefix['weekly_demand']), 130)
        self.assertEqual(len(record['weekly_demand']), 277)
        self.assertEqual(config['_provenance']['environment_calibration_weeks'], 130)

    def test_dataset_specific_initial_calibration_uses_own_first_130_weeks(self):
        records = [('m5', get_representative()), ('synthetic', load_series()[0])]
        for dataset, record in records:
            with self.subTest(dataset=dataset):
                for horizon in (104, 130, 277):
                    calibration_weeks = 130
                    weekly = np.array(record['weekly_demand'][:calibration_weeks])
                    expected_capacity = max(1, int(np.ceil(np.quantile(weekly, .95, method='linear'))))
                    lead_demand = np.convolve(weekly, np.ones(2, dtype=int), mode='valid')
                    expected_inventory = max(1, int(np.ceil(np.quantile(lead_demand, .95, method='linear'))))
                    prefix, config = runner.prepare_config(record, dataset, horizon)
                    self.assertEqual(config['prod_capacities'], [expected_capacity] * 4)
                    self.assertEqual(config['init_inventories'], [expected_inventory] * 4)
                    self.assertEqual([config['demand_fn'](t) for t in range(1, horizon + 1)],
                                     record['weekly_demand'][:horizon])
                    self.assertEqual(len(prefix['weekly_demand']), horizon)
                    metadata = config['_provenance']
                    self.assertEqual(metadata['dataset_source'], dataset)
                    self.assertEqual(metadata['dataset_id'], 'm5' if dataset == 'm5' else 'bernoulli_gamma_v1')
                    self.assertEqual(metadata['series_id'], record['id'])
                    self.assertEqual(metadata['initial_inventory_calibration_weeks'], calibration_weeks)
                    self.assertEqual(metadata['capacity_calibration_weeks'], calibration_weeks)
                    self.assertEqual(metadata['initial_inventory_lead_time'], 2)
                    self.assertEqual(metadata['input_file'], record['input_file'])
                    self.assertEqual(metadata['input_sha256'], record['input_sha256'])
                changed = {**record, 'weekly_demand': record['weekly_demand'][:130] + [1000000] * 147}
                factory = make_m5_config if dataset == 'm5' else make_synthetic_config
                changed_config = factory(changed)
                self.assertEqual(changed_config['prod_capacities'], [expected_capacity] * 4)
                self.assertEqual(changed_config['init_inventories'], [expected_inventory] * 4)

    def test_cross_dataset_records_and_unknown_datasets_are_rejected(self):
        m5 = get_representative()
        synthetic = load_series()[0]
        for factory, record in ((make_m5_config, synthetic), (make_synthetic_config, m5)):
            with self.subTest(factory=factory.__name__), self.assertRaises(ValueError):
                factory(record)
        for dataset, record in (('m5', synthetic), ('synthetic', m5), ('unknown', synthetic)):
            with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                runner.prepare_config(record, dataset, 130)
        with self.assertRaises(ValueError):
            runner.select_records('unknown')

    def test_preflight_creates_no_results(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(['--dataset', 'synthetic', '--all', '--preflight',
                                          '--output-root', directory]), 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_baseline_repeated_runs_preserve_legacy_summary(self):
        record = load_series()[0]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            result_dir = Path(directory) / runner.METHODS['ss']
            result_dir.mkdir()
            legacy = result_dir / 'summary.json'
            legacy.write_bytes(b'[{"historical": true}]')
            kwargs = dict(dataset='synthetic', method='ss', periods=130, output_root=directory)
            self.assertEqual(runner.run_record(record, **kwargs), 0)
            first = result_dir / f"{record['id']}_run_001.json"
            original = first.read_bytes()
            self.assertEqual(runner.run_record(record, **kwargs), 0)
            self.assertEqual(first.read_bytes(), original)
            self.assertEqual(legacy.read_bytes(), b'[{"historical": true}]')
            rows = json.loads((result_dir / 'summary_runs.json').read_text())[0:2]
            self.assertEqual([row['run_id'] for row in rows], ['run_001', 'run_002'])
            self.assertTrue(all(row['completed_periods'] == 130 for row in rows))
            self.assertIn('manufacturer_bullwhip_cv', rows[0])

    def test_llm_synthetic_preflight_and_summary(self):
        record = load_series()[0]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.dict('os.environ', {'GEMINI_API_KEY': 'offline-test'}), \
                patch('agents.gemini_provider.load_dotenv'), \
                patch('agents.gemini_provider.genai.Client') as client:
            client.return_value.models.generate_content.return_value = SimpleNamespace(
                text='[[[0]]]', usage_metadata=None)
            kwargs = dict(dataset='synthetic', method='no_communication',
                          periods=2, output_root=directory)
            self.assertEqual(runner.run_record(record, preflight=True, **kwargs), 0)
            client.return_value.models.generate_content.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(runner.run_record(record, **kwargs), 0)
            row = json.loads((Path(directory) / runner.METHODS['no_communication']
                              / 'summary.json').read_text())[0]
            self.assertEqual(row['record_id'], record['id'])
            self.assertEqual(row['dataset_source'], 'synthetic')
            self.assertEqual(row['dataset_id'], 'bernoulli_gamma_v1')
            self.assertEqual(row['input_sha256'], record['input_sha256'])
            expected = make_synthetic_config(record)
            self.assertEqual(row['initial_inventories'], expected['init_inventories'])
            self.assertEqual(row['capacities'], expected['prod_capacities'])
            self.assertEqual(row['initial_inventory_calibration_weeks'], 130)
            self.assertEqual(row['capacity_calibration_weeks'], 130)
            self.assertEqual(row['completed_periods'], 2)
            self.assertEqual(client.return_value.models.generate_content.call_count, 8)




