"""Frozen publication identities, configurations and incomplete-run separation."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts._results import load_results, select_run
from simulation.config import make_m5_config, make_synthetic_config
from demand.m5_demand import get_representative
from demand.synthetic_demand import load_series
from policies.ss import compute_s_S_from_training_window
from agents import gemini_provider, gemini_local_communication, gemini_no_forecast, gemini_controlled_adjacent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


class PublicationTests(unittest.TestCase):
    def test_frozen_comparisons_and_all_migrated_values(self):
        summary, manifest = load_results()
        expected = {'m5_130_current': '8b1ef2001fdc7c141a2e9f45228d7f1fad61fc427cdfbf5fa40a961a206fe5a2', 'synthetic_130': 'bf33b80aaa7a3ae55df3ede14c5b0cc403aaffb0c26c31a6be12a0a3771acfd1'}
        self.assertEqual(len(summary['reported_runs']), 32)
        self.assertEqual(Counter(row['comparison'] for row in summary['reported_runs']),
                         Counter({family: 16 for family in expected}))
        self.assertEqual(Counter((r['dataset'], r['method'], r['regime']) for r in summary['reported_runs']),
                         Counter({(d, p, r): 1 for d in ('m5', 'synthetic')
                                  for p in ('ss', 'no_communication', 'adjacent', 'no_forecast')
                                  for r in ('smooth', 'erratic', 'intermittent', 'lumpy')}))
        with self.assertRaises(ValueError):
            select_run('m5', 'ss', 'intermittent', 'FOODS_2_368_TX_2_evaluation', 277)
        for family, frozen in expected.items():
            rows = sorted([row for row in summary['reported_runs'] if row['comparison'] == family],
                          key=lambda row: row['id'])
            self.assertEqual(digest([(row['id'], row['summary']) for row in rows]), frozen)
        for row in summary['reported_runs'] + summary['historical_references']:
            evidence = manifest['experiments'][row['id']]
            self.assertEqual(digest(row['summary']), evidence['summary_sha256'])
            self.assertEqual(digest(evidence['configuration']), evidence['configuration_sha256'])
            if 'policy' in evidence:
                self.assertEqual(digest(evidence['policy']), evidence['policy_sha256'])
            source = evidence['source_evidence']
            identity = hashlib.sha256((source['original_path'] + '\0' + str(source['sha256'])).encode()).hexdigest()[:20]
            self.assertEqual(row['id'], 'run-' + identity)
            if 'evaluation_summary' in row:
                self.assertEqual(digest(row['evaluation_summary']), evidence['evaluation_sha256'])

    def test_paused_unknown_and_unevaluated_conditions_are_not_final(self):
        summary, manifest = load_results()
        self.assertEqual(summary['historical_references'], [])
        for row in summary['reported_runs']:
            self.assertEqual(row['run_status'], 'success')
            self.assertEqual(row['completed_periods'], row['comparison_periods'])
            self.assertNotEqual(row['record_id'], 'FOODS_1_188_WI_1_evaluation')
            self.assertNotEqual(row['method'], 'controlled_adjacent')
        paused = [source['record'] for source in manifest['execution_records'] if source['record']['status'] == 'paused']
        self.assertEqual(len(paused), 1)
        self.assertEqual(paused[0]['jobs'][0]['completed_periods'], 19)
        self.assertEqual(paused[0]['jobs'][0]['status'], 'paused')
        self.assertEqual([job['status'] for job in paused[0]['jobs'][1:]], ['queued', 'queued'])
        self.assertIn('controlled_adjacent', manifest['implemented_without_evaluated_results'])

    def test_updated_settings_and_frozen_inputs_match_runtime(self):
        summary, manifest = load_results()
        settings = manifest['current_settings']
        self.assertEqual(settings['environment_calibration_weeks'], 130)
        self.assertEqual(settings['initial_inventory']['q'], .95)
        self.assertEqual(settings['capacity']['quantile'], .95)
        self.assertNotIn('rho', settings)
        self.assertEqual(settings['providers']['no_communication'], gemini_provider.PROVIDER_CONFIG)
        self.assertEqual(settings['providers']['adjacent_prompt_version'], gemini_local_communication.PROMPT_VERSION)
        self.assertEqual(settings['providers']['no_forecast_prompt_version'], gemini_no_forecast.PROMPT_VERSION)
        self.assertEqual(settings['providers']['controlled_adjacent_prompt_version'], gemini_controlled_adjacent.PROMPT_VERSION)
        for rel, expected in {**manifest['scientific_source_sha256'], **manifest['input_sha256'],
                              **manifest.get('public_metadata_sha256', {})}.items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), expected, rel)
        for row in summary['reported_runs']:
            evidence = manifest['experiments'][row['id']]
            self.assertEqual(evidence['condition_role'], 'updated_evaluated')
            self.assertEqual(evidence['run_status'], 'success')
            self.assertEqual(metadata_horizon := evidence['configuration']['num_periods'], 130)
            self.assertEqual(row['comparison_periods'], metadata_horizon)
            record = get_representative(row['record_id']) if row['dataset'] == 'm5' else next(
                record for record in load_series() if record['id'] == row['record_id'])
            config = (make_m5_config if row['dataset'] == 'm5' else make_synthetic_config)(record)
            metadata = evidence['configuration']
            self.assertEqual(metadata['initial_inventories'], config['init_inventories'])
            self.assertEqual(metadata['capacities'], config['prod_capacities'])
            self.assertEqual(metadata['input_sha256'], record['input_sha256'])

            for key in ('lead_times', 'holding_costs', 'backlog_costs', 'fixed_order_costs', 'num_stages'):
                self.assertEqual(metadata[key], config[key])
            for key, value in config['_provenance'].items():
                if key == 'input_file':
                    self.assertEqual((ROOT / metadata[key]).resolve(), Path(value).resolve())
                else:
                    self.assertEqual(metadata[key], value, (row['id'], key))
            self.assertEqual(metadata['evaluation_window'], {'first_week': 1, 'last_week': 130})
            if row['method'] == 'ss':
                policy = evidence['policy']
                self.assertEqual(policy['training_weeks'], 26)
                self.assertEqual(policy['standard_deviation_ddof'], 1)
                self.assertEqual(policy['training_demand'], record['weekly_demand'][:26])
                for parameters in policy['stage_parameters'].values():
                    expected = compute_s_S_from_training_window(record['weekly_demand'][:26], 2)
                    self.assertEqual(parameters, expected)
            else:
                provider = metadata['provider']
                self.assertEqual(provider['model'], 'gemini-3.8-flash')
                expected_prompt = {'no_communication': 'gemini_no_communication_v7',
                                   'adjacent': gemini_local_communication.PROMPT_VERSION,
                                   'no_forecast': gemini_no_forecast.PROMPT_VERSION}[row['method']]
                self.assertEqual(provider['prompt_version'], expected_prompt)

    def test_published_figure_integrity(self):
        _, manifest = load_results()
        self.assertEqual(set(manifest['figures']), {p.name for p in (ROOT / 'results/figures').glob('*.pdf')})
        for figure in manifest['figures'].values():
            self.assertEqual(hashlib.sha256((ROOT / figure['local_path']).read_bytes()).hexdigest(),
                             figure['regenerated_sha256'])
