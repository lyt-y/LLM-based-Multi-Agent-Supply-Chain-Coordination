"""Read frozen publication records without recalculating scientific metrics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = {'rule_based_sS_baseline': 'ss', 'gemini_no_communication': 'no_communication',
           'gemini_local_communication': 'adjacent',
           'gemini_no_communication_without_forecast_input': 'no_forecast'}


def load_results():
    summary = json.loads((ROOT / 'results/summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'results/manifest.json').read_text(encoding='utf-8'))
    rows = summary['reported_runs'] + summary['historical_references']
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(manifest['experiments']):
        raise ValueError('Publication run identities are missing or duplicated')
    for row in summary['reported_runs']:
        evidence = manifest['experiments'][row['id']]
        horizon = evidence['configuration']['num_periods']
        if (horizon != 130 or row['comparison_periods'] != 130
                or evidence['condition_role'] != 'updated_evaluated'
                or row['run_status'] != 'success' or row['completed_periods'] != horizon):
            raise ValueError('Incomplete experiment in reported comparisons')
    return summary, manifest


def select_run(dataset, method, regime, record_id, periods, *, historical=False,
               input_sha256=None):
    summary, manifest = load_results()
    rows = summary['reported_runs'] + (summary['historical_references'] if historical else [])
    matches = []
    for row in rows:
        if (row['dataset'], row['method'], row['regime'], row['record_id']) != (
                dataset, method, regime, record_id):
            continue
        metadata = manifest['experiments'][row['id']]['configuration']
        horizon = row['comparison_periods']
        if horizon != periods:
            continue
        if input_sha256 is not None and row['summary'].get('input_sha256') != input_sha256:
            continue
        matches.append(row)
    if len(matches) != 1:
        raise ValueError(f'Expected one publication row: {dataset}/{method}/{record_id}/{periods}; found {len(matches)}')
    return matches[0], manifest['experiments'][matches[0]['id']]
