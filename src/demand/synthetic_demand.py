"""Load frozen Bernoulli-Gamma inputs without generating data during experiments."""

import json
import hashlib
from pathlib import Path

from demand.demand_data import demand_fn_from_sequence
from project_paths import SYNTHETIC_ARTIFACTS


def load_series(path=SYNTHETIC_ARTIFACTS / 'demand_series.json'):
    source = Path(path)
    payload = source.read_bytes()
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError('Synthetic demand_series.json must contain a list of records')
    seen = set()
    for record in data:
        if not isinstance(record, dict) or not isinstance(record.get('id'), str) or not record['id']:
            raise ValueError('Each synthetic record requires a nonempty id')
        if record['id'] in seen:
            raise ValueError('Duplicate synthetic series id')
        seen.add(record['id'])
        demand_fn_from_sequence(record['weekly_demand'])
        record['dataset_source'] = 'synthetic'
        record['input_file'] = str(source.resolve())
        record['input_sha256'] = hashlib.sha256(payload).hexdigest()
    return data


def get_series(series_id, path=SYNTHETIC_ARTIFACTS / 'demand_series.json'):
    for record in load_series(path):
        if record['id'] == series_id:
            return record
    raise KeyError(f'Synthetic series not found: {series_id}')
