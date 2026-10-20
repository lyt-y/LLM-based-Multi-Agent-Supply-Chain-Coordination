"""Run identities and summary persistence, independent of LLM clients."""

import json
import os
from pathlib import Path
import re
import tempfile

REPRESENTATIVE_ID = 'FOODS_1_002_WI_1_evaluation'
STAGES = ('Retailer', 'Wholesaler', 'Distributor', 'Manufacturer')

def next_run_path(directory, record_id=REPRESENTATIVE_ID):
    """Append after the highest existing run number; never reuse a failed run."""
    directory = Path(directory)
    numbers = []
    for path in directory.glob(f'{record_id}_run_*.json'):
        match = re.fullmatch(re.escape(record_id) + r'_run_([0-9]{3,})(?:\.checkpoint)?\.json', path.name)
        if match is None or int(match.group(1)) < 1:
            raise ValueError('Unexpected existing run filename')
        number = int(match.group(1))
        if match.group(1) != f'{number:03d}':
            raise ValueError('Noncanonical existing run number')
        numbers.append(number)
    return directory / f'{record_id}_run_{max(numbers, default=0) + 1:03d}.json'


def load_summary_rows(path):
    """Read/validate rows; upgrade the original single-pilot identity in memory only."""
    path = Path(path)
    rows = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    if not isinstance(rows, list):
        raise ValueError('Gemini summary must contain a list of run records')
    identities = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Gemini summary rows must be objects')
        record = row.get('record_id', row.get('representative_record_id'))
        if not isinstance(record, str) or not record:
            raise ValueError('Summary row has no record identity')
        if 'run_id' not in row:
            # The original entry point supported only run_001. Preserve every
            # existing value, adding only its previously implicit run identity.
            if not (path.parent / f'{record}_run_001.json').is_file():
                raise ValueError('Cannot identify legacy summary without its run_001 file')
            row['run_id'] = 'run_001'
        if not isinstance(row['run_id'], str) or not re.fullmatch(r'run_[0-9]{3,}', row['run_id']):
            raise ValueError('Invalid summary run_id')
        identity = (record, row['run_id'])
        if identity in identities:
            raise ValueError('Duplicate record_id/run_id in existing summary')
        identities.add(identity)
    return rows


def persist_summary_row(path, row):
    """Atomically upsert only this record/run pair, retaining prior outcomes."""
    path = Path(path)
    if (not isinstance(row.get('record_id'), str) or not row['record_id']
            or not isinstance(row.get('run_id'), str)
            or not re.fullmatch(r'run_[0-9]{3,}', row['run_id'])):
        raise ValueError('Summary row requires record_id and run_id')
    rows = load_summary_rows(path)
    identity = (row['record_id'], row['run_id'])
    for index, existing in enumerate(rows):
        if (existing.get('record_id', existing.get('representative_record_id')), existing['run_id']) == identity:
            rows[index] = row
            break
    else:
        rows.append(row)
    payload = json.dumps(rows, indent=2, allow_nan=False) + '\n'
    descriptor, temporary = tempfile.mkstemp(prefix='.summary.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def build_result_summary(result):
    """One baseline-compatible comparison row, with observed Gemini resources."""
    metadata = result['run_metadata']
    provider = metadata['provider']
    evaluation = result['evaluation_summary']
    metrics = result['coordination_metrics']
    record_id = metadata.get('representative_record_id', metadata.get('series_id'))
    thinking = [report['thoughts_token_count'] for report in provider['usage_reports']]
    reported_thinking = [value for value in thinking if value is not None]
    return {
        **{key: provider[key] for key in ('architecture', 'provider', 'model', 'prompt_version',
                                         'adapter_version', 'schema_version')},
        'run_id': metadata.get('run_id'),
        'record_id': record_id,
        **{key: metadata[key] for key in (
            'dataset_source', 'dataset_id', 'series_id', 'input_file', 'input_sha256',
            'source_num_periods', 'initial_inventories', 'initial_inventory_quantile',
            'initial_inventory_method', 'initial_inventory_lead_time',
            'initial_inventory_calibration_weeks', 'capacities', 'capacity_quantile',
            'capacity_method', 'capacity_calibration_weeks', 'capacity_scope',
            'fulfillment_policy') if key in metadata},
        **({'representative_record_id': record_id} if 'representative_record_id' in metadata
           else {'series_id': record_id}),
        'regime': metadata['regime'],
        'num_periods': metadata['num_periods'],
        'demand_window': metadata.get('demand_window'),
        **({'extension': metadata['extension']} if 'extension' in metadata else {}),
        'calibration_window': metadata.get('calibration_window'),
        'run_status': result['run_status'],
        'completed_periods': result['completed_periods'],
        'error_type': result['error_type'], 'error_message': result['error_message'],
        'terminal_failure_category': (result.get('provider_invocation_trace', [{}])[-1].get('failure_category')
                                      if result['run_status'] == 'failed' and result.get('provider_invocation_trace')
                                      else None),
        **evaluation['operational'],
        'episode_reward': result['episode_reward'],
        **evaluation['service_inventory'],
        **{name.lower() + '_bullwhip': evaluation['bullwhip'][f'stage_{stage}']
           for stage, name in enumerate(STAGES)},
        **{name.lower() + '_bullwhip_cv': evaluation['bullwhip_cv'][f'stage_{stage}']
           for stage, name in enumerate(STAGES)},
        'global_bullwhip': evaluation['global_bullwhip'],
        'stage_metrics': evaluation['stage_metrics'],
        **metrics,
        'api_invocation_count': metrics['decision_llm_call_count'] + metrics['coordination_llm_call_count'],
        # Separate API-reported thinking usage is already included in output_tokens.
        # A partial sum is explicitly accompanied by its missing-report count.
        'thinking_tokens': sum(reported_thinking) if reported_thinking or not thinking else None,
        'thinking_usage_unavailable_call_count': len(thinking) - len(reported_thinking),
        **{key: provider[key] for key in ('failed_invocation_count', 'api_failure_count',
                                         'malformed_output_count', 'invalid_action_count')},
    }



def build_baseline_summary(result):
    """Project baseline KPIs into the run-indexed summary schema."""
    metadata = result['run_metadata']
    evaluation = result['evaluation_summary']
    policy = result['policy']
    params = policy['stage_parameters']['stage_0']
    record_id = metadata.get('series_id', metadata.get('representative_record_id'))
    return {
        **metadata, 'record_id': record_id, 'run_id': metadata.get('run_id'),
        'run_status': result['run_status'], 'completed_periods': result['completed_periods'],
        'calibration_weeks': policy['training_weeks'],
        'standard_deviation_ddof': policy['standard_deviation_ddof'],
        **{key: params[key] for key in ('mu', 'sigma', 'z', 's', 'S', 'eoq_increment',
                                       'zero_demand_fallback_applied')},
        **evaluation['operational'], 'episode_reward': result['episode_reward'],
        **evaluation['service_inventory'],
        **{name.lower() + '_bullwhip': evaluation['bullwhip'][f'stage_{stage}']
           for stage, name in enumerate(STAGES)},
        **{name.lower() + '_bullwhip_cv': evaluation['bullwhip_cv'][f'stage_{stage}']
           for stage, name in enumerate(STAGES)},
        'global_bullwhip': evaluation['global_bullwhip'],
        'stage_metrics': evaluation['stage_metrics'],
        **result['coordination_metrics'],
    }
