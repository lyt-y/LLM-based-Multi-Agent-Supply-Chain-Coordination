"""Gemini no-communication experiment without a demand forecast in agent input."""

from project_paths import M5_RESULTS

import argparse
import json
from pathlib import Path
import tempfile

from evaluation.metrics import CoordinationMetrics
from simulation.config import make_m5_config
from simulation.experiment_interface import json_safe, run_episode
from evaluation.experiment_results import add_service_inventory_summary, persist_run
from agents.gemini_provider import (
    GeminiDecisionProvider, NO_COMMUNICATION_PROMPT, PROJECT_ROOT,
    REPRESENTATIVE_ID, STAGES, build_result_summary, load_summary_rows,
    next_run_path, persist_summary_row, latest_order_definition,
)
from demand.m5_demand import get_representative


EXPERIMENT_ID = 'gemini_no_communication_without_forecast_input'
PROMPT_VERSION = 'gemini_no_communication_without_forecast_input_v6'
ADAPTER_VERSION = 'gemini_no_communication_without_forecast_input_adapter_v1'

_prefix, _rest = NO_COMMUNICATION_PROMPT.split('P5 Demand forecast\n', 1)
_forecast, _suffix = _rest.split('P6 Information and communication rule: local agent\n', 1)
NO_FORECAST_PROMPT = (
    _prefix + 'P5 Information and communication rule: local agent\n'
    + _suffix.replace('Use only the information provided in P3-P5.',
                      'Use only the information provided in P3-P4.\n\n'
                      'No external demand forecast is supplied.')
             .replace('P7 Decision', 'P6 Decision')
)


def build_prompt(stage, observation, context):
    """Render the unchanged local inputs and rules, omitting the forecast section."""
    if type(stage) is not int or not 0 <= stage < 4:
        raise ValueError('Expected a stage index in 0..3')
    local = {
        'inventory_level': observation['inventory'],
        'backlog': observation['backlog'],
        'upstream_backlog': observation['upstream_backlog'],
        'latest_order': observation['latest_order'],
        'in_transit_orders': observation['deliveries'],
        'total_cost_so_far': observation['total_cost_so_far'],
    }
    recent = [{key: entry[key] for key in local}
              for entry in observation['recent_observations'][-10:]]
    return NO_FORECAST_PROMPT.format(
        stage_name=STAGES[stage],
        local_json=json.dumps(json_safe(local), indent=2),
        latest_order_definition=latest_order_definition(stage),
        recent_observations_json=json.dumps(json_safe(recent), indent=2),
        order_capacity=int(observation['prod_capacity']),
    )


class GeminiNoForecastProvider(GeminiDecisionProvider):
    uses_forecast_input = False

    def __init__(self, metrics):
        super().__init__(metrics)
        self._metadata.update(
            architecture='no_communication_without_forecast_input',
            prompt_version=PROMPT_VERSION,
            adapter_version=ADAPTER_VERSION,
            demand_forecast={'forecast_method': 'none', 'available_to_agent': False},
        )

    def _build_prompt(self, stage, observation, context):
        return build_prompt(stage, observation, context)

    def _diagnostic_observation(self, observation):
        return json_safe({key: observation[key] for key in (
            'inventory', 'backlog', 'upstream_backlog', 'latest_order', 'deliveries',
            'total_cost_so_far', 'prod_capacity')})


def preflight(*, record_id=REPRESENTATIVE_ID, num_periods=130, output_root=None):
    """Validate the record and choose a fresh result path without an API request."""
    provider = GeminiNoForecastProvider(CoordinationMetrics())
    try:
        record = get_representative(record_id)
        config = make_m5_config(record)
        if record['id'] != record_id or config['num_stages'] != 4 or config['num_periods'] != 277:
            raise ValueError('Expected the selected four-stage, 277-period record')
        if config['lead_times'] != [2] * 4:
            raise ValueError('Expected two-period lead times')
        if type(num_periods) is not int or not 1 <= num_periods <= 277:
            raise ValueError('num_periods must be an integer from 1 to 277')
        from demand.demand_data import demand_fn_from_sequence
        config['num_periods'] = num_periods
        config['demand_fn'] = demand_fn_from_sequence(record['weekly_demand'][:num_periods])
        config['_provenance']['demand_window'] = f'weeks_1_to_{num_periods}'
        config['_provenance'].update(evaluation_window={'first_week': 1, 'last_week': num_periods})
        root = Path(output_root) if output_root is not None else M5_RESULTS
        destination = next_run_path(root.resolve() / EXPERIMENT_ID, record_id)
        rows = load_summary_rows(destination.parent / 'summary.json')
        run_id = destination.stem[len(record_id) + 1:]
        reserved = {row['run_id'] for row in rows if row.get('record_id') == record_id}
        while run_id in reserved or destination.exists():
            run_id = f'run_{int(run_id.removeprefix("run_")) + 1:03d}'
            destination = destination.with_name(f'{record_id}_{run_id}.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=destination.parent):
            pass
        return provider, config, destination
    except BaseException:
        provider.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--record-id', default=REPRESENTATIVE_ID)
    parser.add_argument('--periods', type=int, default=130)
    parser.add_argument('--resume', type=Path, help='Resume a saved checkpoint via the shared runner')
    args = parser.parse_args(argv)
    if args.resume is not None:
        from scripts.run_experiments import main as resume_main
        return resume_main(['--resume', str(args.resume)])
    provider, config, destination = preflight(record_id=args.record_id, num_periods=args.periods)
    run_id = destination.stem[len(args.record_id) + 1:]
    try:
        if args.preflight:
            print(f'Preflight passed (no API request).\nnum_periods: {config["num_periods"]}'
                  f'\nnext_run_id: {run_id}\noutput_path: {destination}')
            return 0
        from simulation.env import env_creator
        env = env_creator(config)
        result = run_episode(env, provider, 'm5', config,
                             metrics=provider.metrics, provider_metadata=provider.metadata,
                             checkpoint_path=destination.with_suffix('.checkpoint.json'),
                             checkpoint_metadata=dict(dataset='m5', method='no_forecast',
                                                      record_id=args.record_id, periods=args.periods))
        result['run_metadata']['provider'] = provider.metadata
        result['run_metadata']['run_id'] = run_id
        add_service_inventory_summary(result, env, config)
        path = persist_run(result, EXPERIMENT_ID, destination.stem,
                           output_root=destination.parent.parent)
        persist_summary_row(path.parent / 'summary.json', build_result_summary(result))
        print(f'Run {result["run_status"]}; completed {result["completed_periods"]}/'
              f'{config["num_periods"]} periods; result: {path}')
        return 0 if result['run_status'] == 'success' else 1
    finally:
        provider.close()


if __name__ == '__main__':
    raise SystemExit(main())
