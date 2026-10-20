"""Shared entry point for M5 and synthetic baseline/LLM experiments."""
if __package__ in (None, ''):
    from _bootstrap import configure_paths
else:
    from ._bootstrap import configure_paths
configure_paths()

import argparse
import json
import hashlib
from copy import deepcopy
from pathlib import Path

from demand.demand_data import demand_fn_from_sequence
from demand.m5_demand import get_representative, load_representatives
from demand.synthetic_demand import load_series
from evaluation.experiment_results import add_service_inventory_summary, persist_run
from evaluation.metrics import CoordinationMetrics
from evaluation.summary import (build_baseline_summary, build_result_summary,
                                load_summary_rows, next_run_path, persist_summary_row)
from policies.ss import run_sS_baseline
from project_paths import M5_RESULTS, SYNTHETIC_RESULTS
from simulation.config import make_m5_config, make_synthetic_config
from simulation.env import env_creator
from simulation.experiment_interface import run_episode, build_run_metadata

METHODS = {'ss': 'rule_based_sS_baseline', 'no_communication': 'gemini_no_communication',
           'adjacent': 'gemini_local_communication',
           'controlled_adjacent': 'gemini_controlled_adjacent_communication',
           'no_forecast': 'gemini_no_communication_without_forecast_input'}


def select_records(dataset, record_id=None, all_records=False):
    if dataset not in ('m5', 'synthetic'):
        raise ValueError('Expected dataset m5 or synthetic')
    if dataset == 'synthetic':
        records = load_series()
    elif all_records:
        records = [r for group in load_representatives().values() for r in group]
    else:
        return [get_representative(record_id)]
    if record_id is not None:
        matches = [r for r in records if r['id'] == record_id]
        if not matches:
            raise KeyError(f'Record not found: {record_id}')
        return matches
    return records if all_records else records[:1]


def prepare_config(record, dataset, periods):
    if dataset not in ('m5', 'synthetic'):
        raise ValueError('Expected dataset m5 or synthetic')
    if type(periods) is not int or not 1 <= periods <= len(record['weekly_demand']):
        raise ValueError('periods must be between 1 and the source length')
    source_periods = len(record['weekly_demand'])
    prefix = deepcopy(record)
    prefix['weekly_demand'] = record['weekly_demand'][:periods]
    config = (make_m5_config if dataset == 'm5' else make_synthetic_config)(record)
    config['num_periods'] = periods
    config['demand_fn'] = demand_fn_from_sequence(prefix['weekly_demand'])
    config['_provenance'].update(
        source_num_periods=source_periods, demand_window=f'weeks_1_to_{periods}',
        evaluation_window={'first_week': 1, 'last_week': periods})
    return prefix, config


def make_provider(method, metrics):
    if method == 'controlled_adjacent':
        from agents.gemini_controlled_adjacent import GeminiControlledAdjacentProvider
        return GeminiControlledAdjacentProvider(metrics)
    if method == 'no_communication':
        from agents.gemini_provider import GeminiDecisionProvider
        return GeminiDecisionProvider(metrics)
    if method == 'adjacent':
        from agents.gemini_local_communication import GeminiLocalCommunicationProvider
        return GeminiLocalCommunicationProvider(metrics)
    if method == 'no_forecast':
        from agents.gemini_no_forecast import GeminiNoForecastProvider
        return GeminiNoForecastProvider(metrics)
    raise ValueError(f'Unknown LLM method: {method}')


def choose_destination(directory, record_id, summary_path):
    destination = next_run_path(directory, record_id)
    reserved = {row['run_id'] for row in load_summary_rows(summary_path)
                if row.get('record_id', row.get('representative_record_id')) == record_id}
    number = int(destination.stem[len(record_id) + 1:].removeprefix('run_'))
    while f'run_{number:03d}' in reserved:
        number += 1
    return directory / f'{record_id}_run_{number:03d}.json'


def prepare_extension(record, dataset, method, periods, source_path):
    """Validate a completed prefix, then adapt a copy for a longer new run."""
    source_path = Path(source_path).resolve()
    payload = json.loads(source_path.read_text(encoding='utf-8'))
    checkpoint = payload if source_path.name.endswith('.checkpoint.json') else None
    if method != 'ss' and checkpoint is None:
        source_path = source_path.with_suffix('.checkpoint.json')
        checkpoint = json.loads(source_path.read_text(encoding='utf-8'))
    if checkpoint is not None and checkpoint.get('checkpoint_version') != 1:
        raise ValueError('Unsupported extension checkpoint version')
    old = checkpoint['result'] if checkpoint is not None else payload
    previous = old['run_metadata']['num_periods']
    if old['run_status'] != 'success' or old['completed_periods'] != previous:
        raise ValueError('Extend requires a fully completed successful run; use --resume for interruptions')
    if type(periods) is not int or periods <= previous:
        raise ValueError('Extension periods must exceed the original horizon')
    _, old_config = prepare_config(record, dataset, previous)
    expected = build_run_metadata(dataset, old_config)
    observed = deepcopy(old['run_metadata'])
    for key in ('provider', 'run_id', 'extension'):
        observed.pop(key, None)
    if expected != observed:
        raise ValueError('Extension source environment, input, or calibration differs')
    _, config = prepare_config(record, dataset, periods)
    lineage = dict(source_file=str(source_path), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                   previous_num_periods=previous, continuation_first_period=previous + 1)
    if checkpoint is not None:
        info = checkpoint['runner']
        if any(info.get(k) != v for k, v in dict(dataset=dataset, method=method,
                                                record_id=record['id'], periods=previous).items()):
            raise ValueError('Extension checkpoint runner identity differs')
        fingerprint = hashlib.sha256(json.dumps(record['weekly_demand'][:previous]).encode()).hexdigest()
        if checkpoint['demand_sha256'] != fingerprint:
            raise ValueError('Extension demand prefix differs')
        checkpoint = deepcopy(checkpoint)
        checkpoint['result']['run_metadata'] = build_run_metadata(dataset, config, old['run_metadata']['provider'])
        checkpoint['result']['run_metadata']['extension'] = lineage
        checkpoint['demand_sha256'] = hashlib.sha256(json.dumps(record['weekly_demand'][:periods]).encode()).hexdigest()
        checkpoint['runner']['periods'] = periods
    return checkpoint, old, lineage


def run_record(record, *, dataset, method, periods, output_root, preflight=False,
               resume_path=None, resume_checkpoint=None, extend_path=None):
    if method == 'ss' and periods < 26:
        raise ValueError('s,S requires at least 26 calibration weeks')
    prefix, config = prepare_config(record, dataset, periods)
    extension_source = None
    if extend_path is not None:
        if resume_path is not None:
            raise ValueError('Cannot extend and resume simultaneously')
        resume_checkpoint, extension_source, lineage = prepare_extension(
            record, dataset, method, periods, extend_path)
        config['_provenance']['extension'] = lineage
    elif resume_checkpoint is not None:
        lineage = resume_checkpoint['result']['run_metadata'].get('extension')
        if lineage is not None:
            config['_provenance']['extension'] = deepcopy(lineage)
    directory = Path(output_root).resolve() / METHODS[method]
    # Preserve frozen baseline summary.json / summary_130.json aliases.
    summary_path = directory / ('summary_runs.json' if method == 'ss' else 'summary.json')
    destination = (Path(resume_path).resolve().with_name(
        Path(resume_path).name.removesuffix('.checkpoint.json') + '.json')
        if resume_path is not None else choose_destination(directory, record['id'], summary_path))
    if destination.parent != directory:
        raise ValueError('Resume checkpoint is outside the selected result directory')
    if resume_path is not None and destination.exists():
        existing = json.loads(destination.read_text(encoding='utf-8'))
        if existing['run_status'] == 'success':
            raise ValueError('A successful saved run cannot be overwritten by resume')
    provider = None
    try:
        if method != 'ss':
            provider = make_provider(method, CoordinationMetrics())
            if extend_path is not None:
                provider.restore_checkpoint(resume_checkpoint['result']['run_metadata']['provider'],
                                            resume_checkpoint['result'].get('provider_invocation_trace', []))
        if preflight:
            print(f"Preflight passed (no API request): {record['id']}; "
                  f"{periods} periods; output: {destination}")
            return 0
        if method == 'ss':
            result = run_sS_baseline(prefix, config=config)
            if extension_source is not None:
                old_trace = extension_source['action_trace']
                if result['policy'] != extension_source['policy'] or result['action_trace'][:len(old_trace)] != old_trace:
                    raise ValueError('Extended baseline policy or prefix trajectory differs')
            summary_builder = build_baseline_summary
        else:
            env = env_creator(config)
            result = run_episode(env, provider, dataset, config,
                                 metrics=provider.metrics, provider_metadata=provider.metadata,
                                 checkpoint_path=destination.with_suffix('.checkpoint.json'),
                                 resume_checkpoint=resume_checkpoint,
                                 checkpoint_metadata=dict(dataset=dataset, method=method,
                                                          record_id=record['id'], periods=periods))
            result['run_metadata']['provider'] = provider.metadata
            add_service_inventory_summary(result, env, config)
            summary_builder = build_result_summary
        result['run_metadata']['run_id'] = destination.stem[len(record['id']) + 1:]
        path = persist_run(result, METHODS[method], destination.stem, output_root=output_root,
                           overwrite=resume_path is not None)
        row = summary_builder(result)
        row['source_result_file'] = path.name
        persist_summary_row(summary_path, row)
        print(f"Run {result['run_status']}: {path}\nSummary: {summary_path}")
        return 0 if result['run_status'] == 'success' else 1
    finally:
        if provider is not None:
            provider.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('m5', 'synthetic'))
    parser.add_argument('--method', choices=tuple(METHODS))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--record-id')
    selection.add_argument('--all', action='store_true', dest='all_records')
    parser.add_argument('--periods', type=int, help='Submission protocol: specify 130. Generic legacy default: M5 277; synthetic 130')
    parser.add_argument('--output-root', type=Path, help='Dataset result root, before method directory')
    parser.add_argument('--preflight', action='store_true', help='No decisions or result writes')
    parser.add_argument('--resume', type=Path, help='Resume a saved *.checkpoint.json with its original dataset/model/settings')
    parser.add_argument('--extend', type=Path, help='Extend a successful result/checkpoint into a new run; requires --periods')
    args = parser.parse_args(argv)
    if args.extend is not None:
        if args.resume or args.all_records or args.record_id or args.output_root or args.periods is None:
            parser.error('--extend requires --periods and uses the source selection/output directory')
        path = args.extend.resolve()
        payload = json.loads(path.read_text(encoding='utf-8'))
        old = payload['result'] if path.name.endswith('.checkpoint.json') else payload
        metadata = old['run_metadata']
        dataset = metadata['dataset_source']
        method = next((key for key, value in METHODS.items() if value == path.parent.name), None)
        if method is None:
            parser.error('Extension source is outside a recognized method directory')
        if (args.dataset and args.dataset != dataset) or (args.method and args.method != method):
            parser.error('--extend cannot change dataset or method')
        record = select_records(dataset, metadata['series_id'])[0]
        return run_record(record, dataset=dataset, method=method, periods=args.periods,
                          output_root=path.parent.parent, preflight=args.preflight, extend_path=path)
    if args.resume is not None:
        path = args.resume.resolve()
        if not path.name.endswith('.checkpoint.json'):
            parser.error('--resume requires a checkpoint, not a historical result/summary file')
        if args.all_records or args.record_id or args.periods is not None or args.output_root:
            parser.error('--resume uses the saved selection, horizon and output directory')
        checkpoint = json.loads(path.read_text(encoding='utf-8'))
        if checkpoint.get('checkpoint_version') != 1:
            raise ValueError('Unsupported checkpoint version')
        info = checkpoint['runner']
        if ((args.dataset is not None and args.dataset != info['dataset'])
                or (args.method is not None and args.method != info['method'])):
            parser.error('--resume cannot change the saved dataset or method')
        if info['method'] == 'ss':
            raise ValueError('Resume checkpoints currently apply to LLM experiments')
        record = select_records(info['dataset'], info['record_id'])[0]
        return run_record(record, dataset=info['dataset'], method=info['method'],
                          periods=info['periods'], output_root=path.parent.parent,
                          preflight=args.preflight, resume_path=path, resume_checkpoint=checkpoint)
    args.dataset = args.dataset or 'm5'
    args.method = args.method or 'ss'
    periods = args.periods if args.periods is not None else (277 if args.dataset == 'm5' else 130)
    root = args.output_root or (M5_RESULTS if args.dataset == 'm5' else SYNTHETIC_RESULTS)
    for record in select_records(args.dataset, args.record_id, args.all_records):
        status = run_record(record, dataset=args.dataset, method=args.method,
                            periods=periods, output_root=root, preflight=args.preflight)
        if status:
            return status
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
