"""Run three current 130-week M5 methods, resuming transient failures."""
if __package__ in (None, ''):
    from _bootstrap import configure_paths
else:
    from ._bootstrap import configure_paths
configure_paths()

import argparse
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

if __package__ in (None, ''):
    import run_experiments as runner
else:
    from . import run_experiments as runner
from simulation.experiment_interface import write_checkpoint

RECORD_ID = 'FOODS_1_002_WI_1_evaluation'
BATCH_METHODS = ('no_communication', 'adjacent', 'no_forecast')


def retryable(result):
    """Resume transient transport failures and malformed model responses."""
    trace = result.get('provider_invocation_trace', [])
    if (result.get('error_type') == 'GeminiOutputError' and trace
            and trace[-1].get('failure_category') == 'response_format_or_message_failure'):
        return True
    if not trace or trace[-1].get('failure_category') != 'api_transport_failure':
        return False
    error = trace[-1]
    if error.get('failure_error_type') in ('ConnectError', 'ConnectTimeout', 'ReadTimeout',
                                          'WriteTimeout', 'PoolTimeout', 'RemoteProtocolError',
                                          'ConnectionError', 'TimeoutError'):
        return True
    message = error.get('failure_error_message', '')
    return any(code in message for code in ('429', '500', '502', '503', '504',
                                            'RESOURCE_EXHAUSTED', 'UNAVAILABLE'))


def publish(path, state):
    state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_checkpoint(path, state, indent=2)
    lines = [f"# M5 {state.get('regime', 'smooth')} — current settings, {state.get('periods', 130)} weeks", '',
             'Record: `' + state['record_id'] + '`; inventory ' + str(state.get('initial_inventory', 17))
             + '; order capacity ' + str(state.get('order_capacity', 11)) + '; lead time 2.',
             'Capacity: ceil(linear 95th percentile of first 130 weekly demands).',
             'Detail, summary and checkpoint schemas match synthetic experiments.', '',
             '| Method | Status | Completed weeks | Detail | Checkpoint | Summary |',
             '| --- | --- | ---: | --- | --- | --- |']
    for job in state['jobs']:
        links = []
        for key, label in (('result', 'JSON'), ('checkpoint', 'checkpoint'), ('summary', 'summary')):
            relative = os.path.relpath(job[key], path.parent).replace('\\', '/')
            links.append(f'[{label}]({relative})')
        lines.append(f"| {job['method']} | {job['status']} | {job.get('completed_periods', 0)} | "
                     + ' | '.join(links) + ' |')
    lines += ['', 'Live progress is saved after every validated stage decision in the checkpoint.',
              'Transient network/server failures resume the same run after 10–60 seconds.',
              'Malformed model responses also resume automatically; authentication and configuration errors stop the batch.',
              'The approved submission evaluation horizon is 130 periods.', '']
    path.with_name('README.md').write_text('\n'.join(lines), encoding='utf-8')


def prepare(root, record_id=RECORD_ID, periods=130):
    record = runner.select_records('m5', record_id)[0]
    _, config = runner.prepare_config(record, 'm5', periods)
    for method in BATCH_METHODS:
        runner.run_record(record, dataset='m5', method=method, periods=periods,
                          output_root=root, preflight=True)
    batches = root / 'batches'
    number = 1
    label = f"{record['regime']}_{periods}_current"
    while (batches / f'{label}_{number:03d}').exists():
        number += 1
    directory = batches / f'{label}_{number:03d}'
    directory.mkdir(parents=True)
    jobs = []
    for method in BATCH_METHODS:
        folder = root / runner.METHODS[method]
        summary = folder / 'summary.json'
        target = runner.choose_destination(folder, record_id, summary)
        jobs.append(dict(method=method, result=str(target), checkpoint=str(target.with_suffix('.checkpoint.json')),
                         summary=str(summary), status='queued', completed_periods=0, recovery_count=0))
    path = directory / 'batch.json'
    publish(path, dict(dataset='m5', record_id=record_id, regime=record['regime'], periods=periods,
                       initial_inventory=config['init_inventories'][0], order_capacity=config['prod_capacities'][0], output_root=str(root),
                       status='prepared', jobs=jobs))
    return path


def run(path):
    state = json.loads(path.read_text(encoding='utf-8'))
    record = runner.select_records('m5', state['record_id'])[0]
    state['pid'] = os.getpid()
    state['status'] = 'running'
    publish(path, state)
    for job in state['jobs']:
        target, checkpoint = Path(job['result']), Path(job['checkpoint'])
        while True:
            if target.exists():
                saved = json.loads(target.read_text(encoding='utf-8'))
                if saved['run_status'] == 'success':
                    job.update(status='success', completed_periods=saved['completed_periods'])
                    publish(path, state)
                    break
            job['status'] = 'running'
            publish(path, state)
            old = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else None
            if old is None:
                chosen = runner.choose_destination(target.parent, record['id'], Path(job['summary']))
                if chosen != target:
                    raise ValueError('Reserved run identity changed; prepare a new batch')
            try:
                code = runner.run_record(record, dataset='m5', method=job['method'], periods=state.get('periods', 130),
                                         output_root=state['output_root'],
                                         resume_path=checkpoint if old else None, resume_checkpoint=old)
            except Exception as error:
                job.update(status='failed', error_type=type(error).__name__)
                state['status'] = 'failed'
                publish(path, state)
                return 1
            result = json.loads(target.read_text(encoding='utf-8'))
            job['completed_periods'] = result['completed_periods']
            if code == 0:
                job['status'] = 'success'
                publish(path, state)
                break
            if not retryable(result):
                job.update(status='failed', error_type=result['error_type'])
                state['status'] = 'failed'
                publish(path, state)
                return 1
            job['recovery_count'] += 1
            delay = min(60, 10 * 2 ** min(job['recovery_count'] - 1, 3))
            job.update(status='reconnecting', retry_delay_seconds=delay)
            publish(path, state)
            print(f"Reconnecting {job['method']} after {delay}s; completed {job['completed_periods']}/{state.get('periods', 130)}", flush=True)
            time.sleep(delay)
    state['status'] = 'success'
    publish(path, state)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Preflight and create index only; no API requests')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--record-id', default=RECORD_ID, help='M5 record to use when preparing')
    parser.add_argument('--periods', type=int, default=130, help='Total weeks when preparing')
    args = parser.parse_args()
    if args.prepare:
        print(prepare(runner.M5_RESULTS.resolve(), args.record_id, args.periods))
        return 0
    if args.manifest is None:
        parser.error('Use --prepare, then --manifest <batch.json>')
    return run(args.manifest.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
