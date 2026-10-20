"""Evaluation and explicit persistence outside the physical simulator."""

from project_paths import M5_RESULTS

import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np


COST_FIELDS = ('holding_cost', 'backlog_cost', 'fixed_ordering_cost', 'operational_cost')


def bullwhip(orders, customer_demand):
    """Population variance ratio over completed periods; undefined -> None."""
    if len(orders) != len(customer_demand):
        raise ValueError('Orders and customer demand must cover the same periods')
    if not orders:
        return None
    denominator = float(np.var(customer_demand, ddof=0))
    return float(np.var(orders, ddof=0) / denominator) if denominator > 0 else None


def _coefficient_of_variation(values):
    """Population standard deviation divided by mean; undefined -> None."""
    if not values:
        return None
    mean = float(np.mean(values))
    if mean <= 0:
        return None
    return float(np.std(values, ddof=0) / mean)


def bullwhip_cv_ratio(outgoing_orders, received_demand):
    """Fransoo-Wouters CV ratio: CV(outgoing orders) / CV(received demand)."""
    if len(outgoing_orders) != len(received_demand):
        raise ValueError('Outgoing orders and received demand must cover the same periods')
    outgoing_cv = _coefficient_of_variation(outgoing_orders)
    demand_cv = _coefficient_of_variation(received_demand)
    if outgoing_cv is None or demand_cv is None or demand_cv <= 0:
        return None
    return float(outgoing_cv / demand_cv)


def build_evaluation_summary(trace, stage_names, customer_demand, metrics):
    """Aggregate the recorded cost snapshots, never mutable inventory history."""
    if len(stage_names) != 4:
        raise ValueError('Global bullwhip requires exactly four supply-chain stages')
    stages = {}
    stage_orders = {}
    for stage, name in enumerate(stage_names):
        entries = [entry for entry in trace if entry['stage'] == stage]
        orders = [entry['executed_order_quantity'] for entry in entries]
        stage_orders[stage] = orders
        received_demand = customer_demand if stage == 0 else stage_orders[stage - 1]
        stages[f'stage_{stage}'] = {
            'stage_name': name,
            **{field: sum(entry[field] for entry in entries) for field in COST_FIELDS},
            'total_order_quantity': sum(orders),
            'positive_order_periods': sum(order > 0 for order in orders),
            'bullwhip': bullwhip(orders, customer_demand),
            'bullwhip_cv': bullwhip_cv_ratio(orders, received_demand),
        }
    components = [stages[f'stage_{m}']['bullwhip'] for m in range(4)]
    global_bullwhip = (math.prod(components)
                       if all(value is not None and math.isfinite(value) for value in components)
                       else None)
    # Match the undefined-value convention; never repair or omit a component.
    if global_bullwhip is not None and not math.isfinite(global_bullwhip):
        global_bullwhip = None
    operational = {
        field: sum(stage[field] for stage in stages.values())
        for field in COST_FIELDS
    }
    operational['global_operational_cost'] = operational.pop('operational_cost')
    resource_fields = (
        'llm_call_count', 'input_tokens', 'output_tokens', 'total_tokens',
        'decision_llm_call_count', 'decision_input_tokens', 'decision_output_tokens',
        'decision_total_tokens', 'coordination_llm_call_count', 'coordination_input_tokens',
        'coordination_output_tokens', 'coordination_total_tokens',
        'usage_unavailable_call_count', 'decision_usage_unavailable_call_count',
        'coordination_usage_unavailable_call_count',
    )
    return {
        'operational': operational,
        'llm_resource': {key: metrics[key] for key in resource_fields},
        'time': {'wall_clock_seconds': metrics['wall_clock_seconds']},
        'coordination': {key: value for key, value in metrics.items()
                         if key.startswith('coordination_') or key in
                         ('active_communication_links', 'per_link_message_counts')},
        'bullwhip': {key: value['bullwhip'] for key, value in stages.items()},
        'bullwhip_cv': {key: value['bullwhip_cv'] for key, value in stages.items()},
        'global_bullwhip': global_bullwhip,
        'stage_metrics': stages,
    }


def add_service_inventory_summary(result, env, config):
    """Reuse the baseline's retrospective diagnostics for any decision provider."""
    # Reuse calibration definitions. These are retrospective evaluation only.
    count = result['completed_periods']
    for m in range(config['num_stages']):
        entries = [e for e in result['action_trace'] if e['stage'] == m]
        stage = result['evaluation_summary']['stage_metrics'][f'stage_{m}']
        if not count:
            stage.update(average_inventory=None, average_backlog=None, backlog_frequency_pct=None,
                         capacity_saturation_pct=None, strict_capacity_binding_pct=None,
                         supply_shortage_pct=None)
            continue
        # Cost snapshots are immutable and h=b=1 in the frozen environment.
        inventory = np.array([e['holding_cost'] / config['holding_costs'][m] for e in entries])
        backlog = np.array([e['backlog_cost'] / config['backlog_costs'][m] for e in entries])
        available = np.array([e['observation']['inventory'] for e in entries])
        demand = env.demands[1:count+1] if m == 0 else env.orders[m-1, 1:count+1]
        pending = np.array([e['observation']['backlog'] for e in entries]) + demand
        capacity = config['prod_capacities'][m]
        stage.update(
            average_inventory=float(inventory.mean()), average_backlog=float(backlog.mean()),
            backlog_frequency_pct=float(100 * np.mean(backlog > 0)),
            capacity_saturation_pct=float(100 * np.mean(env.orders[m, 1:count+1] == capacity)),
            # The previous metric tested a fulfillment cap, which no longer exists.
            strict_capacity_binding_pct=None,
            supply_shortage_pct=float(100 * np.mean((available < pending) & (pending > 0))))
    demand_total = int(env.demands[1:count+1].sum())
    result['evaluation_summary']['service_inventory'] = {
        'scope': 'retailer; completed periods; cumulative fulfilment includes late sales',
        'cumulative_fulfilled_fraction': float(env.sales[0, 1:count+1].sum() / demand_total) if demand_total else None,
        **{key: result['evaluation_summary']['stage_metrics']['stage_0'][key]
           for key in ('average_inventory', 'average_backlog', 'backlog_frequency_pct',
                       'capacity_saturation_pct', 'strict_capacity_binding_pct', 'supply_shortage_pct')},
    }
    return result


def persist_run(result, experiment_id, run_id, *, output_root=M5_RESULTS, overwrite=False):
    """Atomically publish one strict-JSON run; duplicate IDs fail by default.

    Explicitly call for either successful or failed run results. No file I/O
    occurs inside run_episode or env.step.
    """
    for identifier in (experiment_id, run_id):
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', identifier):
            raise ValueError('IDs must start with an alphanumeric and contain only letters, digits, _, . or -')
    required = {'run_status', 'completed_periods', 'error_type', 'error_message',
                'run_metadata', 'action_trace', 'episode_reward',
                'coordination_metrics', 'evaluation_summary'}
    if not required.issubset(result):
        raise ValueError('A complete canonical result is required')
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    root = Path(output_root).resolve()
    directory = (root / experiment_id).resolve()
    if directory.parent != root:
        raise ValueError('Experiment directory must be inside output_root')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'{run_id}.json'
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{run_id}.', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            # Atomic create-if-absent, including concurrent writers; no check/replace race.
            os.link(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target
