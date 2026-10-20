"""Rule-based s,S calibration, decisions, and baseline evaluation."""

import numpy as np
from copy import deepcopy
from numbers import Integral
from scipy.stats import norm

from simulation.env import env_creator


def compute_s_S_from_training_window(weekly_demand, lead_time, service_level=0.95,
                                     fixed_order_cost=1, holding_cost=1, training_weeks=26):
    """Freeze classical thresholds from the prefix only (sample std, ceilings)."""
    if not isinstance(training_weeks, Integral) or isinstance(training_weeks, bool) or training_weeks < 2:
        raise ValueError('training_weeks must be an integer of at least two')
    if len(weekly_demand) < training_weeks:
        raise ValueError('Insufficient training observations')
    if not all(np.isfinite(x) for x in (lead_time, service_level, fixed_order_cost, holding_cost)):
        raise ValueError('Parameters must be finite')
    if lead_time <= 0 or fixed_order_cost < 0 or holding_cost <= 0 or not 0 < service_level < 1:
        raise ValueError('Invalid lead time, cost or service level')
    training = np.asarray(weekly_demand[:training_weeks], dtype=float)
    if training.shape != (training_weeks,) or not np.isfinite(training).all() or np.any(training < 0):
        raise ValueError('Training demand must be finite and nonnegative')
    mu = float(training.mean())
    sigma = float(training.std(ddof=1))
    z = float(norm.ppf(service_level))
    s = int(np.ceil(mu * lead_time + z * sigma * np.sqrt(lead_time)))
    eoq = float(np.sqrt(2 * fixed_order_cost * mu / holding_cost))
    S = max(s, int(np.ceil(s + eoq)))
    zero_demand_fallback_applied = mu == 0 and sigma == 0
    if zero_demand_fallback_applied:
        # Explicit boundary rule, not a service-level estimate or future-demand adjustment.
        s, S = 0, 1
    return dict(mu=mu, sigma=sigma, z=z, lead_time=lead_time,
                service_level=service_level, training_weeks=int(training_weeks),
                fixed_order_cost=fixed_order_cost, holding_cost=holding_cost,
                s=s, S=S, eoq_increment=eoq,
                zero_demand_fallback_applied=zero_demand_fallback_applied)


class RuleBasedSS:
    """Decision provider with frozen thresholds and no demand/environment access."""

    def __init__(self, stage_parameters):
        self._parameters = deepcopy(stage_parameters)
        self._thresholds = tuple((stage_parameters[f'stage_{m}']['s'], stage_parameters[f'stage_{m}']['S'])
                                 for m in range(len(stage_parameters)))

    @property
    def stage_parameters(self):
        return deepcopy(self._parameters)

    def __call__(self, stage, observation, context):
        # Same position implied by FixedPolicy's sale target minus open inventory.
        position = (observation['inventory'] + observation['upstream_backlog']
                    + sum(observation['deliveries']) - observation['backlog'])
        s, S = self._thresholds[stage]
        order = max(0, S - position) if position <= s else 0
        return int(min(observation['prod_capacity'], order))


def run_sS_baseline(record, *, config=None):
    """Evaluate a frozen series; weeks 1..26 estimate the fixed policy."""
    from simulation.config import make_m5_config
    from simulation.experiment_interface import run_episode

    config = make_m5_config(record) if config is None else deepcopy(config)
    if config['num_periods'] != len(record['weekly_demand']):
        raise ValueError('Environment periods must match the input demand sequence')
    parameters = {
        f'stage_{m}': compute_s_S_from_training_window(
            record['weekly_demand'], config['lead_times'][m],
            fixed_order_cost=config['fixed_order_costs'][m], holding_cost=config['holding_costs'][m])
        for m in range(config['num_stages'])
    }
    provider = RuleBasedSS(parameters)
    env = env_creator(config)
    result = run_episode(env, provider, config.get('_provenance', {}).get('dataset_source', 'm5'), config)
    result['policy'] = dict(
        type='rule_based_sS', training_weeks=26, service_level=0.95,
        parameter_formula='lead_time_demand_plus_safety_stock_and_eoq',
        standard_deviation_ddof=1, training_demand=list(record['weekly_demand'][:26]),
        inventory_position='inventory + upstream_backlog + sum(deliveries) - backlog',
        stage_parameters=provider.stage_parameters)
    from evaluation.experiment_results import add_service_inventory_summary
    add_service_inventory_summary(result, env, config)
    return result


def run_synthetic_sS_baseline(record, num_periods=130):
    """Evaluate a prefix; calibrate environment on weeks 1..130 and policy on 1..26."""
    from simulation.config import make_synthetic_config
    from demand.demand_data import demand_fn_from_sequence

    if (not isinstance(num_periods, Integral) or isinstance(num_periods, bool)
            or not 26 <= num_periods <= len(record['weekly_demand'])):
        raise ValueError('Synthetic baseline horizon must be between 26 and the source length')
    prefix = deepcopy(record)
    prefix['weekly_demand'] = record['weekly_demand'][:num_periods]
    config = make_synthetic_config(record)
    config['num_periods'] = int(num_periods)
    config['demand_fn'] = demand_fn_from_sequence(prefix['weekly_demand'])
    config['_provenance'].update(
        source_num_periods=len(record['weekly_demand']),
        demand_window=f'weeks_1_to_{num_periods}',
        evaluation_window={'first_week': 1, 'last_week': int(num_periods)},
        policy_calibration_weeks=26, evaluation_includes_policy_calibration=True)
    return run_sS_baseline(prefix, config=config)
