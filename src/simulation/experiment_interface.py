"""Reusable decision, validation, tracing, and result interfaces."""

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter, sleep

import numpy as np

from evaluation.experiment_results import build_evaluation_summary
from demand.forecast import next_period_demand_forecast, STAGE_FORECAST_METADATA



@dataclass
class DecisionResult:
    raw_output: Any
    proposed_order_quantity: Any
    executed_order_quantity: Any = None
    decision_summary: str | None = None
    validation_status: str = 'unvalidated'
    validation_error: str | None = None
    outgoing_message: dict | None = None


class ActionValidationError(ValueError):
    """An action does not satisfy the declared stage action space."""

    def __init__(self, result: DecisionResult, stage: int, period: int):
        self.result = result
        self.stage = stage
        self.period = period
        super().__init__(result.validation_error)


def json_safe(value: Any) -> Any:
    """Convert result values to recursively JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _decision_result(decision: Any) -> DecisionResult:
    if isinstance(decision, DecisionResult):
        return decision
    return DecisionResult(raw_output=decision, proposed_order_quantity=decision)


def validate_action(decision: Any, stage: int, period: int, capacity: int) -> DecisionResult:
    """Validate an unexecuted decision without clipping or repairing it."""
    result = _decision_result(decision)
    proposed = result.proposed_order_quantity
    if not isinstance(proposed, Integral) or isinstance(proposed, (bool, np.bool_)):
        error = 'order quantity must be an integer'
    elif proposed < 0 or proposed > capacity:
        error = f'order quantity must be between 0 and {capacity}'
    else:
        result.proposed_order_quantity = int(proposed)
        result.executed_order_quantity = int(proposed)
        result.validation_status = 'valid'
        result.validation_error = None
        return result

    result.validation_status = 'invalid'
    result.validation_error = error
    result.executed_order_quantity = None
    raise ActionValidationError(result, stage, period)


def build_run_metadata(env_config_name: str, env_config: dict, provider_metadata: dict | None = None) -> dict:
    metadata = dict(env_config.get('_provenance', {}))
    metadata.update({
        'environment_config': env_config_name,
        'num_periods': env_config['num_periods'],
        'num_stages': env_config['num_stages'],
        'observation_timing': 'week_start_after_due_receipts_before_orders_and_demand',
        'deliveries_timing': 'future_weeks_only_excluding_receipts_in_observed_inventory',
        'recent_observations_policy': 'last_10_completed_period_decision_snapshots_excluding_current',
        'initial_inventories': env_config['init_inventories'],
        'fixed_order_costs': env_config['fixed_order_costs'],
        'lead_times': env_config['lead_times'],
        'capacities': env_config['prod_capacities'],
        'capacity_scope': 'new_order_quantity_only',
        'fulfillment_policy': 'min_pending_demand_and_opening_inventory',
        'capacity_saturation_definition': 'accepted_new_order_equals_order_limit',
        'strict_capacity_binding_definition': 'not_applicable_no_fulfillment_capacity',
        'holding_costs': env_config['holding_costs'],
        'backlog_costs': env_config['backlog_costs'],
    })
    if provider_metadata is not None:
        metadata['provider'] = provider_metadata
    return json_safe(metadata)


def run_episode(
    env: Any,
    decision_provider: Callable[[int, dict, dict], Any],
    env_config_name: str,
    env_config: dict,
    *,
    metrics: Any = None,
    provider_metadata: dict | None = None,
    fallback_handler: Callable[[ActionValidationError], Any] | None = None,
    checkpoint_path: Any = None,
    resume_checkpoint: dict | None = None,
    checkpoint_metadata: dict | None = None,
) -> dict:
    """Trace only executed actions; count provider/fallback attempts separately.

    Client calls remain observed by the caller through the existing metrics
    instrumentation, including calls made inside an external fallback handler.
    """
    result = {
        'run_status': 'failed',
        'completed_periods': 0,
        'error_type': None,
        'error_message': None,
        'run_metadata': build_run_metadata(env_config_name, env_config, provider_metadata),
        'action_trace': [],
        'episode_reward': 0,
    }
    if metrics is None:
        from evaluation.metrics import CoordinationMetrics
        metrics = CoordinationMetrics()

    customer_demand = []
    local_histories = [[] for _ in range(env.num_stages)]
    demand_histories = [[] for _ in range(env.num_stages)]
    local_costs = [0 for _ in range(env.num_stages)]
    local_communication = getattr(decision_provider, 'uses_local_communication', False)
    controlled_communication = getattr(decision_provider, 'uses_controlled_communication', False)
    communication_enabled = local_communication or controlled_communication
    saved_sharing = {}
    forecast_input = getattr(decision_provider, 'uses_forecast_input', True)
    if communication_enabled:
        result['communication_trace'] = []
    saved_decisions = {}
    completed_before_resume = 0
    prior_trace = []
    demand_fingerprint = hashlib.sha256(json.dumps(
        [int(env_config['demand_fn'](week)) for week in range(1, env.num_periods + 1)]
    ).encode()).hexdigest() if checkpoint_path is not None or resume_checkpoint is not None else None
    if resume_checkpoint is not None:
        if resume_checkpoint.get('checkpoint_version') != 1:
            raise ValueError('Unsupported checkpoint version; historical results cannot be resumed directly')
        old = resume_checkpoint['result']
        expected = dict(result['run_metadata'])
        expected.pop('provider', None)
        observed = dict(old['run_metadata'])
        observed.pop('provider', None)
        if expected != observed or demand_fingerprint != resume_checkpoint['demand_sha256']:
            raise ValueError('Checkpoint environment, dataset, calibration or demand differs')
        completed_before_resume = old['completed_periods']
        if not 0 <= completed_before_resume <= env.num_periods:
            raise ValueError('Invalid checkpoint completed-period count')
        prior_trace = old['action_trace']
        if len(prior_trace) != completed_before_resume * env.num_stages:
            raise ValueError('Checkpoint action trace is incomplete')
        for entry in resume_checkpoint['decisions']:
            key = (entry['period'], entry['stage'])
            if key in saved_decisions or not 1 <= key[0] <= completed_before_resume + 1 or not 0 <= key[1] < env.num_stages:
                raise ValueError('Invalid or duplicate cached decision')
            saved_decisions[key] = entry
        expected_keys = [(week, stage) for week in range(1, completed_before_resume + 1)
                         for stage in range(env.num_stages)]
        partial = sorted(key for key in saved_decisions if key[0] == completed_before_resume + 1)
        if partial != [(completed_before_resume + 1, stage) for stage in range(len(partial))] or any(
                key not in saved_decisions for key in expected_keys):
            raise ValueError('Checkpoint decisions are not a contiguous sequential prefix')
        if old['run_metadata'].get('provider') is not None:
            decision_provider.restore_checkpoint(old['run_metadata']['provider'],
                                                 old.get('provider_invocation_trace', []))
        metrics.restore(old['coordination_metrics'], resume_checkpoint['round_ids'])
        if controlled_communication:
            for entry in resume_checkpoint.get('sharing_decisions', []):
                key = (entry['period'], entry['stage'])
                if key in saved_sharing or not 1 <= key[0] <= completed_before_resume + 1 or not 0 <= key[1] < env.num_stages:
                    raise ValueError('Invalid cached sharing decision')
                saved_sharing[key] = entry
            for key in expected_keys:
                if key not in saved_sharing:
                    raise ValueError('Missing completed-period sharing decision')
            partial_sharing = sorted(k for k in saved_sharing if k[0] == completed_before_resume + 1)
            if partial_sharing != [(completed_before_resume + 1, s) for s in range(len(partial_sharing))]:
                raise ValueError('Sharing decisions are not a contiguous prefix')
            if partial and len(partial_sharing) != env.num_stages:
                raise ValueError('Ordering started before sharing completed')
        if communication_enabled:
            result['communication_trace'] = deepcopy(old['communication_trace'])

    started = perf_counter()
    previous_wall_time = metrics.wall_clock_seconds
    replay_validated = completed_before_resume == 0

    def save_checkpoint():
        if checkpoint_path is None or not replay_validated:
            return
        snapshot = dict(result)
        snapshot['run_metadata'] = dict(result['run_metadata'])
        if snapshot['run_status'] != 'success':
            snapshot['run_status'] = 'in_progress' if not snapshot['error_type'] else 'failed'
        snapshot['coordination_metrics'] = metrics.to_dict()
        snapshot['coordination_metrics']['wall_clock_seconds'] = previous_wall_time + perf_counter() - started
        if hasattr(decision_provider, 'metadata'):
            snapshot['run_metadata']['provider'] = decision_provider.metadata
            snapshot['provider_invocation_trace'] = decision_provider.invocation_trace
        payload = dict(
            checkpoint_version=1, demand_sha256=demand_fingerprint,
            runner=checkpoint_metadata or {}, result=snapshot,
            decisions=list(saved_decisions.values()), round_ids=list(metrics._rounds))
        if controlled_communication:
            payload['sharing_decisions'] = list(saved_sharing.values())
        write_checkpoint(checkpoint_path, payload)

    def record_message(message):
        # Replay reconstructs contexts but must not redeliver or recount saved messages.
        trace = result['communication_trace']
        for existing in trace:
            if all(existing[key] == message[key] for key in ('period', 'sender', 'receiver', 'phase')):
                if existing != json_safe(message):
                    raise ValueError('Replayed communication differs from checkpoint')
                return
        metrics.record_message(message['sender'], message['receiver'], round_id=message['period'])
        trace.append(json_safe(message))

    previous_callback = getattr(decision_provider, 'checkpoint_callback', None)
    if checkpoint_path is not None:
        if resume_checkpoint is None and Path(checkpoint_path).exists():
            raise ValueError('Checkpoint already exists; use resume instead of overwriting it')
        decision_provider.checkpoint_callback = save_checkpoint
        if resume_checkpoint is None:
            save_checkpoint()

    def observe(stage, opening, latest_order):
        observation = deepcopy(opening)
        observation['latest_order'] = latest_order
        observation['total_cost_so_far'] = local_costs[stage]
        observation['recent_observations'] = deepcopy(local_histories[stage])
        if forecast_input:
            history = demand_histories[stage][-26:]
            observation['downstream_demand_history'] = list(history)
            observation['next_period_demand_forecast'] = next_period_demand_forecast(history)
            observation['forecast_target_period'] = period + 1
        return observation

    with metrics.measure_run():
        try:
            env.reset()
            for period in range(1, env.num_periods + 1):
                observations = env.parse_state(env.state_dict)
                action_dict = {}
                period_decisions = []
                upstream_messages = [None for _ in range(env.num_stages)]
                downstream_messages = [None for _ in range(env.num_stages)]
                downstream_decision_message = None
                controlled_messages = [{'upstream': None, 'downstream': None}
                                       for _ in range(env.num_stages)]
                if controlled_communication:
                    from agents.gemini_controlled_adjacent import parse_sharing_response
                    metrics.record_round(period)
                    # Every sharing decision uses local state only, before any new orders.
                    for sender_stage in range(env.num_stages):
                        history = demand_histories[sender_stage]
                        opening = observe(sender_stage, observations[f'stage_{sender_stage}'],
                                          history[-1] if history else None)
                        cached_sharing = saved_sharing.get((period, sender_stage))
                        if cached_sharing is not None:
                            if _observation_hash(opening) != cached_sharing['observation_sha256']:
                                raise ValueError('Replayed sharing observation differs from checkpoint')
                            selected = cached_sharing['selected']
                        else:
                            selected = decision_provider.share_information(
                                sender_stage, opening, {'period': period, 'stage': sender_stage})
                        selected = parse_sharing_response(json.dumps(selected), sender_stage)
                        saved_sharing[(period, sender_stage)] = dict(
                            period=period, stage=sender_stage,
                            observation_sha256=_observation_hash(opening), selected=selected)
                        for direction, insight in selected.items():
                            if insight is None:
                                continue
                            receiver_stage = sender_stage + (1 if direction == 'upstream' else -1)
                            message = json_safe(dict(
                                source_period=period, inventory_level=opening['inventory'],
                                backlog=opening['backlog'], upstream_backlog=opening['upstream_backlog'],
                                in_transit_orders=opening['deliveries'], planning_insight=insight))
                            received_direction = 'downstream' if direction == 'upstream' else 'upstream'
                            controlled_messages[receiver_stage][received_direction] = deepcopy(message)
                            record_message(dict(
                                period=period, sender=env.stage_names[sender_stage],
                                receiver=env.stage_names[receiver_stage],
                                direction=('downstream_to_upstream' if direction == 'upstream'
                                           else 'upstream_to_downstream'),
                                phase='controlled_opening_information_sharing',
                                outgoing_message=message, delivered_message=deepcopy(message)))
                        if period > completed_before_resume:
                            save_checkpoint()
                if local_communication:
                    # Freeze opening states and deliver all snapshots before decisions.
                    metrics.record_round(period)
                    opening_messages = []
                    for sender_stage in range(env.num_stages):
                        opening = observations[f'stage_{sender_stage}']
                        message = json_safe({
                            'inventory_level': opening['inventory'],
                            'backlog': opening['backlog'],
                            'in_transit_orders': opening['deliveries'],
                        })
                        opening_messages.append(message)
                    for sender_stage, message in enumerate(opening_messages):
                        sender = env.stage_names[sender_stage]
                        for receiver_stage in (sender_stage - 1, sender_stage + 1):
                            if not 0 <= receiver_stage < env.num_stages:
                                continue
                            incoming = upstream_messages if receiver_stage < sender_stage else downstream_messages
                            incoming[receiver_stage] = deepcopy(message)
                            receiver = env.stage_names[receiver_stage]
                            record_message(dict(
                                period=period, sender=sender, receiver=receiver,
                                direction=('upstream_to_downstream' if receiver_stage < sender_stage
                                           else 'downstream_to_upstream'),
                                phase='opening_information_sharing',
                                outgoing_message=deepcopy(message),
                                delivered_message=deepcopy(message)))
                for stage in range(env.num_stages):
                    latest_order = (
                        customer_demand[-1] if customer_demand else None
                    ) if stage == 0 else action_dict[f'stage_{stage - 1}']
                    observation = observe(stage, observations[f'stage_{stage}'], latest_order)
                    context = {'period': period, 'stage': stage, 'downstream_orders': deepcopy(action_dict)}
                    if controlled_communication:
                        context['controlled_messages'] = deepcopy(controlled_messages[stage])
                    if local_communication:
                        context['downstream_opening_message'] = deepcopy(downstream_messages[stage])
                        context['downstream_message'] = deepcopy(downstream_decision_message)
                        context['upstream_message'] = deepcopy(upstream_messages[stage])
                    attempt_count = 1
                    try:
                        cached = saved_decisions.get((period, stage))
                        if cached is not None:
                            if _observation_hash(observation) != cached['observation_sha256']:
                                raise ValueError('Replayed observation differs from checkpoint')
                            decision = DecisionResult(**cached['decision'])
                            attempt_count = cached['attempt_count']
                        else:
                            decision = decision_provider(stage, observation, context)
                        validated = validate_action(decision, stage, period, int(env.prod_capacities[stage]))
                    except ActionValidationError as error:
                        if fallback_handler is None:
                            raise
                        attempt_count += 1
                        decision = fallback_handler(error)
                        validated = validate_action(decision, stage, period, int(env.prod_capacities[stage]))
                    period_decisions.append((stage, observation, validated, attempt_count))
                    action_dict[f'stage_{stage}'] = validated.executed_order_quantity
                    if local_communication:
                        outgoing = validated.outgoing_message
                        if stage == env.num_stages - 1:
                            if outgoing is not None:
                                raise ValueError('Final stage has no upstream recipient')
                        else:
                            if not isinstance(outgoing, dict) or set(outgoing) != {
                                    'current_order_to_upstream', 'planning_insight'}:
                                raise ValueError('Expected only the current order and local insight')
                            order = outgoing['current_order_to_upstream']
                            insight = outgoing['planning_insight']
                            if type(order) is not int or order != validated.executed_order_quantity:
                                raise ValueError('Message order differs from the accepted action')
                            if not isinstance(insight, str) or not insight.strip() or len(insight) > 200 or '\n' in insight or '\r' in insight:
                                raise ValueError('Planning insight must be one brief sentence of at most 200 characters')
                            downstream_decision_message = json_safe(outgoing)
                            sender = env.stage_names[stage]
                            receiver = env.stage_names[stage + 1]
                            record_message(dict(
                                period=period, sender=sender, receiver=receiver,
                                direction='downstream_to_upstream',
                                phase='sequential_order_and_insight',
                                outgoing_message=json_safe(outgoing),
                                delivered_message=deepcopy(downstream_decision_message)))
                    saved_decisions[(period, stage)] = dict(
                        period=period, stage=stage, observation_sha256=_observation_hash(observation),
                        decision=json_safe(vars(validated)), attempt_count=attempt_count)
                    if period > completed_before_resume:
                        save_checkpoint()

                next_states, rewards, _, _, infos = env.step(action_dict)
                customer_demand.append(int(env.demands[period]))
                result['completed_periods'] = period
                result['episode_reward'] += sum(rewards.values())
                for stage, observation, validated, attempt_count in period_decisions:
                    # Commit histories only after the entire physical period completes.
                    local_histories[stage].append(json_safe({
                        'inventory_level': observation['inventory'],
                        'backlog': observation['backlog'],
                        'upstream_backlog': observation['upstream_backlog'],
                        'latest_order': observation['latest_order'],
                        'in_transit_orders': observation['deliveries'],
                        'total_cost_so_far': observation['total_cost_so_far'],
                    }))
                    local_histories[stage] = local_histories[stage][-10:]
                    demand_histories[stage].append(int(env.demands[period]) if stage == 0
                                                   else action_dict[f'stage_{stage - 1}'])
                    demand_histories[stage] = demand_histories[stage][-26:]
                    local_costs[stage] += int(infos[f'stage_{stage}']['operational_cost'])
                    result['action_trace'].append(_trace_entry(
                        period, stage, observation, validated, rewards[f'stage_{stage}'], attempt_count,
                        infos[f'stage_{stage}']))
                if period <= completed_before_resume:
                    if result['action_trace'][-env.num_stages:] != prior_trace[(period - 1) * env.num_stages:period * env.num_stages]:
                        raise ValueError('Replayed costs/actions differ from checkpoint')
                    if period == completed_before_resume:
                        replay_validated = True
                else:
                    save_checkpoint()

            result['run_status'] = 'success'
        except BaseException as error:
            result['error_type'] = type(error).__name__
            result['error_message'] = str(error)
    result['coordination_metrics'] = json_safe(metrics.to_dict())
    if checkpoint_path is not None:
        decision_provider.checkpoint_callback = previous_callback
        save_checkpoint()
    # Optional provider diagnostics retain calls from incomplete periods too.
    # They are not executed actions and do not enter physical/economic evaluation.
    invocation_trace = getattr(decision_provider, 'invocation_trace', None)
    if invocation_trace is not None:
        result['provider_invocation_trace'] = json_safe(invocation_trace)
        result['run_metadata']['provider'] = decision_provider.metadata
    result['evaluation_summary'] = build_evaluation_summary(
        result['action_trace'], env.stage_names, customer_demand, result['coordination_metrics'])

    return json_safe(result)


def write_checkpoint(path, payload, *, indent=None):
    """Publish a flushed checkpoint atomically so interrupted writes keep the old copy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.checkpoint.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(payload, default=json_safe, allow_nan=False, indent=indent))
            if indent is not None:
                stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                sleep(0.05)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _observation_hash(observation):
    return hashlib.sha256(json.dumps(json_safe(observation), sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def _trace_entry(period, stage, observation, decision, reward, attempt_count, costs):
    return {
        'period': period,
        'stage': stage,
        'observation': json_safe(observation),
        'raw_output': json_safe(decision.raw_output),
        'proposed_order_quantity': json_safe(decision.proposed_order_quantity),
        'executed_order_quantity': json_safe(decision.executed_order_quantity),
        'validation_status': decision.validation_status,
        'validation_error': decision.validation_error,
        'reward': json_safe(reward),
        **{name: int(costs[name]) for name in
           ('holding_cost', 'backlog_cost', 'fixed_ordering_cost', 'operational_cost')},
        'decision_summary': json_safe(decision.decision_summary),
        'attempt_count': attempt_count,
    }
