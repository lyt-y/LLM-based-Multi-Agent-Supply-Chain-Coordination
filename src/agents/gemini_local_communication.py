"""One Gemini episode with bidirectional adjacent information sharing."""

from project_paths import M5_RESULTS

import argparse
import hashlib
import inspect
import json
import re
from pathlib import Path
import tempfile

from google.genai import types

from simulation.experiment_interface import DecisionResult, json_safe, run_episode
from evaluation.experiment_results import add_service_inventory_summary, persist_run
from agents.gemini_provider import (
    GeminiDecisionProvider, GeminiOutputError, GeminiProviderError,
    NO_COMMUNICATION_PROMPT, PROJECT_ROOT, PROVIDER_CONFIG, REPRESENTATIVE_ID, STAGES,
    build_prompt as build_no_communication_prompt,
    build_result_summary, load_summary_rows, next_run_path, persist_summary_row,
)


EXPERIMENT_ID = 'gemini_local_communication'
NUM_PERIODS = 104  # Default: first two years of weekly demand.
PROMPT_VERSION = 'gemini_local_communication_v10'
SCHEMA_VERSION = 'adjacent_opening_state_then_order_insight_v3'
SYSTEM_PROMPT = 'Make one replenishment decision and generate one local planning insight for your immediate upstream neighbor using only the supplied local and adjacent information.'
OPENING_MESSAGE_FIELDS = frozenset(('inventory_level', 'backlog', 'in_transit_orders'))
MESSAGE_FIELDS = frozenset(('current_order_to_upstream', 'planning_insight'))


# P1-P5 remain identical to the no-communication provider.
LOCAL_COMMUNICATION_RULES = """P6 Information and communication rule: adjacent agents

At the beginning of this week, after due receipts and before any ordering
decisions, each stage shared its own inventory_level, backlog and
in_transit_orders with both immediate neighbors, where they exist.
The simulator delivered all these opening snapshots without any LLM call.
Opening snapshots contain no insight, current-week order or fulfillment outcome.

Your immediate upstream neighbor's opening snapshot is:

{upstream_message}

Your immediate downstream neighbor's opening snapshot is:

{downstream_opening_message}

The upstream snapshot is null for Manufacturer. The downstream snapshot is null
for Retailer. Neither the external supplier nor the end customer shares a state.

In each snapshot, inventory_level already includes that neighbor's receipts due
this week. backlog is its unfulfilled demand carried forward from last week's
end. in_transit_orders lists shipments due to arrive at that neighbor next week
and the week after next. These are separate components of the neighbor's opening
inventory status, not your own inventory position or available stock.
Do not count either neighbor's inventory or incoming shipments as your own.
An upstream snapshot is not a shipment promise or guaranteed arrival schedule
for your own outstanding orders.

Ordering decisions now proceed from Retailer to Manufacturer.
Your immediate downstream neighbor has already decided its current order and
sent this order-and-insight message to you:

{downstream_message}

For Retailer this message is null. Otherwise, current_order_to_upstream is the
accepted new order from your direct downstream neighbor for this week.
It is the SAME order shown as latest_order in P3, not additional demand.
It is not yet included in your opening backlog and is not a guarantee that
you can fulfill it completely this week.
planning_insight was generated together with that neighbor's current order
using its current local observations, forecast and chosen order. Treat it as
advice, not a guaranteed future order, shipment or forecast.

Use both opening snapshots and the downstream order-and-insight message
together with your own observations and forecast when choosing your order.
You do not receive an upstream insight: that stage has not decided its order yet.
Do not assume access to any nonadjacent agent's private information.

In the SAME response as your order, generate one brief planning_insight for
your immediate upstream neighbor. Base it on your own opening observations,
completed history, forecast and chosen order, explaining whether your own
replenishment needs suggest increasing, maintaining or reducing replenishment.
Do not copy, summarize or forward either neighbor's private information,
received message or insight. Do not promise fulfillment or future orders.

Your order-and-insight message is sent only to your immediate upstream neighbor
after your order is validated and before that neighbor decides. There is no
separate insight-generation LLM call. Manufacturer has no upstream agent:
it returns null for its outbound message, but its opening state was already
shared with Distributor.

P7 Decision and outbound insight

Decide how many units to order now.

Your order must be one integer from:
0 to {order_capacity}

inclusive.

Return exactly two lines.

Line 1: Return the integer enclosed in triple brackets.

Example:

[[[2]]]

Line 2: If you have an upstream neighbor, return one valid JSON object with
exactly these fields:

{{"current_order_to_upstream": <same integer as line 1>, "planning_insight": "<one brief sentence of at most 200 characters>"}}

If you are Manufacturer, return null on line 2.
Do not include any explanation, calculation or additional output."""


def _validate_snapshot(message, *, absent):
    if absent:
        if message is not None:
            raise ValueError('No snapshot is allowed beyond the adjacent-agent boundary')
        return
    if not isinstance(message, dict) or set(message) != OPENING_MESSAGE_FIELDS:
        raise ValueError('Expected exactly the three adjacent opening-state fields')
    if any(type(message[key]) is not int or message[key] < 0
           for key in ('inventory_level', 'backlog')):
        raise ValueError('Opening inventory and backlog must be nonnegative integers')
    transit = message['in_transit_orders']
    if not isinstance(transit, list) or len(transit) != 2 or any(
            type(quantity) is not int or quantity < 0 for quantity in transit):
        raise ValueError('Expected two nonnegative future shipment quantities')


def _validate_insight(insight):
    if not isinstance(insight, str) or not insight.strip() or len(insight) > 200 or '\n' in insight or '\r' in insight:
        raise ValueError('Planning insight must be one brief sentence of at most 200 characters')


def _validate_order_message(message, order):
    if not isinstance(message, dict) or set(message) != MESSAGE_FIELDS:
        raise ValueError('Expected only the current order and local planning insight')
    if type(message['current_order_to_upstream']) is not int or message['current_order_to_upstream'] != order:
        raise ValueError('Message order does not match the decision quantity')
    _validate_insight(message['planning_insight'])


def build_prompt(stage, observation, context):
    """One prompt combines local information, frozen snapshots and downstream advice."""
    original = build_no_communication_prompt(stage, observation, context)
    boundary = 'P6 Information and communication rule: local agent'
    if original.count(boundary) != 1 or NO_COMMUNICATION_PROMPT.count(boundary) != 1:
        raise ValueError('No-communication prompt boundary has changed')
    upstream = context.get('upstream_message')
    downstream_opening = context.get('downstream_opening_message')
    downstream = context.get('downstream_message')
    _validate_snapshot(upstream, absent=stage == 3)
    _validate_snapshot(downstream_opening, absent=stage == 0)
    if stage == 0:
        if downstream is not None:
            raise ValueError('Retailer cannot receive a downstream decision message')
    else:
        _validate_order_message(downstream, observation['latest_order'])
        if observation['latest_order'] < 0:
            raise ValueError('Accepted downstream orders must be nonnegative')
    return original.split(boundary, 1)[0] + LOCAL_COMMUNICATION_RULES.format(
        upstream_message=json.dumps(json_safe(upstream), indent=2),
        downstream_opening_message=json.dumps(json_safe(downstream_opening), indent=2),
        downstream_message=json.dumps(json_safe(downstream), indent=2),
        order_capacity=int(observation['prod_capacity']),
    )


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate message field')
        result[key] = value
    return result


def parse_response(raw, stage, observation):
    """Validate the single call's two lines before delivering any message."""
    if not isinstance(raw, str):
        raise ValueError('Response text is unavailable')
    lines = raw.strip().splitlines()
    if len(lines) != 2:
        raise ValueError('Expected exactly two output lines')
    match = re.fullmatch(r'\[\[\[(-?[0-9]+)\]\]\]', lines[0].strip())
    if match is None:
        raise ValueError('Invalid triple-bracket integer')
    order = int(match.group(1))
    try:
        message = json.loads(lines[1].strip(), object_pairs_hook=_unique_pairs,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-JSON constant')))
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError('Invalid order-and-insight JSON') from error
    if stage == 3:
        if message is not None:
            raise ValueError('Manufacturer has no upstream message recipient')
    else:
        _validate_order_message(message, order)
    return order, message


class GeminiLocalCommunicationProvider(GeminiDecisionProvider):
    uses_local_communication = True

    def __init__(self, metrics):
        super().__init__(metrics)
        self._metadata.update(
            architecture='local_communication', prompt_version=PROMPT_VERSION,
            adapter_version='gemini_local_communication_adapter_v4',
            schema_version=SCHEMA_VERSION,
            model_output_format='triple_bracket_integer_plus_order_insight_json_v2',
            communication_policy='bidirectional_opening_state_then_sequential_upstream_order_insight',
            upstream_snapshot_timing='week_start_after_due_receipts_before_all_decisions',
            upstream_snapshot_fields=sorted(OPENING_MESSAGE_FIELDS),
        )
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=PROVIDER_CONFIG['temperature'],
            thinking_config=types.ThinkingConfig(thinking_level='low', include_thoughts=False),
            response_mime_type='text/plain',
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _attempt(self, stage, observation, context, attempt):
        prompt = build_prompt(stage, observation, context)
        diagnostic = {
            'api_invocation_index': len(self._invocation_trace) + 1,
            'decision_attempt': attempt, 'decision_period': context['period'],
            'stage_index': stage, 'stage_name': STAGES[stage],
            'prompt_version': PROMPT_VERSION,
            'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
            'system_prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode('utf-8')).hexdigest(),
            'phase': 'ordering_and_insight', 'call_kind': 'decision',
            'observation': self._diagnostic_observation(observation),
            'incoming_message': json_safe(context.get('downstream_message')),
            'incoming_downstream_opening_message': json_safe(context.get('downstream_opening_message')),
            'incoming_upstream_message': json_safe(context.get('upstream_message')),
            'raw_response_text': None, 'response_text_available': False,
            'candidate_count': None, 'finish_reason': None, 'candidate_statuses': None,
            'prompt_feedback': None, 'prompt_token_count': None,
            'candidates_token_count': None, 'thoughts_token_count': None,
            'total_token_count': None, 'parsing_status': 'not_attempted',
            'parsing_error_type': None, 'parsing_error_message': None,
            'parsed_order_quantity': None, 'outgoing_message': None,
            'failure_category': None, 'failure_error_type': None,
            'failure_error_message': None,
        }
        diagnostic['observation']['recent_observations'] = json_safe(observation['recent_observations'][-10:])
        self._invocation_trace.append(diagnostic)
        try:
            response = self._client.models.generate_content(
                model=PROVIDER_CONFIG['model'], contents=prompt,
                config=self._config)
        except Exception as error:
            self._record_usage(None, stage, context['period'])
            self._failure(diagnostic, 'api_transport_failure', error, 'api_failure_count',
                          parsing_status='not_attempted')
            raise GeminiProviderError('Gemini decision request failed.') from None
        self._record_usage(response, stage, context['period'])
        diagnostic.update({key: self._metadata['usage_reports'][-1][key] for key in (
            'prompt_token_count', 'candidates_token_count', 'thoughts_token_count', 'total_token_count')})
        self._response_diagnostics(diagnostic, response)
        try:
            raw = response.text
        except Exception as error:
            category = ('response_text_unavailable' if isinstance(error, AttributeError)
                        and inspect.getattr_static(response, 'text', None) is None
                        else 'response_text_extraction_failure')
            self._failure(diagnostic, category, error, 'malformed_output_count')
            raise GeminiOutputError('Gemini returned a missing or invalid order/message response.') from None
        if isinstance(raw, str):
            diagnostic['raw_response_text'] = raw
            diagnostic['response_text_available'] = True
        try:
            order, message = parse_response(raw, stage, observation)
        except ValueError as error:
            self._failure(diagnostic, 'response_format_or_message_failure', error, 'malformed_output_count')
            raise GeminiOutputError('Gemini returned a missing or invalid order/message response.') from None
        diagnostic.update(parsing_status='success', parsed_order_quantity=order,
                          outgoing_message=json_safe(message))
        if not 0 <= order <= observation['prod_capacity']:
            self._failure(diagnostic, 'invalid_action',
                          ValueError(f"order quantity must be between 0 and {observation['prod_capacity']}"),
                          'invalid_action_count', parsing_status='success')
        return DecisionResult(raw_output=raw, proposed_order_quantity=order,
                              outgoing_message=message)


def preflight(*, record_id=REPRESENTATIVE_ID, num_periods=NUM_PERIODS, output_root=None):
    """Check local configuration and select the next result path without an API call."""
    from evaluation.metrics import CoordinationMetrics
    from simulation.config import make_m5_config
    from demand.m5_demand import get_representative

    provider = GeminiLocalCommunicationProvider(CoordinationMetrics())
    try:
        record = get_representative(record_id)
        if record['id'] != record_id:
            raise ValueError('Representative record identity mismatch')
        config = make_m5_config(record)
        if config['num_stages'] != 4 or config['num_periods'] != 277 or config['lead_times'] != [2]*4:
            raise ValueError('Expected the frozen four-stage, 277-period environment')
        if type(num_periods) is not int or not 1 <= num_periods <= 277:
            raise ValueError('num_periods must be an integer from 1 to 277')
        # Keep matched full-record inventory/capacity calibration and execute
        # the selected prefix of weekly demand observations.
        config['num_periods'] = num_periods
        config['_provenance']['demand_window'] = f'weeks_1_to_{num_periods}'
        config['_provenance'].update(evaluation_window={'first_week': 1, 'last_week': num_periods})
        root = Path(output_root) if output_root is not None else M5_RESULTS
        destination = next_run_path(root.resolve() / EXPERIMENT_ID, record_id)
        rows = load_summary_rows(destination.parent / 'summary.json')
        run_id = destination.stem[len(record_id) + 1:]
        # A detail file can be removed while its summary row remains. Treat
        # both as reserved identities so a new run never reuses either one.
        reserved = {row['run_id'] for row in rows
                    if row.get('record_id', row.get('representative_record_id')) == record_id}
        while run_id in reserved or destination.exists():
            number = int(run_id.removeprefix('run_')) + 1
            run_id = f'run_{number:03d}'
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
    parser.add_argument('--preflight', action='store_true', help='Check locally without API requests')
    parser.add_argument('--record-id', default=REPRESENTATIVE_ID,
                        help='Representative M5 record ID')
    parser.add_argument('--periods', type=int, default=NUM_PERIODS,
                        help='Number of weekly periods from the record prefix (default: 104)')
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
                             checkpoint_metadata=dict(dataset='m5', method='adjacent',
                                                      record_id=args.record_id, periods=args.periods))
        result['run_metadata']['provider'] = provider.metadata
        result['run_metadata']['run_id'] = run_id
        add_service_inventory_summary(result, env, config)
        path = persist_run(result, EXPERIMENT_ID, destination.stem,
                           output_root=destination.parent.parent)
        summary_path = path.parent / 'summary.json'
        summary = build_result_summary(result)
        summary['num_periods'] = config['num_periods']
        summary['demand_window'] = result['run_metadata']['demand_window']
        summary['calibration_window'] = result['run_metadata']['calibration_window']
        persist_summary_row(summary_path, summary)
        print(f"Run {result['run_status']}; result: {path}")
        print(f'Summary: {summary_path}')
        return 0 if result['run_status'] == 'success' else 1
    finally:
        provider.close()


if __name__ == '__main__':
    raise SystemExit(main())
