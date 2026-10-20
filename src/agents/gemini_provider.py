"""Stateless Gemini decisions and an explicit, single-run M5 entry point."""

from project_paths import M5_RESULTS, PROJECT_ROOT

import argparse
import hashlib
import inspect
from copy import deepcopy
from importlib.metadata import version
import json
import os
import re
from pathlib import Path
import tempfile
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, StrictInt

from simulation.experiment_interface import DecisionResult, json_safe, run_episode, STAGE_FORECAST_METADATA
from evaluation.experiment_results import persist_run, add_service_inventory_summary
from evaluation.summary import (next_run_path, load_summary_rows,
                                persist_summary_row, build_result_summary)

REPRESENTATIVE_ID = 'FOODS_1_002_WI_1_evaluation'
EXPERIMENT_ID = 'gemini_no_communication'
STAGES = ('Retailer', 'Wholesaler', 'Distributor', 'Manufacturer')
PROVIDER_CONFIG = {
    'provider': 'google',
    'model': 'gemini-3.8-flash',
    'thinking_level': 'low',
    'temperature': 0.0,
    'prompt_version': 'gemini_no_communication_v7',
    'adapter_version': 'gemini_no_communication_adapter_v2',
    'demand_forecast': deepcopy(STAGE_FORECAST_METADATA),
    'schema_version': 'decision_action_v2',
    'model_output_format': 'triple_bracket_integer_v1',
    'history_policy': 'stateless_per_period',
    'architecture': 'no_communication',
    'local_history_policy': 'last_10_completed_period_decision_snapshots_excluding_current',
    'retry_policy': {
        'sdk_max_attempts': 1, 'adapter_retries': 2,
        'transport_retries': 2, 'malformed_output_reprompts': 2,
        'invalid_action_reprompts': 0,
        'entrypoint_fallback': 'none',
    },
}


class DecisionAction(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    # Economic bounds intentionally remain the existing validate_action's job.
    order_quantity: StrictInt
    decision_summary: None = None


class GeminiProviderError(RuntimeError):
    """Sanitized transport/client failure, safe for canonical error persistence."""


class GeminiOutputError(ValueError):
    """Missing or malformed structured output; never repaired or re-prompted."""


SYSTEM_PROMPT = "Make one replenishment decision for the stated supply-chain stage using only the supplied local information."

NO_COMMUNICATION_PROMPT = """P1 Environment

You are the {stage_name} in a sequential four-stage supply chain:
Retailer, Wholesaler, Distributor, Manufacturer.

Your objective is to help minimize total supply-chain operating cost over time.
Use only the local information supplied below.

Each week, due shipments arrive before ordering decisions.
Stages then decide sequentially from Retailer to Manufacturer.
Customer demand and downstream orders are fulfilled after these decisions.
Unfulfilled demand is carried forward as backlog.

A new replenishment order does not immediately increase available inventory.
Orders may be fulfilled partially. Unshipped quantities remain outstanding.
Shipments dispatched in week t arrive at the beginning of week t+2.
The lead time starts when goods are dispatched, not when an order is placed.

Your stage's new-order capacity is {order_capacity} units per week.
It limits only your new replenishment order, not your downstream fulfillment.
Fulfillment is limited by available inventory and total pending downstream
demand (last week's backlog plus this week's new demand or downstream order).
You may fulfill more than your order capacity when inventory and pending demand
allow it. There is no separate production, handling or shipment capacity limit.
At the Manufacturer, external replenishment orders are fully dispatched
in the ordering week and arrive two weeks later.

P2 Objective function

Total supply-chain cost is the sum of all stages' operating costs.

Your stage's cost in a week is:
end-of-week inventory
+ end-of-week backlog
+ 1 if order quantity > 0

The three terms represent:
- holding cost,
- backlog cost,
- fixed ordering cost.

Each cost coefficient is 1. There is no variable ordering cost.

Costs are measured after fulfillment, not from the opening observation.
The fixed ordering cost applies even if the supplier cannot ship immediately.
Consider the effect of your order on future inventory, backlog and costs.

P3 Current period information

Your current observations are:

{local_json}

Definitions:

inventory_level:
The stock available at the beginning of this week, including last week's
remaining stock and all shipments due at the beginning of this week.
These opening receipts have already been included in inventory_level.

backlog:
Unfulfilled downstream demand carried forward from last week's end.
It does not yet include this week's new customer demand or downstream order.

upstream_backlog:
Your previously placed replenishment orders that your immediate upstream
supplier has not yet shipped. This is your outstanding order quantity, not
your own backlog and not stock already in transit. Do not count it as available
inventory or as a new order to submit again. It has no guaranteed arrival date.
For the Manufacturer this value is zero at the external-supplier boundary.

latest_order:
{latest_order_definition}

in_transit_orders:
Committed shipments scheduled to arrive at the beginning of next week and
the week after next, respectively. These quantities are not included in
inventory_level. Do not add this week's opening receipts again.

total_cost_so_far:
Your accumulated operating cost through the end of last week.

P4 Recent observations

The following observations cover up to ten completed periods, oldest to newest.
They are the decision-time snapshots from those earlier periods. The current
period's observation and newly decided orders are excluded. They are not
end-of-week outcomes. The list is empty
in the first period. Current-period information is provided separately in P3.

{recent_observations_json}

P5 Demand forecast

The demand forecasting module estimates your stage's downstream demand of:
{forecast_value_statement}

for week {forecast_target_period}, the week after the current decision week.

The prediction target is:
{forecast_target_definition}

This forecast is generated from previously realized demand using a trailing
26-week simple moving average of this stage's own downstream demand, including
zero-demand weeks. With fewer than 26 completed weeks, the tool uses the available-history
mean. The history ends last week: this week's customer demand or downstream
order is not included. A current downstream order in P3 is a separate known
order, not an observation used to calculate this forecast.

If no completed downstream demand history exists, no forecast is available.
This is an uncertain one-week-ahead estimate, not a guaranteed order
and not a forecast of total demand over the two-week shipment lead time.

Consider it together with your current inventory, backlog, incoming orders,
and recent observations.

P6 Information and communication rule: local agent

This is a no-communication, local-information setting.

Use only the information provided in P3-P5.

You do not receive messages or private information about other agents'
inventory, backlog, forecasts, costs, or reasoning.

You do not send messages to other agents.

Your own outstanding orders and incoming shipments are local order-status
information, not access to the supplier's complete internal state.

You may reason from your supplied observations, but do not treat estimates
of another stage's private state as known facts.

P7 Decision

Decide how many units to order now.

Your order must be one integer from:
0 to {order_capacity}

inclusive.

Return only the integer enclosed in triple brackets.

Example:

[[[2]]]

Do not include any explanation, text, calculation, or additional output."""


def latest_order_definition(stage):
    """Distinguish completed customer demand from the current new stage order."""
    if stage == 0:
        return (
            'Customer demand realized last week at your Retailer stage; '
            'null means none is available yet. Current-period customer demand is not observed. '
            'This is not new demand to fulfill again. Any unfulfilled portion is already in backlog.'
        )
    return (
        f'The new replenishment order decided by your direct downstream stage '
        f'({STAGES[stage - 1]}) and accepted for this week at your {STAGES[stage]} stage. '
        'It has not yet been included in the observed backlog and is not a guaranteed '
        'fully fulfilled order.'
    )


def build_prompt(stage, observation, context):
    """Render only allowlisted local information, never complete state/context."""
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
    # The runner supplies only snapshots from completed earlier periods.
    recent = [{key: entry[key] for key in local}
              for entry in observation['recent_observations'][-10:]]
    forecast = observation['next_period_demand_forecast']
    return NO_COMMUNICATION_PROMPT.format(
        stage_name=STAGES[stage],
        local_json=json.dumps(json_safe(local), indent=2),
        latest_order_definition=latest_order_definition(stage),
        recent_observations_json=json.dumps(json_safe(recent), indent=2),
        forecast_target_period=context['period'] + 1,
        forecast_target_definition=('End-customer demand at the Retailer.' if stage == 0 else
                                    f'New replenishment orders from your direct downstream stage ({STAGES[stage - 1]}), not sales or shipments and not accumulated backlog.'),
        forecast_value_statement=('No forecast is available yet.' if forecast is None
                                  else f'{forecast} units'),
        order_capacity=int(observation['prod_capacity']),
    )


def _token_count(value):
    return type(value) is int and value >= 0


def _sdk_field(value, name):
    """Optional diagnostic fields must not interfere with decision extraction."""
    try:
        return value.get(name) if isinstance(value, dict) else getattr(value, name, None)
    except Exception:
        return None


def _safety_ratings(value):
    if not isinstance(value, (list, tuple)):
        return None
    return [json_safe({key: _sdk_field(rating, key) for key in (
        'blocked', 'category', 'probability', 'probability_score', 'severity', 'severity_score')})
        for rating in value]


class GeminiDecisionProvider:
    def __init__(self, metrics):
        # Default dotenv search starts at this module and finds project-root .env.
        # Environment variables retain precedence. Initialization is once per run.
        load_dotenv()
        if not os.environ.get('GEMINI_API_KEY', '').strip():
            raise ValueError('GEMINI_API_KEY is not configured. Add it to the local .env file or '
                             'environment before running a live Gemini experiment.')
        self.metrics = metrics
        self._metadata = deepcopy(PROVIDER_CONFIG)
        self._metadata['sdk_version'] = version('google-genai')
        self._metadata['usage_reports'] = []
        self._metadata['failed_invocation_count'] = 0
        self._metadata['api_failure_count'] = 0
        self._metadata['malformed_output_count'] = 0
        self._metadata['invalid_action_count'] = 0
        self._invocation_trace = []
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=PROVIDER_CONFIG['temperature'],
            thinking_config=types.ThinkingConfig(thinking_level='low', include_thoughts=False),
            response_mime_type='text/plain',
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            # Read the required environment variable explicitly so GOOGLE_API_KEY
            # cannot silently select different credentials. Never retain it in metadata.
            self._client = genai.Client(
                api_key=os.environ['GEMINI_API_KEY'], vertexai=False,
                http_options=types.HttpOptions(
                    api_version='v1beta', retry_options=types.HttpRetryOptions(attempts=1)),
            )
        except Exception:
            raise GeminiProviderError('Gemini client initialization failed.') from None

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def invocation_trace(self):
        return deepcopy(self._invocation_trace)

    def _build_prompt(self, stage, observation, context):
        return build_prompt(stage, observation, context)

    def _diagnostic_observation(self, observation):
        local = {key: observation[key] for key in (
            'inventory', 'backlog', 'upstream_backlog', 'latest_order', 'deliveries', 'total_cost_so_far',
            'next_period_demand_forecast', 'prod_capacity')}
        local.update({key: observation.get(key) for key in (
            'recent_observations', 'downstream_demand_history', 'forecast_target_period')})
        return json_safe(local)

    def _failure(self, diagnostic, category, error, counter, *, parsing_status='failed'):
        # Exception messages may contain transport credentials. Raw model text is
        # separately retained verbatim and never passed through this redaction.
        message = str(error)
        for name in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
            secret = os.environ.get(name)
            if secret:
                message = message.replace(secret, '[REDACTED]')
        diagnostic.update(failure_category=category, parsing_status=parsing_status,
                          failure_error_type=type(error).__name__, failure_error_message=message)
        if parsing_status == 'failed':
            diagnostic.update(parsing_error_type=type(error).__name__, parsing_error_message=message)
        self._metadata['failed_invocation_count'] += 1
        self._metadata[counter] += 1
        progress = getattr(self, 'checkpoint_callback', None)
        if progress is not None:
            progress()

    def restore_checkpoint(self, metadata, invocation_trace):
        """Restore counters and diagnostics, keeping this run's fresh SDK client."""
        for key in ('provider', 'model', 'prompt_version', 'adapter_version', 'schema_version',
                    'temperature', 'thinking_level', 'retry_policy', 'communication_policy',
                    'demand_forecast', 'history_policy'):
            if self._metadata.get(key) != metadata.get(key):
                raise ValueError(f'Checkpoint provider configuration differs: {key}')
        self._metadata = deepcopy(metadata)
        self._invocation_trace = deepcopy(invocation_trace)

    def _response_diagnostics(self, diagnostic, response):
        candidates = _sdk_field(response, 'candidates')
        if isinstance(candidates, (list, tuple)):
            diagnostic['candidate_count'] = len(candidates)
            diagnostic['candidate_statuses'] = [json_safe({
                'index': _sdk_field(candidate, 'index'),
                'finish_reason': _sdk_field(candidate, 'finish_reason'),
                'finish_message': _sdk_field(candidate, 'finish_message'),
                'safety_ratings': _safety_ratings(_sdk_field(candidate, 'safety_ratings')),
            }) for candidate in candidates]
            if candidates:
                diagnostic['finish_reason'] = diagnostic['candidate_statuses'][0]['finish_reason']
        feedback = _sdk_field(response, 'prompt_feedback')
        if feedback is not None:
            diagnostic['prompt_feedback'] = json_safe({
                'block_reason': _sdk_field(feedback, 'block_reason'),
                'block_reason_message': _sdk_field(feedback, 'block_reason_message'),
                'safety_ratings': _safety_ratings(_sdk_field(feedback, 'safety_ratings')),
            })

    def close(self):
        try:
            self._client.close()
        except Exception:
            raise GeminiProviderError('Gemini client cleanup failed.') from None

    def _record_usage(self, response, stage, period, *, kind='decision'):
        usage = getattr(response, 'usage_metadata', None)
        reported = {name: getattr(usage, name, None) for name in (
            'prompt_token_count', 'candidates_token_count', 'thoughts_token_count',
            'total_token_count')}
        # Retain only numeric usage, never the SDK response/headers or hidden thoughts.
        reported = {key: value if _token_count(value) else None
                    for key, value in reported.items()}
        self._metadata['usage_reports'].append(dict(stage=stage, period=period, kind=kind, **reported))
        incoming = reported['prompt_token_count']
        total = reported['total_token_count']
        candidates = reported['candidates_token_count']
        thoughts = reported['thoughts_token_count']
        outgoing = None
        if _token_count(incoming) and _token_count(total) and total >= incoming:
            outgoing = total - incoming  # Includes thinking exactly once.
        elif _token_count(candidates) and _token_count(thoughts):
            outgoing = candidates + thoughts
        self.metrics.record_call(kind, {'usage': {
            'prompt_tokens': incoming, 'completion_tokens': outgoing}})

    def __call__(self, stage, observation, context):
        max_attempts = 1 + self._metadata['retry_policy']['adapter_retries']
        for attempt in range(1, max_attempts + 1):
            try:
                return self._attempt(stage, observation, context, attempt)
            except (GeminiProviderError, GeminiOutputError) as error:
                if attempt == max_attempts:
                    raise
                if isinstance(error, GeminiProviderError):
                    time.sleep(2 ** (attempt - 1))

    def _attempt(self, stage, observation, context, attempt):
        prompt = self._build_prompt(stage, observation, context)
        diagnostic = {
            'api_invocation_index': len(self._invocation_trace) + 1,
            'decision_attempt': attempt,
            'decision_period': context['period'], 'stage_index': stage, 'stage_name': STAGES[stage],
            'prompt_version': self._metadata['prompt_version'],
            'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
            'system_prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode('utf-8')).hexdigest(),
            'observation': self._diagnostic_observation(observation),
            'raw_response_text': None, 'response_text_available': False,
            'candidate_count': None, 'finish_reason': None, 'candidate_statuses': None,
            'prompt_feedback': None,
            'prompt_token_count': None, 'candidates_token_count': None,
            'thoughts_token_count': None, 'total_token_count': None,
            'parsing_status': 'not_attempted', 'parsing_error_type': None,
            'parsing_error_message': None, 'parsed_order_quantity': None,
            'failure_category': None, 'failure_error_type': None, 'failure_error_message': None,
        }
        diagnostic['observation']['recent_observations'] = json_safe([
            {key: entry[key] for key in ('inventory_level', 'backlog', 'upstream_backlog', 'latest_order',
                                        'in_transit_orders', 'total_cost_so_far')}
            for entry in observation['recent_observations'][-10:]])
        # Retain each invocation independently of env.step()/executed-action tracing.
        self._invocation_trace.append(diagnostic)
        try:
            response = self._client.models.generate_content(
                model=PROVIDER_CONFIG['model'], contents=prompt, config=self._config)
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
        except AttributeError as error:
            category = ('response_text_extraction_failure' if inspect.getattr_static(response, 'text', None)
                        is not None else 'response_text_unavailable')
            self._failure(diagnostic, category, error, 'malformed_output_count')
            raise GeminiOutputError('Gemini returned missing or malformed triple-bracket integer output.') from None
        except Exception as error:
            self._failure(diagnostic, 'response_text_extraction_failure', error, 'malformed_output_count')
            raise GeminiOutputError('Gemini returned missing or malformed triple-bracket integer output.') from None
        if isinstance(raw, str):
            diagnostic['raw_response_text'] = raw  # Exact text, before strip or parsing.
            diagnostic['response_text_available'] = True
        if raw is None or raw == '':
            self._failure(diagnostic, 'response_text_unavailable',
                          ValueError('response.text is null or empty'), 'malformed_output_count')
            raise GeminiOutputError('Gemini returned missing or malformed triple-bracket integer output.') from None
        try:
            # Unchanged full-response acceptance rule.
            match = re.fullmatch(r'\[\[\[(-?[0-9]+)\]\]\]', raw.strip())
        except Exception as error:
            self._failure(diagnostic, 'response_text_extraction_failure', error, 'malformed_output_count')
            raise GeminiOutputError('Gemini returned missing or malformed triple-bracket integer output.') from None
        if match is None:
            self._failure(diagnostic, 'regex_format_failure', ValueError('Invalid bracket format'),
                          'malformed_output_count')
            raise GeminiOutputError('Gemini returned missing or malformed triple-bracket integer output.') from None
        try:
            action = DecisionAction(order_quantity=int(match.group(1)))
        except Exception as error:
            self._failure(diagnostic, 'integer_conversion_failure', error, 'malformed_output_count')
            raise GeminiOutputError('Gemini returned missing or malformed triple-bracket integer output.') from None
        diagnostic.update(parsing_status='success', parsed_order_quantity=action.order_quantity)
        if not 0 <= action.order_quantity <= observation['prod_capacity']:
            self._failure(diagnostic, 'invalid_action',
                          ValueError(f"order quantity must be between 0 and {observation['prod_capacity']}"),
                          'invalid_action_count', parsing_status='success')
        return DecisionResult(raw_output=raw,
                              proposed_order_quantity=action.order_quantity,
                              decision_summary=action.decision_summary)


def preflight(*, record_id=REPRESENTATIVE_ID, num_periods=277, output_root=None):
    """Validate local prerequisites and destination without any API requests."""
    from evaluation.metrics import CoordinationMetrics
    from simulation.config import make_m5_config
    from demand.m5_demand import get_representative

    if (PROVIDER_CONFIG.get('architecture') != 'no_communication'
            or PROVIDER_CONFIG.get('model') != 'gemini-3.8-flash'
            or PROVIDER_CONFIG.get('thinking_level') != 'low'):
        raise ValueError('The frozen no-communication model configuration is required')
    provider = GeminiDecisionProvider(CoordinationMetrics())
    try:
        record = get_representative(record_id)
        if record['id'] != record_id:
            raise ValueError('Representative record identity mismatch')
        config = make_m5_config(record)
        if config['num_stages'] != 4 or config['num_periods'] != 277:
            raise ValueError('Expected the frozen four-stage, 277-period environment')
        if config['lead_times'] != [2]*4:
            raise ValueError('The no-communication template requires two-period lead times')
        if type(num_periods) is not int or not 1 <= num_periods <= 277:
            raise ValueError('num_periods must be an integer from 1 to 277')
        config['num_periods'] = num_periods
        config['_provenance']['demand_window'] = f'weeks_1_to_{num_periods}'
        config['_provenance'].update(evaluation_window={'first_week': 1, 'last_week': num_periods})
        root = Path(output_root) if output_root is not None else M5_RESULTS
        destination = next_run_path(root.resolve() / EXPERIMENT_ID, record_id)
        rows = load_summary_rows(destination.parent / 'summary.json')
        run_id = destination.stem[len(record_id) + 1:]
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
    parser.add_argument('--preflight', action='store_true', help='Check locally without generating decisions')
    parser.add_argument('--record-id', default=REPRESENTATIVE_ID,
                        help='Representative M5 record ID')
    parser.add_argument('--periods', type=int, default=277,
                        help='Number of weekly periods from the record prefix (default: 277)')
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
                             checkpoint_metadata=dict(dataset='m5', method='no_communication',
                                                      record_id=args.record_id, periods=args.periods))
        # Existing provider provenance slot also preserves usage on failed/unexecuted calls.
        result['run_metadata']['provider'] = provider.metadata
        result['run_metadata']['run_id'] = run_id
        add_service_inventory_summary(result, env, config)
        path = persist_run(result, EXPERIMENT_ID, destination.stem, output_root=destination.parent.parent)
        summary_path = path.parent / 'summary.json'
        persist_summary_row(summary_path, build_result_summary(result))
        print(f"Run {result['run_status']}; result: {path}")
        print(f'Summary: {summary_path}')
        return 0 if result['run_status'] == 'success' else 1
    finally:
        provider.close()


if __name__ == '__main__':
    raise SystemExit(main())
