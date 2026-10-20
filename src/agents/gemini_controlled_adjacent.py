"""LLM-gated, bidirectional opening information followed by ordering."""

import hashlib
import json

from google.genai import types

from agents.gemini_provider import (
    GeminiDecisionProvider, GeminiOutputError, GeminiProviderError,
    PROVIDER_CONFIG, build_prompt as build_local_prompt, latest_order_definition,
)
from agents.gemini_local_communication import _unique_pairs, _validate_insight
from simulation.experiment_interface import json_safe


PROMPT_VERSION = 'gemini_controlled_adjacent_v1'
BOUNDARY = 'P6 Information and communication rule: local agent'
SHARING_SYSTEM_PROMPT = 'Decide independently whether your local information could materially affect each immediate neighbor\'s replenishment decision.'
SHARING_RULES = """P6 Controlled adjacent coordination: communication selection

This is the opening information-sharing phase, before ANY current-week orders.
In P3, latest_order is last week's realized downstream demand/order, or null
initially. No current-week downstream order or neighbor state is available yet.

Decide separately whether to share with your immediate upstream and downstream
neighbor. Share only when your own supplied information could materially change
that neighbor's replenishment timing, quantity or assessment of supply risk.
Relevant signals can include stockout risk, persistent backlog, a substantial
change in demand or incoming shipments, or a material recovery in availability.
Do not send a routine status report merely because a neighbor exists.
Unchanged conditions can still warrant sharing if a material risk persists.
Do not claim to know a neighbor's private state or whether it already knows this.

An upstream insight should describe your own likely replenishment needs.
A downstream insight should describe your own supply situation and uncertainty.
Use only your own opening state, completed history and forecast; do not promise
shipments, exact arrival dates for unshipped orders, or future order quantities.
Messages contain your opening inventory_level, backlog, upstream_backlog and
in_transit_orders, plus your insight and the source period, added by the simulator.

Retailer has no downstream agent. Manufacturer has no upstream agent.
Return ONLY one JSON object with exactly these two fields:
{{"upstream": null, "downstream": null}}
For each existing neighbor, replace null with one brief insight string of at
most 200 characters ONLY when sharing is warranted. Otherwise return null.
Never send beyond the four-stage boundary. Do not output an order in this phase.
"""
DECISION_RULES = """P6 Controlled adjacent coordination: received information

Before any orders, each stage independently decided whether its own local
opening information could materially affect each immediate neighbor's decision.
Only selected messages were delivered. Your received messages are:

{messages}

null means no message was sent, or no neighbor exists. It does NOT imply zero
inventory, no backlog or no supply risk. Do not reuse old messages as current.
Each message describes the sender's opening state this week, not your inventory.
Do not count a neighbor's stock, outstanding orders or incoming shipments as
your own. Its insight is uncertain advice, not a shipment promise or forecast.
Do not assume access to a nonadjacent agent's state. No messages are forwarded.

Orders now proceed Retailer to Manufacturer. A current downstream order in P3
is known operational demand, not an optional communication message; it is not
included in opening backlog and must not be counted twice. The retailer's
latest_order is last week's demand, as defined in P3.

P7 Decision

Choose your new replenishment order: one integer from 0 to {order_capacity}
inclusive. Return ONLY the integer enclosed in triple brackets, e.g. [[[2]]].
"""


def build_sharing_prompt(stage, observation, context):
    prefix = build_local_prompt(stage, observation, context).split(BOUNDARY, 1)[0]
    prefix = prefix.replace(latest_order_definition(stage),
                            "Last week's realized downstream demand/order; null before any completed week. "
                            "It is historical demand, not a current-week order to fulfill again.")
    prefix = prefix.replace(
        'A current downstream order in P3 is a separate known\norder, not an observation used to calculate this forecast.',
        'P3 latest_order is historical here; there are no current-week orders yet.')
    return prefix + SHARING_RULES.format()


def parse_sharing_response(raw, stage):
    try:
        value = json.loads(raw, object_pairs_hook=_unique_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-JSON constant')))
    except (TypeError, ValueError) as error:
        raise ValueError('Expected a communication-selection JSON object') from error
    if not isinstance(value, dict) or set(value) != {'upstream', 'downstream'}:
        raise ValueError('Expected exactly upstream and downstream')
    for direction, insight in value.items():
        absent = (direction == 'downstream' and stage == 0) or (direction == 'upstream' and stage == 3)
        if absent and insight is not None:
            raise ValueError('No recipient beyond the supply-chain boundary')
        if insight is not None:
            _validate_insight(insight)
    return value


class GeminiControlledAdjacentProvider(GeminiDecisionProvider):
    uses_controlled_communication = True

    def __init__(self, metrics):
        super().__init__(metrics)
        self._metadata.update(
            architecture='controlled_adjacent_coordination', prompt_version=PROMPT_VERSION,
            adapter_version='controlled_adjacent_adapter_v1',
            schema_version='optional_bidirectional_opening_insights_v1',
            communication_policy='llm_selected_opening_messages_before_sequential_orders',
            sharing_latest_order='previous_completed_downstream_demand',
            sharing_prompt_sha256=hashlib.sha256(SHARING_RULES.encode()).hexdigest(),
        )
        self._sharing_config = types.GenerateContentConfig(
            system_instruction=SHARING_SYSTEM_PROMPT,
            temperature=PROVIDER_CONFIG['temperature'],
            thinking_config=types.ThinkingConfig(thinking_level='low', include_thoughts=False),
            response_mime_type='application/json',
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _build_prompt(self, stage, observation, context):
        return build_local_prompt(stage, observation, context).split(BOUNDARY, 1)[0] + DECISION_RULES.format(
            messages=json.dumps(json_safe(context['controlled_messages']), indent=2),
            order_capacity=int(observation['prod_capacity']))

    def restore_checkpoint(self, metadata, invocation_trace):
        if metadata.get('sharing_prompt_sha256') != self._metadata['sharing_prompt_sha256']:
            raise ValueError('Checkpoint communication selection prompt differs')
        super().restore_checkpoint(metadata, invocation_trace)

    def _attempt(self, stage, observation, context, attempt):
        if context.get('phase') == 'controlled_information_sharing':
            return self._sharing_attempt(stage, observation, context, attempt)
        start = len(self._invocation_trace)
        try:
            return super()._attempt(stage, observation, context, attempt)
        finally:
            for diagnostic in self._invocation_trace[start:]:
                diagnostic.update(phase='ordering', call_kind='decision',
                                  incoming_messages=json_safe(context['controlled_messages']))

    def share_information(self, stage, observation, context):
        return self(stage, observation, dict(context, phase='controlled_information_sharing'))

    def _sharing_attempt(self, stage, observation, context, attempt):
        prompt = build_sharing_prompt(stage, observation, context)
        diagnostic = dict(
            api_invocation_index=len(self._invocation_trace) + 1,
            decision_attempt=attempt, decision_period=context['period'], stage_index=stage,
            prompt_version=PROMPT_VERSION, phase='opening_information_sharing', call_kind='coordination',
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            system_prompt_sha256=hashlib.sha256(SHARING_SYSTEM_PROMPT.encode()).hexdigest(),
            observation=self._diagnostic_observation(observation), raw_response_text=None,
            parsing_status='not_attempted', failure_category=None, selected_messages=None,
        )
        self._invocation_trace.append(diagnostic)
        try:
            response = self._client.models.generate_content(
                model=PROVIDER_CONFIG['model'], contents=prompt, config=self._sharing_config)
        except Exception as error:
            self._record_usage(None, stage, context['period'], kind='coordination')
            self._failure(diagnostic, 'api_transport_failure', error, 'api_failure_count',
                          parsing_status='not_attempted')
            raise GeminiProviderError('Gemini communication selection failed.') from None
        self._record_usage(response, stage, context['period'], kind='coordination')
        diagnostic.update({key: self._metadata['usage_reports'][-1][key] for key in (
            'prompt_token_count', 'candidates_token_count', 'thoughts_token_count', 'total_token_count')})
        self._response_diagnostics(diagnostic, response)
        try:
            raw = response.text
            diagnostic['raw_response_text'] = raw
            selected = parse_sharing_response(raw, stage)
        except Exception as error:
            self._failure(diagnostic, 'communication_selection_failure', error, 'malformed_output_count')
            raise GeminiOutputError('Invalid communication selection response.') from None
        diagnostic.update(parsing_status='success', selected_messages=selected)
        return selected
