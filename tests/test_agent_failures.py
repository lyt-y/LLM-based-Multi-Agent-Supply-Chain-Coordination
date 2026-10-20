"""Offline only: client construction and all generation calls are mocked."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import importlib
import io
import json
import os
import re
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluation.metrics import CoordinationMetrics
from simulation.config import env_configs
from simulation.env import env_creator
from simulation.experiment_interface import ActionValidationError, DecisionResult, run_episode, validate_action
from evaluation.experiment_results import persist_run
import agents.gemini_provider as gp
from _gemini_fixtures import GeminiFixture, response, local_from_prompt, adjacent_context

class AgentFailureTests(GeminiFixture):
    def test_forensic_format_diagnostics_keep_exact_raw_text(self):
        for raw in ('2', '[[2]]', 'explanation [[[2]]]', ' \n[[[2]]] explanation\n'):
            self.client.models.generate_content.return_value = response(text=raw)
            with self.assertRaises(gp.GeminiOutputError):
                self.provider(0, self.observation, self.context)
            diagnostic = self.provider.invocation_trace[-1]
            self.assertEqual(diagnostic['raw_response_text'], raw)
            self.assertTrue(diagnostic['response_text_available'])
            self.assertEqual(diagnostic['failure_category'], 'regex_format_failure')
            self.assertEqual(diagnostic['parsing_status'], 'failed')
            self.assertEqual(diagnostic['parsing_error_type'], 'ValueError')
            self.assertEqual(diagnostic['parsing_error_message'], 'Invalid bracket format')
        self.assertEqual(self.provider.metadata['malformed_output_count'], 12)
        self.assertEqual(self.metrics.decision_llm_call_count, 12)

    def test_forensic_text_unavailable_and_extraction_failure(self):
        class BrokenExtraction:
            @property
            def text(self):
                raise RuntimeError('SDK extraction failed')
        for reply, category, error_type in (
            (SimpleNamespace(), 'response_text_unavailable', 'AttributeError'),
            (SimpleNamespace(text=None), 'response_text_unavailable', 'ValueError'),
            (SimpleNamespace(text=''), 'response_text_unavailable', 'ValueError'),
            (BrokenExtraction(), 'response_text_extraction_failure', 'RuntimeError'),
        ):
            self.client.models.generate_content.return_value = reply
            with self.assertRaises(gp.GeminiOutputError):
                self.provider(0, self.observation, self.context)
            diagnostic = self.provider.invocation_trace[-1]
            self.assertEqual(diagnostic['failure_category'], category)
            self.assertEqual(diagnostic['parsing_error_type'], error_type)
            self.assertIsNone(diagnostic['candidate_count'])
            self.assertIsNone(diagnostic['finish_reason'])
        self.assertEqual(self.provider.metadata['malformed_output_count'], 12)
        self.assertEqual(self.provider.metadata['api_failure_count'], 0)

    def test_forensic_success_invalid_range_and_finish_status(self):
        reply = response()
        reply.candidates = [SimpleNamespace(index=0, finish_reason='STOP', finish_message=None,
                                            safety_ratings=[SimpleNamespace(blocked=False, category='HARM')])]
        reply.prompt_feedback = SimpleNamespace(block_reason='BLOCK_REASON_UNSPECIFIED',
                                                 block_reason_message=None, safety_ratings=[])
        self.client.models.generate_content.return_value = reply
        decision = self.provider(0, self.observation, self.context)
        self.assertEqual(decision.proposed_order_quantity, 2)
        diagnostic = self.provider.invocation_trace[-1]
        self.assertEqual(diagnostic['parsing_status'], 'success')
        self.assertEqual(diagnostic['parsed_order_quantity'], 2)
        self.assertEqual(diagnostic['candidate_count'], 1)
        self.assertEqual(diagnostic['finish_reason'], 'STOP')
        self.assertFalse(diagnostic['candidate_statuses'][0]['safety_ratings'][0]['blocked'])
        self.assertEqual(diagnostic['prompt_feedback']['block_reason'], 'BLOCK_REASON_UNSPECIFIED')
        self.assertEqual(diagnostic['candidates_token_count'], 3)
        self.assertEqual(diagnostic['thoughts_token_count'], 5)
        for index, quantity in enumerate((-1, 11)):
            self.client.models.generate_content.return_value = response(quantity)
            decision = self.provider(0, self.observation, self.context)
            self.assertEqual(decision.proposed_order_quantity, quantity)
            with self.assertRaises(ActionValidationError):
                validate_action(decision, 0, 1, 10)
            self.assertIsNone(decision.executed_order_quantity)
            diagnostic = self.provider.invocation_trace[-1]
            self.assertEqual(diagnostic['failure_category'], 'invalid_action')
            self.assertEqual(diagnostic['parsed_order_quantity'], quantity)
            self.assertEqual(diagnostic['parsing_status'], 'success')
            self.assertEqual(self.provider.metadata['invalid_action_count'], index + 1)
            self.assertEqual(self.provider.metadata['malformed_output_count'], 0)

    def test_partial_period_diagnostic_trace_and_failed_raw_persist(self):
        import hashlib
        failed_response = response(text='explain [[[2]]]')
        failed_response.candidates = [SimpleNamespace(index=0, finish_reason='MAX_TOKENS')]
        self.client.models.generate_content.side_effect = [response(2)]*3 + [failed_response]*3
        with tempfile.TemporaryDirectory() as directory:
            provider, config, destination = gp.preflight(output_root=directory)
            with patch.object(gp, 'preflight', return_value=(provider, config, destination)), redirect_stdout(io.StringIO()):
                self.assertEqual(gp.main([]), 1)
            saved = json.loads(destination.read_text())
            summary = json.loads((destination.parent/'summary.json').read_text())[0]
            self.assertEqual(saved['completed_periods'], 0)
            self.assertEqual(saved['action_trace'], [])
            trace = saved['provider_invocation_trace']
            self.assertEqual(len(trace), 6)
            self.assertEqual([entry['parsing_status'] for entry in trace], ['success']*3+['failed']*3)
            for index, diagnostic in enumerate(trace):
                stage = min(index, 3)
                self.assertEqual(diagnostic['api_invocation_index'], index+1)
                self.assertEqual(diagnostic['decision_period'], 1)
                self.assertEqual(diagnostic['stage_index'], stage)
                self.assertEqual(diagnostic['stage_name'], gp.STAGES[stage])
                self.assertEqual(diagnostic['prompt_version'], 'gemini_no_communication_v7')
                prompt = self.client.models.generate_content.call_args_list[index].kwargs['contents']
                self.assertEqual(diagnostic['prompt_sha256'], hashlib.sha256(prompt.encode()).hexdigest())
                self.assertEqual(diagnostic['observation']['upstream_backlog'], 0)
                self.assertIsNone(diagnostic['observation']['next_period_demand_forecast'])
            self.assertEqual(trace[-1]['raw_response_text'], 'explain [[[2]]]')
            self.assertEqual(trace[-1]['finish_reason'], 'MAX_TOKENS')
            self.assertEqual(trace[-1]['candidate_statuses'][0]['finish_reason'], 'MAX_TOKENS')
            self.assertEqual(summary['terminal_failure_category'], 'regex_format_failure')
            self.assertNotIn('explain [[[2]]]', json.dumps(summary))
            self.assertNotIn('provider_invocation_trace', summary)
            self.assertEqual(summary['api_invocation_count'], 6)
            self.assertEqual(summary['failed_invocation_count'], 3)
            self.assertEqual(summary['malformed_output_count'], 3)

    def test_malformed_outputs_fail_after_bounded_retries_and_count_usage(self):
        for text in ('text [[[2]]]', '[[[2]]] explanation', '[[[2.0]]]', '[[[true]]]',
                     '[[[2]]]\n[[[3]]]', '2', '[[2]]', '{"order_quantity":2}'):
            self.client.models.generate_content.return_value = response(text=text)
            before = self.metrics.decision_llm_call_count
            with self.assertRaises(gp.GeminiOutputError):
                self.provider(0, self.observation, self.context)
            self.assertEqual(self.metrics.decision_llm_call_count, before + 3)
        self.assertEqual(self.client.models.generate_content.call_count, 24)
        self.assertEqual(self.provider.metadata['failed_invocation_count'], 24)
        self.assertEqual(self.provider.metadata['malformed_output_count'], 24)

    def test_transport_failure_sanitized_and_counted_for_each_attempt(self):
        self.client.models.generate_content.side_effect = RuntimeError(self.secret)
        stream = io.StringIO()
        with redirect_stdout(stream), redirect_stderr(stream):
            result = run_episode(env_creator(env_configs['m5']), self.provider, 'm5', env_configs['m5'],
                                 metrics=self.metrics, provider_metadata=self.provider.metadata)
        self.assertEqual(result['run_status'], 'failed')
        self.assertEqual(result['error_type'], 'GeminiProviderError')
        self.assertEqual(result['coordination_metrics']['decision_llm_call_count'], 3)
        self.assertEqual(result['coordination_metrics']['usage_unavailable_call_count'], 3)
        self.assertEqual(result['completed_periods'], 0)
        diagnostic = result['provider_invocation_trace'][-1]
        self.assertEqual(diagnostic['failure_category'], 'api_transport_failure')
        self.assertEqual(diagnostic['parsing_status'], 'not_attempted')
        self.assertIsNone(diagnostic['parsing_error_type'])
        self.assertIsNone(diagnostic['raw_response_text'])
        self.assertNotIn(self.secret, json.dumps(result) + stream.getvalue() + json.dumps(self.provider.metadata))

    def test_initialization_error_sanitized(self):
        self.client_factory.side_effect = RuntimeError(self.secret)
        with self.assertRaises(gp.GeminiProviderError) as caught:
            gp.GeminiDecisionProvider(self.metrics)
        self.assertNotIn(self.secret, str(caught.exception))

    def test_failed_run_summary_persisted_after_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            provider, config, destination = gp.preflight(output_root=directory)
            self.client.models.generate_content.side_effect = RuntimeError(self.secret)
            with patch.object(gp, 'preflight', return_value=(provider, config, destination)), redirect_stdout(io.StringIO()):
                self.assertEqual(gp.main([]), 1)
            summary = json.loads((destination.parent / 'summary.json').read_text())[0]
            self.assertEqual(summary['completed_periods'], 0)
            self.assertEqual(summary['api_invocation_count'], 3)
            self.assertEqual(summary['failed_invocation_count'], 3)
            self.assertEqual(summary['api_failure_count'], 3)
            self.assertEqual(summary['usage_unavailable_call_count'], 3)
            self.assertIsNone(summary['thinking_tokens'])
            self.assertEqual(summary['thinking_usage_unavailable_call_count'], 3)
            self.assertIsNone(summary['global_bullwhip'])
            self.assertIsNone(summary['cumulative_fulfilled_fraction'])
            self.assertNotIn(self.secret, json.dumps(summary))

