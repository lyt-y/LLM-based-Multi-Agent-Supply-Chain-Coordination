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

class GeminiTests(GeminiFixture):
    def test_next_run_selection_and_existing_file_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            for number in (1, 2, 3):
                path = gp.next_run_path(directory)
                self.assertEqual(path.name, f'{gp.REPRESENTATIVE_ID}_run_{number:03d}.json')
                self.assertEqual(path, gp.next_run_path(directory))
                path.write_bytes(b'original failed pilot')
            self.assertEqual((Path(directory)/(gp.REPRESENTATIVE_ID+'_run_001.json')).read_bytes(),
                             b'original failed pilot')
            self.client.models.generate_content.assert_not_called()

    def test_summary_legacy_preservation_upsert_and_record_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'summary.json'
            first = {'record_id': gp.REPRESENTATIVE_ID, 'run_status': 'failed', 'completed_periods': 14}
            (path.parent/(gp.REPRESENTATIVE_ID+'_run_001.json')).write_bytes(b'original pilot')
            path.write_text(json.dumps([first]))
            before = path.read_bytes()
            self.assertEqual(gp.load_summary_rows(path)[0], dict(first, run_id='run_001'))
            self.assertEqual(path.read_bytes(), before)
            second = dict(first, run_id='run_002', completed_periods=20)
            gp.persist_summary_row(path, second)
            gp.persist_summary_row(path, second)
            rows = json.loads(path.read_text())
            self.assertEqual(rows, [dict(first, run_id='run_001'), second])
            gp.persist_summary_row(path, dict(second, record_id='other_record'))
            self.assertEqual(len(json.loads(path.read_text())), 3)
            self.assertEqual((path.parent/(gp.REPRESENTATIVE_ID+'_run_001.json')).read_bytes(), b'original pilot')

    def test_two_failed_offline_runs_persist_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            original_preflight = gp.preflight
            self.client.models.generate_content.side_effect = RuntimeError('offline transport failure')
            with patch.object(gp, 'preflight', side_effect=lambda **kwargs: original_preflight(output_root=directory, **kwargs)), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(gp.main([]), 1)
                first = Path(directory)/gp.EXPERIMENT_ID/(gp.REPRESENTATIVE_ID+'_run_001.json')
                first_bytes = first.read_bytes()
                self.assertEqual(gp.main(['--preflight']), 0)
                self.assertEqual(gp.main(['--preflight']), 0)
                self.assertIn('next_run_id: run_002', output.getvalue())
                self.assertEqual(gp.main([]), 1)
            self.assertEqual(first.read_bytes(), first_bytes)
            second = first.with_name(gp.REPRESENTATIVE_ID+'_run_002.json')
            result = json.loads(second.read_text())
            self.assertEqual(result['run_metadata']['run_id'], 'run_002')
            self.assertEqual(result['run_status'], 'failed')
            with self.assertRaises(FileExistsError):
                persist_run(result, gp.EXPERIMENT_ID, first.stem, output_root=directory)
            self.assertEqual(first.read_bytes(), first_bytes)
            rows = json.loads((first.parent/'summary.json').read_text())
            self.assertEqual([row['run_id'] for row in rows], ['run_001', 'run_002'])
            self.assertTrue(all(row['run_status'] == 'failed' for row in rows))
            self.assertEqual(self.client.models.generate_content.call_count, 6)
            self.assertEqual(gp.next_run_path(first.parent).name, gp.REPRESENTATIVE_ID+'_run_003.json')

    def test_summary_duplicate_identity_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'summary.json'
            row = {'record_id': gp.REPRESENTATIVE_ID, 'run_id': 'run_001'}
            path.write_text(json.dumps([row, row]))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                gp.persist_summary_row(path, dict(row, run_id='run_002'))
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'requires record_id and run_id'):
                gp.persist_summary_row(path, {'record_id': gp.REPRESENTATIVE_ID})
            self.assertEqual(path.read_bytes(), before)

    def test_frozen_prompt_hashes(self):
        import hashlib
        self.assertEqual(hashlib.sha256(gp.SYSTEM_PROMPT.encode()).hexdigest(),
                         '5703807663b806f9d648577a60650a452dc46ead75bf068dbaa654198537034b')
        self.assertEqual(hashlib.sha256(gp.NO_COMMUNICATION_PROMPT.encode()).hexdigest(),
                         '57cf6e52b6612077419a58d0721ecb6a44357c0f2a967bd2620c3e09caa51088')
        self.assertEqual(gp.PROVIDER_CONFIG['prompt_version'], 'gemini_no_communication_v7')

    def test_exported_prompt_document_matches_all_current_templates(self):
        from agents import gemini_local_communication as adjacent
        from agents import gemini_no_forecast as no_forecast
        documentation = (Path(__file__).resolve().parents[1]
                         / 'docs' / 'methodology_latest_full_zh.md').read_text(encoding='utf-8')
        variants = (
            ('No Communication With Forecast', gp.SYSTEM_PROMPT,
             gp.NO_COMMUNICATION_PROMPT, gp.PROVIDER_CONFIG['prompt_version']),
            ('Adjacent Communication With Forecast', adjacent.SYSTEM_PROMPT,
             gp.NO_COMMUNICATION_PROMPT.split('P6 Information and communication rule: local agent', 1)[0]
             + adjacent.LOCAL_COMMUNICATION_RULES, adjacent.PROMPT_VERSION),
            ('No Communication Without Forecast', gp.SYSTEM_PROMPT,
             no_forecast.NO_FORECAST_PROMPT, no_forecast.PROMPT_VERSION),
        )
        for title, system, template, version in variants:
            with self.subTest(variant=title):
                section = documentation.split(f'## {title}\n', 1)[1].split('\n## ', 1)[0]
                blocks = re.findall(r'```text\n(.*?)\n```', section, re.DOTALL)
                self.assertEqual(blocks, [system, template.replace('{{', '{').replace('}}', '}')])
                self.assertIn(f'Prompt version: `{version}`', section)
        self.assertNotIn('## Adjacent Information Sharing With Forecast', documentation)
        self.assertEqual(documentation.count('Source: `src/agents/gemini_local_communication.py`'), 1)
        self.assertIn('`{forecast_value_statement}`', documentation)
        self.assertNotIn('`{forecast_next_period}`', documentation)

    def test_dotenv_file_loaded_once(self):
        from dotenv import load_dotenv
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text('GEMINI_API_KEY=' + self.secret + '\n')
            with patch.dict(os.environ, {}, clear=True), patch.object(
                    gp, 'load_dotenv', side_effect=lambda: load_dotenv(path)) as loader:
                provider = gp.GeminiDecisionProvider(CoordinationMetrics())
                provider(0, self.observation, self.context)
                provider(0, self.observation, self.context)
                self.assertEqual(os.environ['GEMINI_API_KEY'], self.secret)
                loader.assert_called_once_with()

    def test_missing_key_fails_before_client_construction(self):
        for value in ('', '   '):
            with patch.dict(os.environ, {'GEMINI_API_KEY': value}, clear=True):
                self.client_factory.reset_mock()
                with self.assertRaisesRegex(ValueError, 'GEMINI_API_KEY is not configured'):
                    gp.GeminiDecisionProvider(self.metrics)
                self.client_factory.assert_not_called()

    def test_frozen_request_and_client_configuration(self):
        result = self.provider(0, self.observation, self.context)
        request = self.client.models.generate_content.call_args.kwargs
        self.assertEqual(request['model'], 'gemini-3.8-flash')
        self.assertEqual(request['config'].thinking_config.thinking_level.value, 'LOW')
        self.assertFalse(request['config'].thinking_config.include_thoughts)
        self.assertEqual(request['config'].temperature, 0.0)
        self.assertEqual(request['config'].response_mime_type, 'text/plain')
        self.assertIsNone(request['config'].response_schema)
        client_options = self.client_factory.call_args.kwargs
        self.assertEqual(client_options['http_options'].retry_options.attempts, 1)
        self.assertFalse(client_options['vertexai'])
        self.assertEqual(client_options['api_key'], self.secret)
        self.assertEqual(self.provider.metadata['history_policy'], 'stateless_per_period')
        self.assertEqual(self.provider.metadata['prompt_version'], 'gemini_no_communication_v7')
        self.assertEqual(self.provider.metadata['schema_version'], 'decision_action_v2')

        self.assertIsInstance(result, DecisionResult)
        self.assertEqual(result.proposed_order_quantity, 2)
        self.assertEqual(result.raw_output, response().text)
        self.assertIsNone(result.decision_summary)
        self.assertEqual(result.validation_status, 'unvalidated')

        counts = self.metrics.to_dict()
        self.assertEqual(counts['decision_llm_call_count'], 1)
        self.assertEqual(counts['decision_input_tokens'], 11)
        self.assertEqual(counts['decision_output_tokens'], 8)
        self.assertEqual(counts['decision_total_tokens'], 19)
        self.assertEqual(self.provider.metadata['usage_reports'][0]['thoughts_token_count'], 5)
        for key in ('coordination_llm_call_count', 'coordination_message_count',
                    'coordination_round_count', 'active_communication_links'):
            self.assertEqual(counts[key], 0)

    def test_all_prompt_variants_expose_own_upstream_backlog(self):
        from agents.gemini_local_communication import build_prompt as adjacent_prompt
        from agents.gemini_no_forecast import build_prompt as no_forecast_prompt
        for builder in (gp.build_prompt, adjacent_prompt, no_forecast_prompt):
            for stage in range(4):
                quantity = 0 if stage == 3 else 17
                history = dict(self.observation['recent_observations'][0], upstream_backlog=5)
                observation = dict(self.observation, upstream_backlog=quantity,
                                   latest_order=None if stage == 0 else 2,
                                   recent_observations=[history])
                context = adjacent_context(stage, self.context, observation['latest_order'])
                prompt = builder(stage, observation, context)
                self.assertEqual(local_from_prompt(prompt)['upstream_backlog'], quantity)
                self.assertIn('"upstream_backlog": 5', prompt)
                self.assertIn('not stock already in transit', prompt)

    def test_forecast_history_is_not_rendered_in_either_communication_prompt(self):
        from agents.gemini_local_communication import build_prompt as build_adjacent_prompt
        observation = dict(self.observation, next_period_demand_forecast=7.5)
        for builder in (gp.build_prompt, build_adjacent_prompt):
            for history in ([987654321]*26, []):
                observation['downstream_demand_history'] = history
                context = adjacent_context(0, self.context, observation['latest_order'])
                prompt = builder(0, observation, context)
                self.assertIn('\n7.5 units\n', prompt)
                self.assertNotIn('987654321', prompt)
                self.assertNotIn('downstream_demand_history', prompt)
                self.assertNotIn('Your downstream demand in up to 26 completed weeks', prompt)

    def test_prompt_objective_and_no_policy_formula(self):
        prompt = gp.SYSTEM_PROMPT + gp.build_prompt(0, self.observation, self.context)
        self.assertIn('end-of-week inventory\n+ end-of-week backlog\n+ 1 if order quantity > 0', prompt)
        self.assertIn('There is no variable ordering cost', prompt)
        self.assertIn('26-week simple moving average', prompt)
        self.assertNotIn('Linear Regression', prompt)
        self.assertIn('No forecast is available yet.', prompt)
        self.assertNotIn('unavailable units', prompt)
        self.assertIn('0 to 10', prompt)
        for forbidden in ('S - IP', 'base-stock', 'EOQ', 'reorder-point', '(s,S)', 'bullwhip', 'smooth orders'):
            self.assertNotIn(forbidden, prompt)
        for section in ('P1 Environment', 'P2 Objective function', 'P3 Current period information',
                        'P4 Recent observations', 'P5 Demand forecast',
                        'P6 Information and communication rule: local agent', 'P7 Decision'):
            self.assertIn(section, prompt)

    def test_all_prompt_variants_describe_timing_capacity_and_order_status(self):
        from agents.gemini_local_communication import build_prompt as adjacent_prompt
        from agents.gemini_no_forecast import build_prompt as no_forecast_prompt
        for builder in (gp.build_prompt, adjacent_prompt, no_forecast_prompt):
            for stage, name in enumerate(gp.STAGES):
                with self.subTest(builder=builder.__module__, stage=stage):
                    observation = dict(self.observation, latest_order=None if stage == 0 else 2)
                    context = adjacent_context(stage, self.context, observation['latest_order'])
                    prompt = builder(stage, observation, context)
                    self.assertIn(f'You are the {name}', prompt)
                    self.assertIn('end-of-week inventory\n+ end-of-week backlog', prompt)
                    self.assertIn('Costs are measured after fulfillment', prompt)
                    self.assertIn('The fixed ordering cost applies even if', prompt)
                    self.assertIn('The lead time starts when goods are dispatched', prompt)
                    self.assertIn('It limits only your new replenishment order, not your downstream fulfillment.', prompt)
                    self.assertIn('There is no separate production, handling or shipment capacity limit.', prompt)
                    self.assertIn('It does not yet include this week', prompt)
                    self.assertIn('They are not\nend-of-week outcomes.', prompt)
                    if stage == 0:
                        self.assertIn('This is not new demand to fulfill again.', prompt)
                    else:
                        self.assertIn('It has not yet been included in the observed backlog', prompt)
                        self.assertIn('fully fulfilled order.', prompt)
                    if builder is not adjacent_prompt:
                        self.assertIn('do not treat estimates', prompt)
                        self.assertNotIn('Do not infer or assume', prompt)

    def test_information_allowlist_and_direct_order_for_every_stage(self):
        hidden = dict(current_customer_demand=810001, future_demand=810002,
                      weekly_demand=[810003] * 277, representative_mean=810004,
                      adi=810005, cv2=810006, regime='hidden_regime',
                      future_sales=810007, future_shipments=810008,
                      other_inventory=810009, other_backlog=810010,
                      simulator_arrays=810011, s=810012, S=810013, calibration=810014)
        observation = {**self.observation, **hidden}
        context = {**self.context, **hidden}
        for stage in range(4):
            prompt = gp.build_prompt(stage, observation, context)
            clean = gp.build_prompt(stage, self.observation, self.context)
            self.assertEqual(prompt, clean)
            local = local_from_prompt(prompt)
            self.assertEqual(local, self.observation['recent_observations'][-1])
            altered = {**context, 'downstream_orders': dict(context['downstream_orders'])}
            for key in altered['downstream_orders']:
                if key != f'stage_{stage-1}':
                    altered['downstream_orders'][key] = 999999
            self.assertEqual(prompt, gp.build_prompt(stage, observation, altered))

    def test_malformed_response_recovers_with_same_prompt(self):
        self.client.models.generate_content.side_effect = [response(text='[[4]]'), response(4)]
        decision = self.provider(0, self.observation, self.context)
        self.assertEqual(decision.proposed_order_quantity, 4)
        calls = self.client.models.generate_content.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual([item['decision_attempt'] for item in self.provider.invocation_trace], [1, 2])
        self.assertEqual(self.provider.metadata['malformed_output_count'], 1)

    def test_transport_failure_recovers_with_same_prompt(self):
        self.client.models.generate_content.side_effect = [RuntimeError('offline disconnect'), response(5)]
        decision = self.provider(0, self.observation, self.context)
        self.assertEqual(decision.proposed_order_quantity, 5)
        self.assertEqual(self.client.models.generate_content.call_args_list[0],
                         self.client.models.generate_content.call_args_list[1])
        self.assertEqual(self.metrics.decision_llm_call_count, 2)
        self.assertEqual(self.provider.metadata['api_failure_count'], 1)

    def test_only_one_next_period_forecast_reaches_gemini(self):
        from simulation.config import make_m5_config
        from demand.m5_demand import get_representative
        record = get_representative(gp.REPRESENTATIVE_ID)
        config = make_m5_config(record)
        self.client.models.generate_content.return_value = response(0)
        result = run_episode(env_creator(config), self.provider, 'm5', config,
                             metrics=self.metrics, provider_metadata=self.provider.metadata)
        self.assertEqual(result['run_status'], 'success')
        for index, call in enumerate(self.client.models.generate_content.call_args_list):
            t = index // 4
            stage = index % 4
            prompt = call.kwargs['contents']
            forecast_section = prompt.split('P5 Demand forecast')[1].split('P6 Information')[0]
            completed = record['weekly_demand'][max(0, t-26):t] if stage == 0 else [0]*min(t, 26)
            forecast = sum(completed) / len(completed) if completed else None
            expected = 'No forecast is available yet.' if forecast is None else f'{forecast} units'
            self.assertIn(f'\n{expected}\n', forecast_section)
            self.assertNotIn('forecast', json.dumps(local_from_prompt(prompt)))
            self.assertEqual(result['action_trace'][index]['observation']['next_period_demand_forecast'], forecast)
            self.assertEqual(result['action_trace'][index]['observation']['downstream_demand_history'], completed)
            self.assertEqual(result['action_trace'][index]['observation']['forecast_target_period'], t+2)
            local = local_from_prompt(prompt)
            expected_order = (record['weekly_demand'][t-1] if t else None) if stage == 0 else 0
            self.assertEqual(local['latest_order'], expected_order)
            history_text = prompt.split('P4 Recent observations')[1].split('P5 Demand forecast')[0]
            history = json.JSONDecoder().raw_decode(history_text[history_text.index('['):])[0]
            self.assertEqual(len(history), min(t, 10))
            expected_history = [local_from_prompt(self.client.models.generate_content.call_args_list[
                earlier*4+stage].kwargs['contents']) for earlier in range(max(0, t-10), t)]
            self.assertEqual(history, expected_history)
            cost = sum(entry['operational_cost'] for entry in result['action_trace'][:index]
                       if entry['stage'] == stage)
            self.assertEqual(local['total_cost_so_far'], cost)
            self.assertIn(f"0 to {config['prod_capacities'][stage]}", prompt)
        observation = dict(self.observation, next_period_demand_forecast=7.5,
                           future_forecasts=[888888], current_demand=999999)
        context = dict(self.context, forecast_trajectory=[777777])
        prompt = gp.build_prompt(0, observation, context)
        self.assertIn('\n7.5 units\n', prompt)
        for forbidden in ('888888', '999999', '777777', 'future_forecasts', 'forecast_trajectory'):
            self.assertNotIn(forbidden, prompt)

    def test_partial_usage_is_not_fabricated(self):
        for usage, expected in ((None, (0, 0, 1)),
                               (SimpleNamespace(prompt_token_count=11), (0, 0, 1)),
                               (SimpleNamespace(prompt_token_count=11, candidates_token_count=3,
                                                thoughts_token_count=5), (11, 8, 0)),
                               (SimpleNamespace(prompt_token_count=11, total_token_count=14), (11, 3, 0))):
            self.provider.metrics = CoordinationMetrics()
            reply = response()
            reply.usage_metadata = usage
            self.client.models.generate_content.return_value = reply
            self.provider(0, self.observation, self.context)
            counts = self.provider.metrics.to_dict()
            self.assertEqual((counts['input_tokens'], counts['output_tokens'],
                              counts['usage_unavailable_call_count']), expected)

    def test_stateless_requests_reuse_client(self):
        self.provider(1, self.observation, self.context)
        self.provider(1, self.observation, self.context)
        calls = self.client.models.generate_content.call_args_list
        self.assertEqual(calls[0], calls[1])
        self.client_factory.assert_called_once()
        self.client.chats.create.assert_not_called()

    def test_real_sdk_serialization_with_offline_http_transport(self):
        import httpx
        from google.genai.client import Client
        requests = []

        def transport(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                'candidates': [{'content': {'role': 'model', 'parts': [{'text': response().text}]},
                                'finishReason': 'STOP'}],
                'usageMetadata': {'promptTokenCount': 11, 'candidatesTokenCount': 3,
                                  'thoughtsTokenCount': 5, 'totalTokenCount': 19}})

        def factory(**kwargs):
            kwargs['http_options'].client_args = {'transport': httpx.MockTransport(transport)}
            return Client(**kwargs)

        self.client_factory.side_effect = factory
        provider = gp.GeminiDecisionProvider(CoordinationMetrics())
        try:
            result = provider(0, self.observation, self.context)
            self.assertEqual(result.proposed_order_quantity, 2)
            self.assertEqual(len(requests), 1)
            generation = requests[0]['generationConfig']
            # SDK 2.24 serializes this nested protobuf field in snake_case.
            self.assertEqual(generation['thinkingConfig']['thinking_level'], 'LOW')
            self.assertEqual(generation['temperature'], 0)
            self.assertEqual(generation['responseMimeType'], 'text/plain')
            self.assertEqual(provider.metrics.to_dict()['total_tokens'], 19)
            self.assertNotIn(self.secret, json.dumps(requests))
        finally:
            provider.close()

    def test_real_sdk_transient_failure_has_bounded_retries(self):
        import httpx
        from google.genai.client import Client
        attempts = []

        def transport(request):
            attempts.append(1)
            return httpx.Response(503, json={'error': {'code': 503, 'message': 'unavailable'}})

        def factory(**kwargs):
            kwargs['http_options'].client_args = {'transport': httpx.MockTransport(transport)}
            return Client(**kwargs)

        self.client_factory.side_effect = factory
        provider = gp.GeminiDecisionProvider(CoordinationMetrics())
        try:
            with self.assertRaises(gp.GeminiProviderError):
                provider(0, self.observation, self.context)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(provider.metrics.to_dict()['decision_llm_call_count'], 3)
        finally:
            provider.close()

    def test_import_never_constructs_client(self):
        self.client_factory.reset_mock()
        # A separate module namespace avoids invalidating schema class identities.
        spec = importlib.util.spec_from_file_location('gemini_import_probe', gp.__file__)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.client_factory.assert_not_called()

    def test_preflight_destination_representative_and_no_call(self):
        with tempfile.TemporaryDirectory() as directory:
            provider, config, destination = gp.preflight(output_root=directory)
            self.assertEqual(config['_provenance']['representative_record_id'], gp.REPRESENTATIVE_ID)
            self.assertEqual(config['num_periods'], 130)
            self.assertEqual(destination, Path(directory) / gp.EXPERIMENT_ID / (gp.REPRESENTATIVE_ID + '_run_001.json'))
            self.assertEqual(destination.relative_to(directory).as_posix(),
                             'gemini_no_communication/FOODS_1_002_WI_1_evaluation_run_001.json')
            self.client.models.generate_content.assert_not_called()
            destination.write_text('{}')
            second, _, next_path = gp.preflight(output_root=directory)
            self.assertEqual(next_path.name, gp.REPRESENTATIVE_ID + '_run_002.json')
            second.close()
            provider.close()

        record_id = 'FOODS_1_019_CA_3_evaluation'
        with tempfile.TemporaryDirectory() as directory:
            provider, config, destination = gp.preflight(
                record_id=record_id, num_periods=104, output_root=directory)
            try:
                self.assertEqual(config['num_periods'], 104)
                self.assertEqual(config['_provenance']['representative_record_id'], record_id)
                self.assertEqual(config['_provenance']['regime'], 'erratic')
                self.assertEqual(destination.name, record_id + '_run_001.json')
                self.client.models.generate_content.assert_not_called()
            finally:
                provider.close()

    def test_preflight_reserves_summary_only_run_id(self):
        record_id = 'FOODS_1_019_CA_3_evaluation'
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / gp.EXPERIMENT_ID
            result_dir.mkdir()
            summary = result_dir / 'summary.json'
            summary.write_text(json.dumps([{'record_id': record_id, 'run_id': 'run_001'}]))
            provider, config, destination = gp.preflight(record_id=record_id, output_root=directory)
            try:
                self.assertEqual(config['num_periods'], 130)
                self.assertEqual(destination.name, record_id + '_run_002.json')
                self.assertEqual(len(json.loads(summary.read_text())), 1)
            finally:
                provider.close()

    def test_preflight_rejects_coordination_and_invalid_destination(self):
        with patch.dict(gp.PROVIDER_CONFIG, architecture='coordination'):
            with self.assertRaises(ValueError):
                gp.preflight()
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'file'
            file.write_text('occupied')
            with self.assertRaises(OSError):
                gp.preflight(output_root=file)
        self.client.models.generate_content.assert_not_called()

    def test_offline_full_episode_and_json_persistence(self):
        self.client.models.generate_content.return_value = response(0)
        result = run_episode(env_creator(env_configs['m5']), self.provider, 'm5', env_configs['m5'],
                             metrics=self.metrics, provider_metadata=self.provider.metadata)
        result['run_metadata']['provider'] = self.provider.metadata
        self.assertEqual(result['run_status'], 'success')
        self.assertEqual(len(result['action_trace']), 1108)
        self.assertEqual(result['coordination_metrics']['decision_llm_call_count'], 1108)
        self.assertEqual(result['coordination_metrics']['decision_total_tokens'], 1108 * 19)
        self.assertEqual(len(result['run_metadata']['provider']['usage_reports']), 1108)
        with tempfile.TemporaryDirectory() as directory:
            path = persist_run(result, gp.EXPERIMENT_ID, gp.REPRESENTATIVE_ID + '_run_001', output_root=directory)
            encoded = path.read_text()
            self.assertEqual(json.loads(encoded)['run_status'], 'success')
            self.assertNotIn(self.secret, encoded)

    def test_entrypoint_preflight_and_run(self):
        with tempfile.TemporaryDirectory() as directory:
            provider, config, destination = gp.preflight(output_root=directory)
            config["num_periods"] = 3
            config["_provenance"]["demand_window"] = "weeks_1_to_3"
            with patch.object(gp, 'preflight', return_value=(provider, config, destination)), redirect_stdout(io.StringIO()):
                self.assertEqual(gp.main(['--preflight']), 0)
                self.client.models.generate_content.assert_not_called()
                self.client.models.generate_content.return_value = response(0)
                self.assertEqual(gp.main([]), 0)
            saved = json.loads(destination.read_text())
            self.assertEqual(saved['run_status'], 'success')
            self.assertEqual(saved['run_metadata']['provider']['model'], 'gemini-3.8-flash')
            self.assertNotIn(self.secret, destination.read_text())
            summary = json.loads((destination.parent / 'summary.json').read_text())[0]
            self.assertEqual(summary['architecture'], 'no_communication')
            self.assertEqual(summary['prompt_version'], 'gemini_no_communication_v7')
            self.assertEqual(summary['api_invocation_count'], 12)
            self.assertEqual(summary['failed_invocation_count'], 0)
            self.assertEqual(summary['schema_version'], 'decision_action_v2')
            self.assertEqual(summary['thinking_tokens'], 12 * 5)
            self.assertEqual(summary['thinking_usage_unavailable_call_count'], 0)
            self.assertEqual(summary['output_tokens'], 12 * 8)
            self.assertEqual(summary['total_tokens'], 12 * 19)
            self.assertEqual((destination.parent / 'summary.json').relative_to(directory).as_posix(),
                             'gemini_no_communication/summary.json')
            self.assertFalse((Path(directory) / 'gemini_no_coordination').exists())
            self.assertEqual(set(saved['run_metadata']['provider']) &
                             {'architecture', 'provider', 'model', 'prompt_version', 'schema_version'},
                             {'architecture', 'provider', 'model', 'prompt_version', 'schema_version'})
            self.assertEqual(saved['run_metadata']['representative_record_id'], gp.REPRESENTATIVE_ID)
            self.assertEqual(saved['run_metadata']['regime'], 'smooth')
            self.assertEqual(len(saved['action_trace']), 12)
            self.assertIn('next_period_demand_forecast', saved['action_trace'][0]['observation'])
            self.assertIn('raw_output', saved['action_trace'][0])
            self.assertIn('validation_status', saved['action_trace'][0])
            self.assertIn('usage_reports', saved['run_metadata']['provider'])
            self.assertEqual(set(saved['evaluation_summary']['operational']),
                             {'holding_cost', 'backlog_cost', 'fixed_ordering_cost', 'global_operational_cost'})
            self.assertFalse(any('variable' in key for key in summary))
            self.assertEqual(summary['global_bullwhip'], saved['evaluation_summary']['global_bullwhip'])
            for stage, name in enumerate(gp.STAGES):
                self.assertEqual(summary[name.lower() + '_bullwhip'],
                                 saved['evaluation_summary']['bullwhip'][f'stage_{stage}'])
                self.assertEqual(summary[name.lower() + '_bullwhip_cv'],
                                 saved['evaluation_summary']['bullwhip_cv'][f'stage_{stage}'])
            for key in ('cumulative_fulfilled_fraction', 'average_inventory', 'average_backlog',
                        'backlog_frequency_pct', 'capacity_saturation_pct', 'strict_capacity_binding_pct',
                        'supply_shortage_pct', 'holding_cost', 'backlog_cost', 'fixed_ordering_cost',
                        'global_operational_cost', 'input_tokens', 'output_tokens', 'total_tokens',
                        'usage_unavailable_call_count', 'wall_clock_seconds'):
                self.assertIn(key, summary)

    def test_summary_shared_baseline_kpis_and_partial_thinking_usage(self):
        from policies.ss import run_sS_baseline
        from demand.m5_demand import get_representative
        # Exercise summary projection with real baseline KPIs and fake provenance.
        result = run_sS_baseline(get_representative(gp.REPRESENTATIVE_ID))
        result['run_metadata']['provider'] = self.provider.metadata
        summary = gp.build_result_summary(result)
        evaluation = result['evaluation_summary']
        for key, value in {**evaluation['operational'], **evaluation['service_inventory'],
                           'stage_metrics': evaluation['stage_metrics'],
                           'episode_reward': result['episode_reward']}.items():
            self.assertEqual(summary[key], value)
        import math
        self.assertEqual(summary['global_bullwhip'], math.prod(
            summary[name.lower() + '_bullwhip'] for name in gp.STAGES))
        for stage, name in enumerate(gp.STAGES):
            self.assertEqual(summary[name.lower() + '_bullwhip_cv'],
                             evaluation['bullwhip_cv'][f'stage_{stage}'])
        self.assertEqual(summary['thinking_tokens'], 0)
        self.assertEqual(summary['thinking_usage_unavailable_call_count'], 0)
        for reports, expected, missing in (([None], None, 1), ([0], 0, 0),
                                           ([5, None, 7], 12, 1)):
            result['run_metadata']['provider']['usage_reports'] = [
                {'thoughts_token_count': value} for value in reports]
            summary = gp.build_result_summary(result)
            self.assertEqual(summary['thinking_tokens'], expected)
            self.assertEqual(summary['thinking_usage_unavailable_call_count'], missing)

