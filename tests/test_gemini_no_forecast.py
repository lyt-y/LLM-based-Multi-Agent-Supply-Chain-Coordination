"""Offline checks for the no-communication, no-forecast experiment."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from evaluation.metrics import CoordinationMetrics
from simulation.config import make_m5_config
from simulation.env import env_creator
from simulation.experiment_interface import run_episode
from agents.gemini_no_forecast import GeminiNoForecastProvider, build_prompt
from agents.gemini_provider import build_prompt as build_original_prompt, REPRESENTATIVE_ID
from demand.m5_demand import get_representative


class NoForecastTests(unittest.TestCase):
    def test_prompt_only_removes_forecast_section(self):
        local = dict(inventory_level=4, backlog=1, upstream_backlog=6, latest_order=2,
                     in_transit_orders=[1, 0], total_cost_so_far=7)
        observation = dict(inventory=4, backlog=1, upstream_backlog=6, latest_order=2, deliveries=[1, 0],
                           total_cost_so_far=7, recent_observations=[local], prod_capacity=3,
                           next_period_demand_forecast='HIDDEN_FORECAST_SENTINEL',
                           downstream_demand_history=[])
        context = {'period': 2, 'stage': 1}
        prompt = build_prompt(1, observation, context)
        original = build_original_prompt(1, observation, context)
        self.assertEqual(prompt.split('P5 Information and communication rule:', 1)[0],
                         original.split('P5 Demand forecast', 1)[0])
        self.assertNotIn('HIDDEN_FORECAST_SENTINEL', prompt)
        self.assertNotIn('P5 Demand forecast', prompt)
        self.assertIn('Use only the information provided in P3-P4.', prompt)
        self.assertIn('P6 Decision', prompt)
        self.assertIn('No external demand forecast is supplied.', prompt)
        self.assertIn('There is no variable ordering cost.', prompt)
        self.assertIn('new replenishment order decided by your direct downstream stage', prompt)

    def test_episode_never_exposes_forecast_to_agent_or_trace(self):
        record = get_representative(REPRESENTATIVE_ID)
        config = make_m5_config(record)
        config['num_periods'] = 2
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'offline-test'}), \
             patch('agents.gemini_provider.genai.Client') as client_factory:
            client = client_factory.return_value
            client.models.generate_content.return_value = SimpleNamespace(
                text='[[[0]]]', usage_metadata=None)
            metrics = CoordinationMetrics()
            provider = GeminiNoForecastProvider(metrics)
            try:
                result = run_episode(env_creator(config), provider, 'm5', config,
                                     metrics=metrics, provider_metadata=provider.metadata)
                self.assertEqual(result['run_status'], 'success')
                self.assertEqual(result['completed_periods'], 2)
                self.assertEqual(client.models.generate_content.call_count, 8)
                for call in client.models.generate_content.call_args_list:
                    self.assertNotIn('P5 Demand forecast', call.kwargs['contents'])
                    self.assertNotIn('next period', call.kwargs['contents'].lower())
                for entry in result['action_trace']:
                    self.assertNotIn('next_period_demand_forecast', entry['observation'])
                for entry in provider.invocation_trace:
                    self.assertNotIn('next_period_demand_forecast', entry['observation'])
                self.assertEqual(provider.metadata['architecture'],
                                 'no_communication_without_forecast_input')
            finally:
                provider.close()


if __name__ == '__main__':
    unittest.main()
