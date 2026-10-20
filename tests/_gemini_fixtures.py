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
def local_from_prompt(prompt):
    return json.JSONDecoder().raw_decode(prompt.split('Your current observations are:\n\n')[1])[0]


def adjacent_context(stage, base, order):
    snapshot = dict(inventory_level=20, backlog=3, in_transit_orders=[4, 5])
    return dict(base, upstream_message=None if stage == 3 else dict(snapshot),
                downstream_opening_message=None if stage == 0 else dict(snapshot),
                downstream_message=None if stage == 0 else dict(
                    current_order_to_upstream=order, planning_insight='Maintain replenishment.'))


def response(quantity=2, usage=True, text=None):
    return SimpleNamespace(
        text=text if text is not None else f'[[[{quantity}]]]',
        usage_metadata=SimpleNamespace(prompt_token_count=11, candidates_token_count=3,
                                       thoughts_token_count=5, total_token_count=19) if usage else None)


class GeminiFixture(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        # Generated test-only value; no key or sample token is stored in fixtures.
        self.secret = uuid.uuid4().hex
        # Retain Windows/Conda DLL paths required by the real SDK's SSL setup.
        stack.enter_context(patch.dict(os.environ, {'GEMINI_API_KEY': self.secret}))
        self.load = stack.enter_context(patch.object(gp, 'load_dotenv'))
        stack.enter_context(patch.object(gp.time, 'sleep'))
        self.client_factory = stack.enter_context(patch.object(gp.genai, 'Client'))
        self.client = self.client_factory.return_value
        self.client.models.generate_content.return_value = response()
        self.metrics = CoordinationMetrics()
        self.provider = gp.GeminiDecisionProvider(self.metrics)
        self.observation = dict(inventory=4, backlog=3, upstream_backlog=2,
                                lead_time=2, prod_capacity=10, sales=[1, 2], deliveries=[3, 4],
                                latest_order=None, total_cost_so_far=0,
                                next_period_demand_forecast=None, downstream_demand_history=[])
        self.observation['recent_observations'] = [dict(
            inventory_level=4, backlog=3, upstream_backlog=2, latest_order=None,
            in_transit_orders=[3, 4], total_cost_so_far=0)]
        self.context = dict(period=1, downstream_orders={'stage_0': 1, 'stage_1': 6, 'stage_2': 9})
