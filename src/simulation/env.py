"""
Multi-Period Inventory Management Problem (IMP)

This file is the reimplementation of the IMP from the OR-Gym library, an open-source project developed to bring
reinforcement learning to the operations research community. OR-Gym is licensed under the MIT License. For more
information, please visit the OR-Gym GitHub repository: https://github.com/hubbs5/or-gym.
"""
from typing import Callable
from numbers import Integral

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from simulation.config import env_configs

np.random.seed(0)


class InventoryManagementEnv(MultiAgentEnv):
    """
    Inventory Management Environment

    A multi-period, multi-echelon production-inventory system for a single non-perishable product sold in discrete
    quantities. Each stage in the supply chain consists of an inventory holding area and a production area. The
    inventory holding area stores the materials necessary for production at that stage. One unit of inventory
    produces one unit of product at each stage. There are lead times for transferring products between stages. The
    outgoing material from stage i serves as the feed material for production at stage i-1. Stages are numbered in
    ascending order: Stages = {0, 1, ..., M-1}, with the zeroth stage being the retailer. Production at each stage is
    limited by available inventory. Capacity limits new orders only, not fulfillment.

    At the beginning of each time period, the following sequence of events occurs:

    1) Check deliveries: Each stage receives incoming inventory replenishment shipments that have arrived after the
       stage's respective lead time.
    2) Check orders and demands: Each stage places replenishment orders to their  respective suppliers. Replenishment
       orders are filled according to available inventory at the suppliers. Customer demand
       occurs at the retailer and is filled based on the available  inventory at the retailer.
    3) Deliver orders and demands: Each stage delivers as many products as possible to satisfy  downstream demand or
       replenishment orders. Unfulfilled sales and replenishment orders are backlogged, with backlogged sales taking
       priority in the following period.
    4) Compute normalized operational costs from end stock, backlog and positive replenishment orders.
    """

    def __init__(
        self, num_stages: int, num_periods: int, init_inventories: list, lead_times: list, demand_fn: Callable,
        prod_capacities: list, fixed_order_costs: list, backlog_costs: list, holding_costs: list,
        stage_names: list, init_seed: int = 0):
        """
        Initialize the inventory management environment

        :param num_stages: number of stages (M)
        :param num_periods: number of periods (N)
        :param init_inventories: initial inventory quantities (I0)
        :param lead_times: lead times (L)
        :param demand_fn: demand function (D)
        :param prod_capacities: per-stage new-order limits (c); retained legacy field name
        :param fixed_order_costs: cost per positive replenishment order (K)
        :param backlog_costs: unit backlog costs for unfulfilled orders (k)
        :param holding_costs: unit inventory holding costs (h)
        :param stage_names: stage names
        :param init_seed: initial seed
        """
        super().__init__()

        # Check the validity of inputs
        assert num_stages >= 2, "The number of stages should be at least 2."
        assert num_periods >= 1, "The number of periods should be at least 1."
        assert len(init_inventories) == num_stages, \
            "The number of initial inventories quantities should be the number of stages."
        assert min(init_inventories) >= 0, "The initial inventory quantities should be non-negative."
        assert len(lead_times) == num_stages, "The number of lead times should be the number of stages."
        assert min(lead_times) >= 0, "The lead times should be non-negative."
        assert len(prod_capacities) == num_stages, "The number of production capacities should be the number of stages."
        assert min(prod_capacities) > 0, "The production capacities should be positive."
        assert len(fixed_order_costs) == num_stages, "One fixed ordering coefficient per stage is required."
        assert min(fixed_order_costs) >= 0, "Fixed ordering coefficients must be non-negative."
        assert len(backlog_costs) == num_stages, \
            "The number of unit backlog costs for unfulfilled orders should be the number of stages."
        assert min(backlog_costs) >= 0, "The unit penalties for unfulfilled orders should be non-negative."
        assert len(holding_costs) == num_stages, \
            "The number of unit inventory holding costs should be the number of stages."
        assert min(holding_costs) >= 0, "The unit inventory holding costs should be non-negative."
        assert len(stage_names) == num_stages, "The number of stage names should be the number of stages."

        # Set the environment configurations
        self.num_stages = num_stages
        self.num_periods = num_periods
        self.stage_names = stage_names
        self.init_inventories = np.array(init_inventories, dtype=int)
        self.lead_times = np.array(lead_times, dtype=int)
        self.max_lead_time = np.max(self.lead_times)
        self.demand_fn = demand_fn
        self.prod_capacities = np.array(prod_capacities, dtype=int)
        self.max_production = np.max(self.prod_capacities)
        self.fixed_order_costs = np.array(fixed_order_costs, dtype=int)
        self.backlog_costs = np.array(backlog_costs, dtype=int)
        self.holding_costs = np.array(holding_costs, dtype=int)

        # Create all variables
        self.period = 0
        self.inventories = np.zeros((self.num_stages, self.num_periods + 1), dtype=int)
        self.orders = np.zeros((self.num_stages, self.num_periods + 1), dtype=int)
        self.arriving_orders = np.zeros((self.num_stages, self.num_periods + 1), dtype=int)
        self.sales = np.zeros((self.num_stages, self.num_periods + 1), dtype=int)
        self.backlogs = np.zeros((self.num_stages, self.num_periods + 1), dtype=int)
        self.demands = np.zeros(self.num_periods + 1, dtype=int)
        self.reward_history = np.zeros((self.num_stages, self.num_periods + 1), dtype=int)
        self.total_rewards = np.zeros(self.num_periods + 1, dtype=int)
        self.period_costs = {
            name: np.zeros((self.num_stages, self.num_periods + 1), dtype=int)
            for name in ('holding_cost', 'backlog_cost', 'fixed_ordering_cost', 'operational_cost')
        }

        # Compute the upper bounds for state variables
        max_production = self.max_production
        max_fixed_order_cost = np.max(self.fixed_order_costs)
        max_backlog_cost = np.max(self.backlog_costs)
        max_holding_cost = np.max(self.holding_costs)
        max_lead_time = self.max_lead_time
        max_order = max_production
        max_inventory = int(np.max(self.init_inventories)) + max_order * self.num_periods

        # Set the observation and action spaces
        max_observations = np.array(
            [max_production + 1, max_fixed_order_cost + 1, max_backlog_cost + 1, max_holding_cost + 1,
             max_lead_time + 1, max_inventory + 1, np.iinfo(np.int64).max, np.iinfo(np.int64).max]
            + [max_inventory + 1] * (2 * self.max_lead_time), dtype=np.int64)
        self.agent_observation_space = spaces.MultiDiscrete(nvec=max_observations, seed=init_seed)
        self.agent_action_space = spaces.Discrete(n=max_order + 1, start=0, seed=init_seed)
        self.observation_space = spaces.Dict({
            f"stage_{stage}": spaces.MultiDiscrete(nvec=max_observations, seed=init_seed + stage)
            for stage in range(self.num_stages)})
        self.action_space = spaces.Dict({
            f"stage_{stage}": spaces.Discrete(n=int(self.prod_capacities[stage]) + 1, start=0, seed=init_seed + stage)
            for stage in range(self.num_stages)})
        self.state_dict = {f"stage_{m}": None for m in range(self.num_stages)}

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        """
        Reset the environment variables including:
            - I: inventory at each stage
            - O: order placed by each stage
            - R: arriving order for each stage
            - S: sales by each stage
            - B: backlog for each stage
            - D: customer demand at the retailer
            - rewards and period cost components at each stage

        :param seed: seed for the new episode
        :param options: options
        :return: states, infos
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset all variables
        self.period = 0
        self.inventories.fill(0)
        self.orders.fill(0)
        self.arriving_orders.fill(0)
        self.sales.fill(0)
        self.backlogs.fill(0)
        self.demands.fill(0)
        self.reward_history.fill(0)
        self.total_rewards.fill(0)
        for values in self.period_costs.values():
            values.fill(0)

        # Set the initial condition and state
        self.inventories[:, 0] = self.init_inventories
        self.update_state()

        return self.state_dict, {}

    def update_state(self) -> None:
        """
        Build next week's decision observation after its opening receipts.
        Inventory history remains end-of-week stock. Deliveries contain only
        future receipts, one slot per week after the decision week.
        """
        t = self.period
        opening_inventory = (self._opening_inventory(t + 1) if t < self.num_periods
                             else self.inventories[:, t].copy())
        states = np.zeros((self.num_stages, 8 + 2 * self.max_lead_time), dtype=int)
        states[:, :7] = np.stack([
            self.prod_capacities, self.fixed_order_costs, self.backlog_costs, self.holding_costs,
            self.lead_times, opening_inventory, self.backlogs[:, t]], axis=-1)
        states[:-1, 7] = self.backlogs[1:, t]

        lt_max = self.max_lead_time
        if t >= lt_max:
            states[:, (-2 * lt_max):-lt_max] = self.sales[:, (t - lt_max + 1):(t + 1)]
        elif t > 0:
            states[:, (-lt_max - t):-lt_max] = self.sales[:, 1:(t + 1)]

        for m in range(self.num_stages):
            for offset in range(1, self.max_lead_time + 1):
                dispatched_week = t + 1 + offset - self.lead_times[m]
                if t < self.num_periods and 1 <= dispatched_week <= t:
                    states[m, -self.max_lead_time + offset - 1] = self.arriving_orders[m, dispatched_week]

        self.state_dict = {f"stage_{m}": states[m] for m in range(self.num_stages)}

    def _opening_inventory(self, period: int) -> np.ndarray:
        """Project committed receipts once without mutating end-stock history."""
        inventory = self.inventories[:, period - 1].copy()
        for stage, lead_time in enumerate(self.lead_times):
            dispatched_week = period - lead_time
            if 1 <= dispatched_week < period:
                inventory[stage] += self.arriving_orders[stage, dispatched_week]
        return inventory

    def step(self, action_dict: dict[str, int]) -> tuple[dict, dict, dict, dict, dict]:
        """
        Take a step and return the next observation

        :param action_dict: action (order quantity) for each stage
        :return: states, rewards, terminations, truncations, infos
        """
        assert all(f"stage_{m}" in action_dict for m in range(self.num_stages)), \
            "Order quantities for all stages are required."
        assert all(isinstance(action_dict[f"stage_{m}"], Integral)
                   and not isinstance(action_dict[f"stage_{m}"], (bool, np.bool_))
                   and 0 <= action_dict[f"stage_{m}"] <= self.prod_capacities[m]
                   for m in range(self.num_stages)), \
            "Order quantities must be integers within each stage's order limit."

        # Get the inventory at the beginning of the period
        self.period += 1
        t = self.period
        M = self.num_stages
        current_inventories = self._opening_inventory(t)
        self.orders[:, t] = np.array([action_dict[f"stage_{m}"] for m in range(self.num_stages)])
        self.demands[t] = int(self.demand_fn(t))

        # Compute the fulfilled orders
        # Dispatch can exceed the new-order limit when clearing old backlog.
        self.arriving_orders[:-1, t] = np.minimum(
            self.backlogs[1:, t - 1] + self.orders[:-1, t], current_inventories[1:])
        # R_{M-1,t} = O_{M-1,t}
        self.arriving_orders[M - 1, t] = self.orders[M - 1, t]

        # Compute the sales
        # S_{m,t} = R_{m-1,t}, m = 1, ..., M - 1
        self.sales[1:, t] = self.arriving_orders[:-1, t]
        # Customer fulfillment is inventory-limited, not order-capacity-limited.
        self.sales[0, t] = min(self.backlogs[0, t - 1] + self.demands[t], current_inventories[0])

        # Compute the backlogs
        # B_{m,t} = B_{m,t-1} + O_{m-1,t} - S_{m,t}, m = 1, ..., M - 1
        self.backlogs[1:, t] = self.backlogs[1:, t - 1] + self.orders[:-1, t] - self.sales[1:, t]
        # B_{0,t} = B_{0,t-1} + D_{t} - S_{0,t}
        self.backlogs[0, t] = self.backlogs[0, t - 1] + self.demands[t] - self.sales[0, t]

        # Compute the inventory at the end of the period
        # I_{m,t} = I_{m,t-1} + R_{m,t-L_m} - S_{m,t} (after sales)
        self.inventories[:, t] = current_inventories - self.sales[:, t]

        # Snapshot end-of-week costs independently of decision observations.
        self.period_costs['holding_cost'][:, t] = self.holding_costs * self.inventories[:, t]
        self.period_costs['backlog_cost'][:, t] = self.backlog_costs * self.backlogs[:, t]
        self.period_costs['fixed_ordering_cost'][:, t] = self.fixed_order_costs * (self.orders[:, t] > 0)
        self.period_costs['operational_cost'][:, t] = (
            self.period_costs['holding_cost'][:, t]
            + self.period_costs['backlog_cost'][:, t]
            + self.period_costs['fixed_ordering_cost'][:, t])
        self.reward_history[:, t] = -self.period_costs['operational_cost'][:, t]
        self.total_rewards[t] = np.sum(self.reward_history[:, t])

        # Determine rewards and terminations
        rewards = {f"stage_{m}": self.reward_history[m, t] for m in range(self.num_stages)}
        all_termination = self.period >= self.num_periods
        terminations = {f"stage_{m}": all_termination for m in range(self.num_stages)}
        terminations["__all__"] = all_termination
        truncations = {f"stage_{m}": False for m in range(self.num_stages)}
        truncations["__all__"] = False
        infos = {
            f"stage_{m}": {**{name: int(values[m, t]) for name, values in self.period_costs.items()},
                           'reward': int(self.reward_history[m, t])}
            for m in range(self.num_stages)
        }

        # Update the state
        self.update_state()

        return self.state_dict, rewards, terminations, truncations, infos

    def _parse_state(self, state: list) -> dict:
        """
        Parse a single stage state

        :param state: state
        :return: parsed state
        """
        lt_max = self.max_lead_time
        return {
            'prod_capacity': state[0],
            'fixed_order_cost': state[1],
            'backlog_cost': state[2],
            'holding_cost': state[3],
            'lead_time': state[4],
            'inventory': state[5],
            'backlog': state[6],
            'upstream_backlog': state[7],
            'sales': state[(-2 * lt_max):(-lt_max)].tolist(),
            'deliveries': state[-lt_max:].tolist(),
        }

    def parse_state(self, state_dict: dict = None) -> dict:
        """
        Parse the state dictionary

        :param state_dict: state dictionary
        :return: parsed state dict
        """
        if state_dict is None:
            state_dict = self.state_dict

        parsed_state = {}

        for stage_id_name, state in state_dict.items():
            parsed_state[stage_id_name] = self._parse_state(state)

        return parsed_state


def env_creator(env_config):
    """
    Create the environment
    """
    if env_config is None:
        env_config = env_configs['m5']

    return InventoryManagementEnv(
        num_stages=env_config['num_stages'],
        num_periods=env_config['num_periods'],
        init_inventories=env_config['init_inventories'],
        lead_times=env_config['lead_times'],
        demand_fn=env_config['demand_fn'],
        prod_capacities=env_config['prod_capacities'],
        fixed_order_costs=env_config['fixed_order_costs'],
        backlog_costs=env_config['backlog_costs'],
        holding_costs=env_config['holding_costs'],
        stage_names=env_config['stage_names'],
    )


if __name__ == '__main__':
    im_env = env_creator(None)
    im_env.reset()
    print(f"stage_names = {im_env.stage_names}")
    print(f"state_dict = {im_env.state_dict}")
    print(f"state_dict = {im_env.parse_state(im_env.state_dict)}")
    print(f"observation_space = {im_env.observation_space}")
    print(f"observation_sample = {im_env.observation_space.sample()}")
    print(f"action_space = {im_env.action_space}")
    print(f"action_sample = {im_env.action_space.sample()}")

    for t in range(im_env.num_periods):
        next_state_dict, rewards, terminations, truncations, infos = im_env.step(
            action_dict={f"stage_{m}": 4 for m in range(im_env.num_stages)})
        print('-' * 80)
        print(f"period = {t}")
        print(f"next_state_dict = {next_state_dict}")
        print(f"next_state_dict = {im_env.parse_state(next_state_dict)}")
        print(f"rewards = {rewards}")
        print(f"terminations = {terminations}")
        print(f"truncations = {truncations}")
        print(f"infos = {infos}")
