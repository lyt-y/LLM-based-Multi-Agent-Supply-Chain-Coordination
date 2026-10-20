"""Shared validation and calibration for frozen integer demand sequences."""

from numbers import Integral
import numpy as np


def demand_fn_from_sequence(demand):
    values = tuple(demand)
    if not values or any(not isinstance(x, Integral) or isinstance(x, bool) or x < 0
                         for x in values):
        raise ValueError('Expected a nonempty nonnegative integer demand sequence')

    def demand_fn(period):
        if not isinstance(period, Integral) or isinstance(period, bool) or not 1 <= period <= len(values):
            raise IndexError(f'Demand period must be between 1 and {len(values)}')
        return int(values[period - 1])

    return demand_fn


def initial_inventory(demand, q=0.95, lead_time=2):
    demand_fn_from_sequence(demand)
    if not 0 <= q <= 1 or not isinstance(lead_time, Integral) or isinstance(lead_time, bool) or not 1 <= lead_time <= len(demand):
        raise ValueError('Invalid quantile or lead time')
    sums = np.convolve(demand, np.ones(lead_time, dtype=np.int64), mode='valid')
    return max(1, int(np.ceil(np.quantile(sums, q, method='linear'))))


def demand_capacity(demand, capacity_quantile=0.95):
    """Integer order limit from the full sequence, including zero-demand weeks."""
    demand_fn_from_sequence(demand)
    if not 0 <= capacity_quantile <= 1:
        raise ValueError('capacity_quantile must be between 0 and 1')
    return max(1, int(np.ceil(np.quantile(demand, capacity_quantile, method='linear'))))
