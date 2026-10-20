"""Stage-local forecasts from completed downstream demand only."""

import numpy as np

FORECAST_WINDOW = 26

STAGE_FORECAST_METADATA = {
    'forecast_method': 'simple_moving_average', 'window': 26,
    'forecast_horizon': 1, 'target': 'stage_specific_downstream_demand',
    'sources': ['customer_demand', 'retailer_orders', 'wholesaler_orders', 'distributor_orders'],
    'warmup_method': 'expanding_mean_of_completed_periods',
    'alignment': 'decision week t: predict week t+1 using completed weeks through t-1 only',
    'initial_forecast': None,
}

def next_period_demand_forecast(completed_demand):
    """Target t+1 at decision t; caller supplies only D[0:t], never D[t]."""
    history = completed_demand[-FORECAST_WINDOW:]
    return float(np.mean(history)) if len(history) else None

