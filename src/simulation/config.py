"""Dataset-specific environments calibrated on the first 130 demand periods."""
from collections.abc import Mapping

from demand.m5_demand import get_representative, m5_demand_fn
from demand.demand_data import demand_fn_from_sequence, initial_inventory, demand_capacity

# Set an exported record ID, or choose an ID-sorted position within a regime.
M5_RECORD_ID = None
M5_REGIME = 'smooth'
M5_REPRESENTATIVE_INDEX = 0
HOLDING_COST = 1
BACKLOG_COST = 1
FIXED_ORDER_COST = 1
ENVIRONMENT_CALIBRATION_WEEKS = 130


def make_demand_config(record, q=0.95, capacity_quantile=0.95, *, dataset_source, dataset_id):
    """Build the common four-stage environment from a frozen weekly series."""
    if dataset_source not in ('m5', 'synthetic'):
        raise ValueError('Expected dataset_source m5 or synthetic')
    if record.get('dataset_source', dataset_source) != dataset_source:
        raise ValueError('Demand record source does not match the requested environment')
    weekly = record['weekly_demand']
    calibration = weekly[:ENVIRONMENT_CALIBRATION_WEEKS]
    config = {
        'num_stages': 4,
        'num_periods': len(weekly),
        'init_inventories': [initial_inventory(calibration, q)] * 4,
        'lead_times': [2, 2, 2, 2],
        'demand_fn': demand_fn_from_sequence(weekly),
        'prod_capacities': [demand_capacity(calibration, capacity_quantile)] * 4,
        'fixed_order_costs': [FIXED_ORDER_COST] * 4,
        'backlog_costs': [BACKLOG_COST] * 4,
        'holding_costs': [HOLDING_COST] * 4,
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
    }
    config['_provenance'] = dict(dataset_source=dataset_source, dataset_id=dataset_id,
                                 series_id=record['id'], regime=record.get('regime'), q=q,
                                 source_num_periods=len(weekly),
                                 environment_calibration_weeks=len(calibration),
                                 calibration_window=f'weeks_1_to_{len(calibration)}',
                                 initial_inventory_quantile=q,
                                 initial_inventory_method='ceil_first_130_weeks_2week_rolling_demand_quantile_linear_min_1',
                                 initial_inventory_lead_time=2,
                                 initial_inventory_calibration_weeks=len(calibration),
                                 capacity_quantile=capacity_quantile,
                                 capacity_method='ceil_first_130_weeks_quantile_linear_min_1',
                                 capacity_calibration_weeks=len(calibration),
                                 capacity_scope='new_order_quantity_only',
                                 fulfillment_policy='min_pending_demand_and_opening_inventory')
    for key in ('input_file', 'input_sha256', 'seed', 'generation_parameters'):
        if key in record:
            config['_provenance'][key] = record[key]
    return config


def make_synthetic_config(record, q=0.95, capacity_quantile=0.95, *, dataset_id='bernoulli_gamma_v1'):
    """Create an environment from a loaded synthetic record."""
    return make_demand_config(record, q, capacity_quantile, dataset_source='synthetic', dataset_id=dataset_id)


def make_m5_config(record=None, q=0.95, capacity_quantile=0.95, *, record_id=None,
                   regime=M5_REGIME, index=M5_REPRESENTATIVE_INDEX):
    """Build fresh parameters for a selected record (or a selected record ID)."""
    if record is not None and record_id is not None:
        raise ValueError('Specify record or record_id, not both')
    if record is None:
        record = get_representative(record_id, regime, index)
    elif isinstance(record, str):
        record = get_representative(record)
    weekly = record['weekly_demand']
    m5_demand_fn(weekly)  # Keep the existing 277-week M5 input contract.
    config = make_demand_config(record, q, capacity_quantile, dataset_source='m5', dataset_id='m5')
    config['_provenance'].update(representative_record_id=record['id'],
                                regime=record.get('regime', regime))
    return config


class _EnvironmentConfigs(Mapping):
    """Legacy configuration mapping, loaded only when explicitly requested."""

    def __getitem__(self, name):
        if name != 'm5':
            raise KeyError(name)
        return make_m5_config(record_id=M5_RECORD_ID)

    def __iter__(self):
        return iter(('m5',))

    def __len__(self):
        return 1


env_configs = _EnvironmentConfigs()
