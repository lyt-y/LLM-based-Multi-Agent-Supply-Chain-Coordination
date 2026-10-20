"""Full-trajectory M5 preprocessing. Run this file explicitly to write artifacts."""

from project_paths import M5_ARTIFACTS, PROJECT_ROOT
import argparse
from collections import Counter
import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import scipy
from demand.forecast import next_period_demand_forecast
from scipy.spatial.distance import cdist

DATA_PATH = PROJECT_ROOT.parent / 'Data' / 'sales_train_evaluation.csv'
ARTIFACT_PATH = M5_ARTIFACTS
NUM_WEEKS = 277
NUM_DAYS = 1941
REGIMES = ('smooth', 'erratic', 'intermittent', 'lumpy')
FEATURES = ('log1p_mean_weekly_demand', 'adi', 'positive_cv2', 'lag1_autocorrelation')
FORECAST_METADATA = {
    'forecast_method': 'simple_moving_average',
    'window': 26,
    'forecast_horizon': 1,
    'target': 'next_period_demand',
    'warmup_method': 'expanding_mean_of_all_realized_observations_so_far',
    'alignment': 'entry t: zero-based decision t, target t+1, completed demand D[0:t] only',
    'initial_forecast': None,
}


def build_demand_forecasts(weekly_demand):
    """One value per decision; the last value targets the period beyond the run."""
    return [next_period_demand_forecast(weekly_demand[:t])
            for t in range(len(weekly_demand))]


def persist_demand_forecasts(output_dir=ARTIFACT_PATH):
    """Export forecasts from frozen representatives without redoing selection."""
    source = ARTIFACT_PATH / 'representatives.json'
    representatives = load_representatives(source)
    records = {record['id']: build_demand_forecasts(record['weekly_demand'])
               for group in representatives.values() for record in group}
    artifact = dict(metadata=dict(FORECAST_METADATA), forecasts=records)
    manifest = dict(FORECAST_METADATA,
                    source_file='representatives.json',
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    artifact_file='demand_forecasts_26.json',
                    num_records=len(records), num_decision_periods=NUM_WEEKS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in (('demand_forecasts_26.json', artifact),
                              ('forecast_manifest.json', manifest)):
        (output_dir / filename).write_text(
            json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return artifact, manifest


def read_m5_records(data_path=DATA_PATH):
    with Path(data_path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        required = {'id', 'item_id', 'store_id'} | {f'd_{i}' for i in range(1, 1942)}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Expected M5 identifiers and d_1 through d_1941')
        yield from reader


def aggregate_weekly(row):
    """Sum days 1..1939 in disjoint blocks; discard days 1940 and 1941."""
    daily = np.array([int(row[f'd_{i}']) for i in range(1, 1942)], dtype=np.int64)
    if np.any(daily < 0):
        raise ValueError('M5 demand must be nonnegative')
    return daily[:1939].reshape(NUM_WEEKS, 7).sum(axis=1).tolist()


def load_m5_weekly_demand(record_id, data_path=DATA_PATH):
    for row in read_m5_records(data_path):
        if row['id'] == record_id:
            return aggregate_weekly(row)
    raise KeyError(f'M5 record not found: {record_id}')


def classify_regime(adi, positive_cv2):
    if adi <= 1.32:
        return 'smooth' if positive_cv2 <= 0.49 else 'erratic'
    return 'intermittent' if positive_cv2 <= 0.49 else 'lumpy'


def characterize(weekly_demand):
    """Full-sequence statistics, or a deterministic exclusion reason."""
    x = np.asarray(weekly_demand, dtype=np.float64)
    if x.shape != (NUM_WEEKS,) or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Expected 277 finite nonnegative observations')
    positive = x[x > 0]
    if positive.size < 2:
        return {'eligible': False, 'reason': 'fewer_than_two_positive_weeks'}
    adi = NUM_WEEKS / positive.size
    cv2 = float((positive.std(ddof=1) / positive.mean()) ** 2)
    stats = dict(mean_weekly_demand=float(x.mean()), adi=adi,
                 positive_cv2=cv2, regime=classify_regime(adi, cv2))
    left, right = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return dict(stats, eligible=False, reason='undefined_lag1_autocorrelation')
    stats['lag1_autocorrelation'] = float(np.dot(left, right) / denominator)
    return dict(stats, eligible=True)


def robust_scale(features):
    median = np.median(features, axis=0)
    q25, q75 = np.percentile(features, [25, 75], axis=0, method='linear')
    scale = q75 - q25
    scale[scale == 0] = 1.0
    return (features - median) / scale, median, scale


def alternating_kmedoids(z, k=30, max_iter=300, block_size=256):
    """Alternating Euclidean k-medoids, not PAM or a global optimum.

    Input rows are ID-sorted. Ties use row order; update ties retain the
    incumbent. Medoids own themselves even when feature vectors duplicate.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or not np.isfinite(z).all() or not 1 <= k <= len(z):
        raise ValueError('Expected finite features and at least k records')
    first = int(np.argmin(cdist(z, np.median(z, axis=0)[None, :])[:, 0]))
    medoids = [first]
    nearest = cdist(z, z[[first]])[:, 0]
    while len(medoids) < k:
        nearest[medoids] = -np.inf
        chosen = int(np.argmax(nearest))
        medoids.append(chosen)
        nearest = np.minimum(nearest, cdist(z, z[[chosen]])[:, 0])
    medoids = np.sort(medoids)
    history = []
    for iteration in range(1, max_iter + 1):
        distances = cdist(z, z[medoids], metric='euclidean')
        labels = distances.argmin(axis=1)
        labels[medoids] = np.arange(k)
        history.append(float(distances.min(axis=1).sum()))
        updated = []
        for cluster, incumbent in enumerate(medoids):
            members = np.flatnonzero(labels == cluster)
            costs = np.empty(len(members))
            for start in range(0, len(members), block_size):
                candidates = members[start:start + block_size]
                costs[start:start + block_size] = cdist(z[candidates], z[members]).sum(axis=1)
            best = int(costs.argmin())
            current = int(np.searchsorted(members, incumbent))
            updated.append(int(incumbent if costs[current] <= costs[best] else members[best]))
        updated = np.sort(updated)
        if np.array_equal(updated, medoids):
            return medoids, dict(converged=True, iterations=iteration,
                                 objective=history[-1], objective_history=history)
        medoids = updated
    raise RuntimeError(f'k-medoids did not converge in {max_iter} iterations')


def select_representatives(records, k=30):
    representatives, details = {}, {}
    for regime in REGIMES:
        group = sorted((r for r in records if r['regime'] == regime), key=lambda r: r['id'])
        if len(group) < k:
            raise ValueError(f'{regime}: only {len(group)} eligible records; need {k}')
        features = np.array([[np.log1p(r['mean_weekly_demand']), r['adi'],
                              r['positive_cv2'], r['lag1_autocorrelation']] for r in group])
        z, median, scale = robust_scale(features)
        indices, convergence = alternating_kmedoids(z, k=k)
        representatives[regime] = [group[i] for i in indices]
        details[regime] = dict(convergence, feature_medians=median.tolist(), feature_scales=scale.tolist())
        print(f'{regime}: {len(group)} eligible, {k} selected, {convergence["iterations"]} iterations', flush=True)
    return representatives, details


def preprocess(data_path=DATA_PATH, output_dir=ARTIFACT_PATH):
    records, excluded, classified = [], Counter(), Counter()
    total, seen = 0, set()
    for row in read_m5_records(data_path):
        total += 1
        if row['id'] in seen:
            raise ValueError(f'Duplicate source ID: {row["id"]}')
        seen.add(row['id'])
        weekly = aggregate_weekly(row)
        stats = characterize(weekly)
        if 'regime' in stats:
            classified[stats['regime']] += 1
        if not stats.pop('eligible'):
            excluded[stats['reason']] += 1
            continue
        records.append(dict(id=row['id'], item_id=row['item_id'], store_id=row['store_id'],
                            weekly_demand=weekly, **stats))
    print(f'Processed {total}; eligible {len(records)}; excluded {sum(excluded.values())}', flush=True)
    representatives, details = select_representatives(records)
    digest = hashlib.sha256()
    with Path(data_path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    manifest = {
        'source_file': str(Path(data_path).resolve()), 'source_sha256': digest.hexdigest(),
        'aggregation_rule': 'Sum consecutive non-overlapping 7-day blocks d_1..d_1939; discard d_1940,d_1941',
        'num_weeks': NUM_WEEKS, 'statistics_window': 'all 277 weeks; no split',
        'positive_cv2_ddof': 1, 'lag1': 'Pearson correlation of weeks 1..276 and 2..277, including zeros',
        'regime_thresholds': {'adi': 1.32, 'positive_cv2': 0.49, 'boundary': 'low side uses <='},
        'source_records': total, 'eligible_records': len(records), 'excluded_records': sum(excluded.values()),
        'exclusion_reasons': dict(excluded),
        'classified_regime_counts': {r: classified[r] for r in REGIMES},
        'eligible_regime_counts': {r: sum(x['regime'] == r for x in records) for r in REGIMES},
        'kmedoids': {'algorithm': 'deterministic alternating k-medoids', 'metric': 'euclidean', 'k': 30,
                     'initialization': 'nearest to componentwise median, then farthest-first',
                     'ties': 'sorted ID; incumbent retained on update ties; medoids own themselves',
                     'max_iter': 300, 'block_size': 256, 'features': FEATURES,
                     'scaling': 'within-regime median/IQR; linear percentiles; zero IQR replaced by 1',
                     'regimes': details},
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
        'selected_record_ids': {r: [x['id'] for x in representatives[r]] for r in REGIMES},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in (('representatives.json', representatives), ('selection_manifest.json', manifest)):
        (output_dir / filename).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return representatives, manifest


def load_representatives(path=ARTIFACT_PATH / 'representatives.json'):
    source = Path(path)
    payload = source.read_bytes()
    records = json.loads(payload)
    provenance = dict(dataset_source='m5', input_file=str(source.resolve()),
                      input_sha256=hashlib.sha256(payload).hexdigest())
    for group in records.values():
        for record in group:
            record.update(provenance)
    return records


def get_representative(record_id=None, regime='smooth', index=0, path=ARTIFACT_PATH / 'representatives.json'):
    """Look up a selected ID, or its ID-sorted position within a regime."""
    representatives = load_representatives(path)
    if record_id is None:
        return representatives[regime][index]
    for group in representatives.values():
        for record in group:
            if record['id'] == record_id:
                return record
    raise KeyError(f'Selected M5 record not found: {record_id}')


def m5_demand_fn(weekly_demand):
    values = tuple(weekly_demand)
    if len(values) != NUM_WEEKS or any(not isinstance(x, Integral) or isinstance(x, bool) or x < 0 for x in values):
        raise ValueError('Expected 277 nonnegative integer weekly observations')

    def demand_fn(period):
        if not isinstance(period, Integral) or isinstance(period, bool) or not 1 <= period <= NUM_WEEKS:
            raise IndexError('Demand period must be an integer between 1 and 277')
        return int(values[period - 1])

    return demand_fn


def empirical_lead_time_inventory(weekly_demand, q=0.95, lead_time=2):
    """Initial stock from overlapping lead-time sums, using linear quantiles."""
    m5_demand_fn(weekly_demand)  # Validate the unchanged 277 integer observations.
    if not 0 <= q <= 1:
        raise ValueError('q must be between 0 and 1')
    if not isinstance(lead_time, Integral) or isinstance(lead_time, bool) or not 1 <= lead_time <= NUM_WEEKS:
        raise ValueError('lead_time must be an integer between 1 and 277')
    lead_time_demand = np.convolve(weekly_demand, np.ones(lead_time, dtype=np.int64), mode='valid')
    return max(1, int(np.ceil(np.quantile(lead_time_demand, q, method='linear'))))


def scaled_capacity(weekly_demand, capacity_quantile=0.95):
    """Equal per-stage order limits from the full weekly-demand quantile."""
    m5_demand_fn(weekly_demand)
    from demand.demand_data import demand_capacity
    return demand_capacity(weekly_demand, capacity_quantile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, default=DATA_PATH)
    parser.add_argument('--output-dir', type=Path, default=ARTIFACT_PATH)
    parser.add_argument('--forecasts-only', action='store_true',
                        help='Export forecasts from frozen representatives; do not rerun preprocessing')
    args = parser.parse_args()
    if args.forecasts_only:
        persist_demand_forecasts(args.output_dir)
    else:
        preprocess(args.data_path, args.output_dir)
