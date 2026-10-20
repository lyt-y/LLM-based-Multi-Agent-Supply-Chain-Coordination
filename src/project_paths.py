"""Dataset-specific input and experiment output locations."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / 'artifacts'
RESULTS = PROJECT_ROOT / 'results'
M5_ARTIFACTS = ARTIFACTS / 'm5'
M5_RESULTS = RESULTS / 'm5'
M5_COMPARISONS = M5_RESULTS / 'comparisons'
SYNTHETIC_ARTIFACTS = ARTIFACTS / 'synthetic' / 'bernoulli_gamma_v1'
SYNTHETIC_RESULTS = RESULTS / 'synthetic' / 'bernoulli_gamma_v1'
