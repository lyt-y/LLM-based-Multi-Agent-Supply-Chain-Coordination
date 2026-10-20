# Supply-chain coordination: Project Study

A four-stage simulator compares s,S replenishment with Gemini agents on frozen M5
and Bernoulli-Gamma demand. Scientific source, demand inputs and KPI definitions
are unchanged.

- `src/`: simulator, demand preparation, policies, agents and evaluation.
- `scripts/`: preparation, execution, recovery and figure tools.
- `artifacts/`: frozen demand, generation/selection definitions and current calibration metadata.
- [results/summary.json](results/summary.json): individual valid completed runs.
- [results/manifest.json](results/manifest.json): stable run IDs, configurations, seeds, status, hashes and provenance.
- `results/figures/`: four report figures.
- `tests/`: scientific, agent, execution/recovery and publication-integrity regressions.
- [Methodology](docs/methodology_latest_full_zh.md): definitions, limitations and exact current prompts.

## Updated settings and evaluated scope

Current calibration uses the first 130 weeks: the 95th percentile of rolling
two-week demand sums for opening inventory and weekly demand for new-order capacity,
using linear interpolation, ceiling and minimum one. Lead times are two weeks;
holding/backlog/fixed-order coefficients are one. s,S uses weeks 1-26, ddof=1.
The obsolete rho-based calibration sweep is not an active setting.

| Preserved evidence | Completed runs | Conditions |
|---|---:|---|
| M5, 130 periods | 16 | Updated evaluated settings |
| Synthetic revision 2, 130 periods | 16 | Updated evaluated settings |

There are **32 valid completed runs**, checked against current calibration, inputs,
policy settings, prompt versions and completion metadata. One separate M5
intermittent job is paused at 19/130 periods; two jobs are queued. These are not
completed evidence. Controlled adjacent has offline tests but no evaluated results.
Abandoned 277-period M5 runs and their historical-reference rows are excluded.
No replacement results were generated.

## Setup and offline checks

Use Python 3.10. The first command assumes the Windows launcher has Python 3.10
registered. Otherwise replace `py -3.10` with the full path to your Python 3.10
executable. The verified local environment uses Python 3.10.21. From this directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts/prepare_m5.py --help
.\.venv\Scripts\python.exe scripts/run_experiments.py --dataset synthetic --method ss --all --periods 130 --preflight
.\.venv\Scripts\python.exe scripts/run_experiments.py --dataset synthetic --method ss --all --periods 130 --output-root results/generated/offline_checks
.\.venv\Scripts\python.exe scripts/run_experiments.py --dataset m5 --method ss --record-id FOODS_2_368_TX_2_evaluation --periods 130 --preflight
```

Ray/RLlib and Gymnasium remain simulator dependencies. Tests mock generation calls
or use offline transports. No live LLM calls are needed for validation. For live
experiments, copy `.env.example` to `.env`, configure `GEMINI_API_KEY`, and select
`--method no_communication`, `adjacent`, `no_forecast`, or `controlled_adjacent`.
LLM preflight constructs an SDK client and needs a nonempty key but sends no requests;
a placeholder suffices for preflight-only checks. Removing `--preflight` makes paid
LLM calls. Always specify `--periods 130` for submission runs (the generic runner retains its legacy default).
Use a separate `--output-root` for new runs. Resume/extension require
compatible checkpoints; historical checkpoints are external to this submission.

## External M5 preprocessing

Obtain `sales_train_evaluation.csv` from the M5 Forecasting Accuracy competition on
Kaggle. Place it in ignored `Data/` and verify its SHA-256 against
`artifacts/m5/selection_manifest.json`. Explicitly override the historical `../Data/`
default, preparing into a separate directory to protect frozen inputs:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_m5.py --data-path Data/sales_train_evaluation.csv --output-dir Data/reproduced_m5
$env:M5_DATA_PATH = (Resolve-Path Data/sales_train_evaluation.csv).Path
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Only raw-source validation skips when this external CSV is absent. Other tests and
frozen-input simulations run offline. Synthetic seeds, generation and calibration
are checked against frozen definitions. Calibration overlaps evaluation; this is not
a strict out-of-sample benchmark.

## Regenerate the four figures

```powershell
.\.venv\Scripts\python.exe scripts/plot_m5_130_latest_comparison.py
.\.venv\Scripts\python.exe scripts/plot_four_regime_experiment_comparison.py --dataset synthetic
.\.venv\Scripts\python.exe scripts/plot_four_demand_series.py --dataset m5
.\.venv\Scripts\python.exe scripts/plot_four_demand_series.py --dataset synthetic
```

PDF/PNG outputs go to `results/figures/`; derived M5 CSV/JSON exports go to ignored
`results/generated/`. M5 figures use the approved intermittent replacement and
weeks 1-130. The synthetic source-demand figure retains all 277 source weeks;
it is an input illustration, not a 277-period evaluation.
Only preserved per-run values are plotted, without new cross-condition aggregation.

Full historical KPI auditing or resume needs trajectories, raw responses and
checkpoints retained in the original sibling `Simulator_New/`. The source repository
is `https://github.com/DoraJing77/Project-Study-LLM-`; remote verification returned 404,
so no verified remote archive is claimed. Source paths and available hashes remain
in the manifest. Do not remove original evidence before verified archival. LLM reruns
are nondeterministic and depend on continued model access. The submission is a separate Git repository.
