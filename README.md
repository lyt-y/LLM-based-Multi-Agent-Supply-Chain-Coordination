# LLM-Based Multi-Agent Supply Chain Coordination

Simulation-based evaluation of replenishment decisions and adjacent-agent communication under heterogeneous demand.

## 1. Project Overview

This project studies decentralized replenishment in a four-stage supply chain comprising a retailer, wholesaler, distributor, and manufacturer. It compares a fixed rule-based policy with large language model (LLM) agents operating under different information protocols, using observed retail demand and controlled synthetic demand.

The repository provides a simulation framework and a retained set of matched policy comparisons across four demand regimes. The framework records operational performance, demand amplification, and the computational resources required for LLM decisions and communication. It supports investigation of these trade-offs without assuming that communication improves coordination.

## 2. Research Objective and Questions

The objective is to examine how replenishment policies and local information affect supply-chain performance under common physical and cost conditions. The study addresses three questions:

- How do independent LLM replenishment decisions compare with a fixed `(s,S)` policy across demand regimes?
- How does removing explicit demand-forecast input affect LLM decisions and operational outcomes?
- How does the complete adjacent-agent communication protocol, including the additional information supplied to agents, compare with independent LLM decisions in operational performance and computational resource use?

These are descriptive simulation comparisons; the retained experiments do not establish statistically significant effects or general coordination benefits. The communication comparison evaluates the complete information protocol and does not isolate a causal effect of communication alone.

## 3. Methodology

Each stage manages local inventory, backlog, and replenishment orders for a single product. Orders propagate upstream and shipments move downstream with two-period lead times at every stage. LLM decisions are made sequentially from the retailer to the manufacturer using the information permitted by each protocol. The simulator applies inventory, shipment, backlog, and cost transitions; there is no centralized optimization or reinforcement-learning training.

The evaluated methods are:

| Method | Decision and information protocol |
|---|---|
| Rule-based replenishment (`ss`) | Fixed `(s,S)` thresholds calibrated from weeks 1-26, using the sample standard deviation (`ddof=1`). |
| Independent LLM decisions (`no_communication`) | Stage-local decisions with local history and an explicit downstream-demand forecast; no inter-agent messages. |
| No-forecast LLM decisions (`no_forecast`) | Independent decisions with the explicit forecast input removed. |
| LLM communication (`adjacent`) | Adjacent stages exchange opening-state information and order insights in addition to local decision inputs. |

Forecasts use a trailing mean of at most 26 completed observations of each stage's downstream demand, with an expanding history during warm-up and no forecast before any observation is available. At decision week `t`, the forecast targets `t+1` and uses completed weeks through `t-1`. Upstream forecasts use downstream orders rather than future customer demand. Current experiments compute these forecasts online; the stored M5 forecast arrays are source-length artifacts.

The retained LLM runs use the model identifier `gemini-3.8-flash`, temperature zero, and low thinking. Exact prompts, policy definitions, transition rules, and metric conventions are documented in the [methodology](docs/methodology_latest_full_zh.md); per-run settings are recorded in the [manifest](results/manifest.json).

## 4. Experimental Design

Both datasets are evaluated over **weeks 1-130**. The shared runner and standalone LLM entry points default to 130 periods. The frozen source sequences contain 277 weeks: `source_num_periods` records source length, whereas run-level `num_periods` records the evaluation horizon.

M5 inputs aggregate daily product-store sales into consecutive seven-day totals. Frozen preprocessing selects 30 representative series per regime using deterministic alternating k-medoids, giving 120 available series. The retained evaluation uses one selected series per regime, rather than all 120.

Synthetic inputs comprise one frozen Bernoulli-Gamma series per regime (`bernoulli_gamma_v1`, current revision 2). Weekly demand is a Bernoulli occurrence indicator multiplied by a rounded-up Gamma quantity. Generation parameters, sampling order, and seeds are retained in the [generation configuration](artifacts/synthetic/bernoulli_gamma_v1/generation_config.json).

The four regimes are **smooth, erratic, intermittent, and lumpy**, distinguished by demand occurrence and positive-demand variability. M5 classification uses average demand interval and squared coefficient of variation of positive demand, with thresholds 1.32 and 0.49, respectively. Classification and representative selection use the full source sequences.

The retained design uses the following series. Each row includes all four evaluated methods: `ss`, `no_communication`, `no_forecast`, and `adjacent`. There is one completed run per method and series.

| Dataset | Regime | Evaluated series | Completed runs | Evaluation horizon |
|---|---|---|---:|---:|
| M5 | Smooth | `FOODS_1_002_WI_1_evaluation` | 4 | 130 |
| M5 | Erratic | `FOODS_1_019_CA_3_evaluation` | 4 | 130 |
| M5 | Intermittent | `FOODS_2_368_TX_2_evaluation` | 4 | 130 |
| M5 | Lumpy | `FOODS_1_037_TX_2_evaluation` | 4 | 130 |
| Synthetic | Smooth | `smooth_001` | 4 | 130 |
| Synthetic | Erratic | `erratic_001` | 4 | 130 |
| Synthetic | Intermittent | `intermittent_001` | 4 | 130 |
| Synthetic | Lumpy | `lumpy_001` | 4 | 130 |

For each evaluated series, all four stages share the following calibration:

- Opening inventory: the ceiling of the linear 95th percentile of overlapping two-week demand sums within weeks 1-130, with minimum one.
- New-order capacity: the ceiling of the linear 95th percentile of weekly demand in weeks 1-130, including zeros, with minimum one. This limits new replenishment orders; fulfillment is limited by available opening inventory.
- Holding, backlog, and fixed positive-order cost coefficients: one each.

Calibration overlaps evaluation. The design is therefore a controlled simulation comparison, rather than a strict out-of-sample forecasting or inventory-control benchmark.

## 5. Evaluation Metrics

The implementation records:

- **Operational cost:** holding, backlog, and fixed ordering costs, by stage and in total. Episode reward is the negative operational cost; API charges are excluded.
- **Service and inventory:** average inventory, average backlog, backlog frequency, order-capacity saturation, and supply-shortage frequency. The retailer's cumulative fulfilled fraction includes late fulfillment and is not an on-time service rate. Some diagnostics are unavailable in retained records.
- **Demand amplification:** each stage's population order-variance ratio relative to customer demand; global bullwhip is the product of the four stage ratios. A separate coefficient-of-variation ratio compares outgoing orders with received demand. Undefined values remain null.
- **Computational and communication resources:** recorded model calls, token usage, runtime, and communication counts. Token accounting includes recorded failed attempts and is not a monetary cost estimate.

Detailed definitions and accounting scopes are given in the [methodology](docs/methodology_latest_full_zh.md).

## 6. Results

The [retained summary](results/summary.json) contains **32 successfully completed runs**, all evaluated for 130 periods:

| Dataset | Demand regimes | Evaluated methods | Completed runs |
|---|---:|---:|---:|
| M5 | 4 | 4 | 16 |
| Synthetic | 4 | 4 | 16 |

There is one retained run per dataset-regime-method combination. The [manifest](results/manifest.json) links each run to its configuration, input checksums, and evidence provenance. Results are individual observations without replication-based uncertainty estimates.

Retained figures:

- [M5 policy comparison, 130 periods](results/figures/m5_130_comparison.pdf).
- [Synthetic policy comparison, 130 periods](results/figures/synthetic_130_comparison.pdf).
- [M5 demand inputs, weeks 1-130](results/figures/m5_demand_130.pdf).
- [Synthetic demand inputs, weeks 1-130](results/figures/synthetic_demand_130.pdf). Both demand figures show the evaluation window; the frozen source sequences remain 277 weeks long.

The comparison figures show operational cost, global bullwhip, and token usage for each regime. Their values support inspection of these particular conditions; they do not establish a universally superior policy. Controlled adjacent communication is implemented and tested offline but has no retained evaluated results.

### Descriptive findings

Operational cost varies by dataset, regime, and protocol. No-forecast decisions have the lowest recorded cost in M5 smooth, intermittent, and lumpy demand; the rule-based policy has the lowest in M5 erratic demand. For synthetic demand, independent decisions have the lowest cost in smooth, erratic, and lumpy demand, while no-forecast decisions have the lowest in intermittent demand. Adjacent communication costs less than independent decisions in M5 intermittent demand (54,367 versus 67,180) and synthetic lumpy demand (36,698 versus 37,000), but costs more in the other six matched comparisons.

The rule-based policy has the lowest global bullwhip in seven of the eight evaluated series. The exception is synthetic lumpy demand, where independent decisions record approximately 0.2683 versus 0.4160 for the rule-based policy. Lower cost does not consistently coincide with lower bullwhip: for M5 smooth demand, no-forecast decisions cost 2,988 versus 5,551 for the rule-based policy, but global bullwhip is approximately 18.1409 versus 2.6251.

Among LLM methods, adjacent communication records the highest token usage and no-forecast decisions the lowest in every matched series comparison. Across the 24 retained LLM runs, recorded token usage ranges from 1,181,897 to 1,666,675 tokens per run, including recorded failed attempts. These are resource counts, not API expenditure estimates.

These observations describe one completed run per condition. They do not establish statistical significance, an isolated causal effect of information sharing, or a generally superior replenishment method.

## 7. Repository Structure

| Path | Contents |
|---|---|
| `src/` | Demand loaders, simulator, replenishment policies, LLM providers, and evaluation. |
| `scripts/` | Data preparation, experiment execution, recovery, and plotting commands. |
| `artifacts/` | Frozen demand inputs, selection and generation definitions, and source forecast artifacts. |
| `results/summary.json` | Retained individual experimental summaries. |
| `results/manifest.json` | Run identities, configurations, checksums, and provenance. |
| `results/figures/` | Retained comparison and demand figures. |
| `tests/` | Offline simulation, protocol, recovery, and publication-integrity tests. |
| `docs/` | Detailed methodology. |

## 8. Installation and Reproduction

Use Python 3.10. From the repository root, create a virtual environment:

```sh
python -m venv .venv
```

Activate the environment using the procedure for your platform, then install dependencies and run the offline checks:

```sh
python -m pip install -r requirements.txt
python -m pip check
python -m pytest tests -q -p no:cacheprovider
```

Ray/RLlib and Gymnasium are required by the simulator. Frozen demand inputs support offline execution without the external raw M5 CSV or live model calls. The raw-source validation test skips if the CSV is unavailable; set `M5_DATA_PATH` to its path to enable it.

Validate a 130-period configuration without generating results:

```sh
python scripts/run_experiments.py --dataset synthetic --method ss --all --periods 130 --preflight
python scripts/run_experiments.py --dataset m5 --method ss --record-id FOODS_2_368_TX_2_evaluation --periods 130 --preflight
```

Run deterministic baselines into a separate output directory:

```sh
python scripts/run_experiments.py --dataset synthetic --method ss --all --periods 130 --output-root results/generated/synthetic_baselines
python scripts/run_experiments.py --dataset m5 --method ss --record-id FOODS_2_368_TX_2_evaluation --periods 130 --output-root results/generated/m5_baseline
```

For M5, `--all` selects all 120 frozen representatives, not only the four evaluated series. Other explicit horizons are supported for general simulation within source-length bounds; such runs are separate from the approved 130-period study. Calibration remains based on the first 130 source weeks even when a shorter horizon is requested.

For live LLM experiments, copy `.env.example` to `.env`, set `GEMINI_API_KEY`, and replace `--method ss` with `no_communication`, `no_forecast`, or `adjacent`. Use the same selected record, `--periods 130`, and a separate output root. Removing `--preflight` executes model calls and may incur charges. LLM preflight requires a nonempty key but sends no generation requests.

To reproduce M5 preprocessing, obtain `sales_train_evaluation.csv` from the M5 Forecasting Accuracy competition, place it in `Data/`, and verify its SHA-256 against the [selection manifest](artifacts/m5/selection_manifest.json). Prepare into a separate directory:

```sh
python scripts/prepare_m5.py --data-path Data/sales_train_evaluation.csv --output-dir Data/reproduced_m5
```

Regenerate the four figures offline from retained inputs and summaries:

```sh
python scripts/plot_m5_130_latest_comparison.py
python scripts/plot_four_regime_experiment_comparison.py --dataset synthetic
python scripts/plot_four_demand_series.py --dataset m5
python scripts/plot_four_demand_series.py --dataset synthetic
```

These commands write PDF and PNG files to `results/figures/`; the M5 comparison also exports data to `results/generated/`. Regenerated PDF byte checksums may differ because of export metadata; the manifest records the retained files' checksums.

## 9. Limitations

The retained design contains one series and one completed run per regime and method for each dataset. It provides no replication-based inference or evidence that findings generalize to all M5 products or demand processes. M5 selection uses full-source statistics, environment calibration overlaps weeks 1-130, and `(s,S)` calibration uses weeks 1-26 of evaluation. Runtime forecasts use completed observations, but the overall experiment is not temporally held out.

The model represents a single product with fixed lead times, costs, and order limits. Synthetic demand omits explicit seasonality and temporal dependence. LLM reruns depend on external model availability and may differ even at temperature zero. Retained summaries and checksums support figure reproduction and configuration validation; independently recomputing historical trajectory KPIs or resuming original runs requires external trajectories, raw responses, and compatible checkpoints that are not bundled here.
