# LLM-Based Replenishment Decisions across Demand Regimes

### When Do Forecasting and Information Sharing Add Value?

**Project Study · Technical University of Munich**  
**Authors:** Jing He and Yiting Liu  
**Program:** Master in Management and Digital Technology

This repository accompanies a Project Study examining how large language model (LLM) agents make decentralized replenishment decisions in a four-echelon supply chain. We compare a conventional rule-based inventory policy with three LLM-based decision configurations under **smooth, erratic, intermittent, and lumpy demand**, using both synthetic demand and observed retail-sales series from the M5 dataset.

The focus is not simply whether LLMs can place orders, but **when forecasting support and adjacent-stage information sharing improve—or fail to improve—operational outcomes relative to simpler decision configurations**.

## Research Questions

Our overarching question is:

> How does the operational performance of LLM-based replenishment agents vary across demand regimes, and under what conditions do forecasting support and adjacent-stage information exchange provide additional value?

- **RQ1 — Relative performance:** How do LLM-based replenishment configurations perform relative to a conventional rule-based inventory policy across demand regimes and supply chain stages?
- **RQ2 — Forecasting support:** How does external forecasting support change the operational performance of independent LLM agents across demand regimes?
- **RQ3 — Information exchange:** Given external forecasting support, how does adjacent-stage information exchange affect the performance of LLM agents across demand regimes?

These questions are investigated using matched simulation cases, not a population-wide or statistically replicated causal design.

## Supply Chain Simulation

The simulator represents a single product passing through a serial, four-stage supply chain:

**Retailer → Wholesaler → Distributor → Manufacturer**

Customer demand enters at the retailer; replenishment orders travel upstream, and shipments move downstream. Each stage holds inventory, places weekly orders, and accumulates unfulfilled demand as backlog. Orders are decided sequentially from retailer to manufacturer. Shipments have a deterministic **two-week lead time**; the external source supplies the manufacturer without an additional production-capacity constraint.

The evaluated configurations share the same physical environment and operational-cost coefficients for each demand series. Initial inventory and new-order limits are calibrated using the first **130 weeks** of each source sequence. Operational cost consists of holding, backlog, and fixed positive-order costs, with each coefficient set to **1**. The `(s,S)` benchmark uses thresholds calibrated from weeks 1–26, while LLM decisions use their permitted observations and completed-period histories. These retrospective calibration choices are important limitations, noted below.

## Decision Frameworks

| Configuration | CLI method | Information available |
|---|---|---|
| **Rule-based `(s,S)`** | `ss` | Fixed replenishment thresholds; no LLM calls |
| **Independent LLM** | `no_forecast` | Local operational state and history, without an externally supplied forecast |
| **Independent LLM + forecasting** | `no_communication` | Same decentralized decisions, with an SMA-26 next-period demand forecast |
| **LLM + forecasting + adjacent information sharing** | `adjacent` | Forecast support plus neighboring opening-state information and upstream replenishment insights |

The LLM configurations use a pretrained Gemini model without task-specific fine-tuning. Forecasts use a 26-period simple moving average of available completed observations, with an expanding-history warm-up. Adjacent information sharing adds observations and messages between neighboring stages, **not** negotiation or centralized order optimization.

The code also includes an experimental `controlled_adjacent` implementation, but this variant has **no retained evaluated runs** and is not part of the comparisons summarized here.

## Demand Data and Experimental Design

We compare two demand sources:

- **Synthetic:** Bernoulli–Gamma demand sequences with parameters chosen to represent the four demand regimes, with frozen generation seeds and source artifacts.
- **M5:** Daily item–store sales aggregated into nonoverlapping seven-day totals. Series are classified using average inter-demand interval (ADI) and squared coefficient of variation of positive demand (CV²). A k-medoids procedure retains 30 representatives per regime; one representative per regime is evaluated in the study.

Both source collections contain **277 weeks**, but **only weeks 1–130 are used for the reported simulations**. Source-series length should not be confused with the evaluation horizon.

| Dataset | Demand regimes | Evaluated policies | Episodes | Evaluation horizon |
|---|---:|---:|---:|---:|
| Synthetic Bernoulli–Gamma | 4 | 4 | 16 | 130 weeks |
| M5 retail-sales series | 4 | 4 | 16 | 130 weeks |
| **Total** | | | **32** | |

Each dataset–regime–policy combination has **one completed episode**. All reported comparisons use matched demand trajectories and environment settings within the relevant case. Detailed run identities, parameters, and checksums are preserved in [`results/manifest.json`](results/manifest.json) and [`results/summary.json`](results/summary.json).

## Evaluation

The report evaluates three complementary dimensions:

1. **Operational cost:** cumulative inventory holding, backlog, and fixed ordering costs at each stage and across the supply chain.
2. **Order variability / bullwhip:** stage-level and system-level indicators of demand amplification.
3. **LLM resource use:** input and output token consumption, including recorded failed attempts where usage is available. Token counts are distinct from operational costs and estimated monetary API expenditures.

## Main Findings

The **Project Study report** describes the following patterns for these 32 evaluated cases:

- **LLM performance varies by demand regime.** Averaged across the three LLM configurations, operational costs are lower than the rule-based benchmark in **six of eight** dataset–regime cases. The strongest reported relative cost reductions occur under smooth demand: **29.3% for M5** and **14.3% for synthetic demand**. Under erratic demand, average LLM costs are higher in both datasets.
- **Forecasting is not uniformly beneficial.** Adding SMA-26 forecast input lowers operational cost in three of four synthetic regimes but increases cost in all four M5 regimes. Forecast accuracy alone does not explain the differences observed in replenishment cost.
- **More information sharing is not automatically better.** Compared with independent forecast-supported LLM agents, the adjacent-communication configuration records higher operational cost in **six of eight** cases and higher token consumption. The comparison measures the **whole communication-and-information protocol**, not an isolated causal effect of messaging.
- **Supply-chain effects vary by stage.** The report finds more pronounced downstream–upstream cost asymmetry in the selected M5 cases than in the selected synthetic cases, pointing to the importance of inventory and backlog allocation across the chain.

These are **descriptive observations for selected cases**, not statistically significant or universally generalizable effects.

### Figures available in this repository

- [M5 policy comparison (130 weeks)](results/figures/m5_130_comparison.pdf)
- [Synthetic policy comparison (130 weeks)](results/figures/synthetic_130_comparison.pdf)
- [M5 demand patterns (weeks 1–130)](results/figures/m5_demand_130.pdf)
- [Synthetic demand patterns (weeks 1–130)](results/figures/synthetic_demand_130.pdf)

The manuscript includes additional stage-level and ablation figures that are **not bundled as separate files** in this compact repository.

## Repository Structure

```text
src/                  Simulation, demand models, policies, agents, and evaluation
scripts/              Data preparation, experiment execution, and plotting
artifacts/            Frozen demand inputs and selection/generation metadata
results/              Retained run summaries, provenance, and comparison figures
tests/                Offline scientific and publication-integrity tests
docs/                 Detailed implementation methodology
requirements.txt      Python dependencies
.env.example          Example environment-variable configuration
```

## Getting Started

Python **3.10** is the locally verified environment. From the repository root, create and activate a virtual environment, then install requirements:

```bash
python -m venv .venv
# Activate .venv using your operating system's procedure
python -m pip install -r requirements.txt
python -m pytest tests -q -p no:cacheprovider
```

The frozen demand artifacts allow offline checks without downloading the raw M5 data. For full raw-data validation, obtain `sales_train_evaluation.csv` from the [M5 Forecasting Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy), and place it under the locally ignored `Data/` directory. Validate its checksum against [`artifacts/m5/selection_manifest.json`](artifacts/m5/selection_manifest.json).

```bash
python scripts/prepare_m5.py --data-path Data/sales_train_evaluation.csv --output-dir Data/reproduced_m5
```

### Run offline preflights and a baseline

```bash
python scripts/run_experiments.py --dataset synthetic --method ss --all --periods 130 --preflight
python scripts/run_experiments.py --dataset m5 --method ss --record-id FOODS_2_368_TX_2_evaluation --periods 130 --preflight
python scripts/run_experiments.py --dataset synthetic --method ss --all --periods 130 --output-root results/generated/synthetic_baselines
```

To run an LLM configuration, choose `--method no_forecast`, `--method no_communication`, or `--method adjacent` with the same record and horizon. Set `GEMINI_API_KEY` in a local `.env` file as required. **Removing `--preflight` for an LLM method sends live model requests and may incur charges.** Write new experiments to a separate output root rather than overwriting the retained results.

### Regenerate repository figures

```bash
python scripts/plot_m5_130_latest_comparison.py
python scripts/plot_four_regime_experiment_comparison.py --dataset synthetic
python scripts/plot_four_demand_series.py --dataset m5
python scripts/plot_four_demand_series.py --dataset synthetic
```

## Reproducibility and Limitations

- **Small sample:** one demand series and one run per dataset–regime–method combination; no repeated LLM trials or confidence intervals.
- **Retrospective information:** M5 regime classification and series selection use the full 277-week source; initial inventory and order limits are calibrated on the first 130 evaluation weeks, and the `(s,S)` benchmark uses weeks 1–26 while being applied from week 1. These are not strict forward-looking out-of-sample experiments.
- **Simplified environment:** one product, fixed lead times and cost coefficients, no explicit seasonality or serial correlation in synthetic generation, and no externally imposed supply disruption scenarios.
- **Limited intervention scope:** SMA-26 forecasts and adjacent-stage exchange are examined; the study does not separately estimate all forecasting–communication interaction effects.
- **LLM reproducibility:** historical raw prompts/responses, detailed trajectories and checkpoints are external to this compact submission; model behavior may vary across reruns even with temperature set to zero.

For precise current implementation details, see [`docs/methodology_latest_full_zh.md`](docs/methodology_latest_full_zh.md). For the complete scientific narrative and literature review, consult the accompanying Project Study manuscript.
