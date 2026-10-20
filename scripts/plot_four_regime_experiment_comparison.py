"""Compare demand regimes across four experiment methods for M5 or synthetic data."""

if __package__ in (None, ''):
    from _bootstrap import configure_paths
else:
    from ._bootstrap import configure_paths
configure_paths()
if __package__ in (None, ""):
    from _results import select_run, METHODS as RESULT_METHODS
else:
    from ._results import select_run, METHODS as RESULT_METHODS


from project_paths import M5_RESULTS, M5_COMPARISONS, SYNTHETIC_RESULTS, SYNTHETIC_ARTIFACTS

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
RESULTS = M5_RESULTS
REGIMES = (
    ("Smooth", "smooth", "FOODS_1_002_WI_1_evaluation"),
    ("Erratic", "erratic", "FOODS_1_019_CA_3_evaluation"),
    ("Intermittent", "intermittent", "FOODS_2_368_TX_2_evaluation"),
    ("Lumpy", "lumpy", "FOODS_1_037_TX_2_evaluation"),
)
METHODS = (
    ("Rule based (s, S)", "rule_based_sS_baseline", "summary.json", "#64748b"),
    ("Gemini: no communication", "gemini_no_communication", "summary.json", "#2563eb"),
    ("Gemini: local communication", "gemini_local_communication", None, "#059669"),
    (
        "Gemini: no forecast input",
        "gemini_no_communication_without_forecast_input",
        None,
        "#d97706",
    ),
)


def result_row(folder, summary_name, regime, record_id, alternative=False,
               dataset="m5", input_sha256=None):
    selected, _ = select_run(dataset, RESULT_METHODS[folder], regime, record_id,
                             130,
                             input_sha256=input_sha256)
    row = selected['summary'] if dataset == 'synthetic' else selected['evaluation_summary']
    if dataset == 'synthetic' and row['global_operational_cost'] != sum(
            row[key] for key in ('holding_cost', 'backlog_cost', 'fixed_ordering_cost')):
        raise ValueError('Inconsistent stored operational cost')
    return row


def metric_label(metric, value):
    if metric == "global_operational_cost":
        return f"{value:,.0f}"
    if metric == "global_bullwhip":
        return f"{value:.5f}" if value >= 0.001 else f"{value:.2e}"
    return "0" if value == 0 else f"{value / 1_000_000:.2f}M"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("m5", "synthetic"), default="m5")
    parser.add_argument("--intermittent-alternative", action="store_true",
                        help="Plot the newer FOODS_2_368_TX_2 intermittent series")
    parser.add_argument("--updated-intermittent", action="store_true",
                        help="Compare all four regimes using the newer intermittent series")
    args = parser.parse_args()
    if args.intermittent_alternative and args.updated_intermittent:
        parser.error("Choose one intermittent plot option")
    if args.dataset == "synthetic" and (args.intermittent_alternative or args.updated_intermittent):
        parser.error("Intermittent alternative options apply only to M5")
    if args.dataset == "m5" and not args.intermittent_alternative:
        import runpy
        runpy.run_path(str(ROOT / "scripts/plot_m5_130_latest_comparison.py"), run_name="__main__")
        return
    synthetic = args.dataset == "synthetic"
    periods = 130
    input_sha256 = (hashlib.sha256((SYNTHETIC_ARTIFACTS / "demand_series.json").read_bytes()).hexdigest()
                    if synthetic else None)
    updated_record = ("Intermittent", "intermittent", "FOODS_2_368_TX_2_evaluation")
    regimes = ((updated_record,) if args.intermittent_alternative else
               tuple(updated_record if row[1] == "intermittent" else row
                     for row in REGIMES)
               if args.updated_intermittent else REGIMES)
    if synthetic:
        regimes = tuple((label, regime, f"{regime}_001") for label, regime, _ in REGIMES)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(len(regimes), 3, figsize=(17, 4.5 if args.intermittent_alternative else 12.2),
                             squeeze=False)
    fig.subplots_adjust(left=0.225, right=0.97,
                        top=0.77 if args.intermittent_alternative else 0.90,
                        bottom=0.25 if args.intermittent_alternative else 0.09,
                        wspace=0.27, hspace=0.44)
    title = ("Intermittent demand - four inventory policies (130 periods)"
             if args.intermittent_alternative else
             "Four demand regimes - four inventory policies (130 periods)")
    if synthetic:
        title = f"Synthetic data: four demand regimes - four inventory policies ({periods} periods)"
    fig.suptitle(title, fontsize=17, fontweight="bold",
                 y=0.96 if args.intermittent_alternative else 0.97)
    columns = (
        ("global_operational_cost", "Global operational cost", "Cost units"),
        ("global_bullwhip", "Global bullwhip", "Ratio"),
        ("total_tokens", "LLM token usage", "Million tokens"),
    )
    colors = [method[3] for method in METHODS]
    labels = [method[0] for method in METHODS]

    for row_idx, (regime_label, regime, record_id) in enumerate(regimes):
        rows = [result_row(folder, summary, regime, record_id,
                           args.intermittent_alternative or
                           (args.updated_intermittent and regime == "intermittent"),
                           dataset=args.dataset, input_sha256=input_sha256)
                for _, folder, summary, _ in METHODS]
        for col_idx, (metric, title, x_label) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            values = [float(row.get(metric, 0)) for row in rows]
            positions = list(range(len(METHODS)))
            ax.barh(positions, values, color=colors, height=0.65)
            ax.set_yticks(positions, labels if col_idx == 0 else [""] * len(labels))
            ax.invert_yaxis()
            maximum = max(values)
            ax.set_xlim(0, maximum * 1.25 if maximum else 1)
            for position, value in zip(positions, values):
                ax.text(value + maximum * 0.015, position, metric_label(metric, value),
                        va="center", ha="left", fontsize=8.5, color="#1f2937")
            ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            if col_idx == 0:
                ax.text(-0.85, 0.5, regime_label,
                        transform=ax.transAxes, va="center", ha="left",
                        fontsize=10, fontweight="bold", color="#111827")
            if row_idx == 0:
                ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
            if row_idx == len(regimes) - 1:
                ax.set_xlabel(x_label)
            if metric == "total_tokens":
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}"))
            elif metric == "global_operational_cost":
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                if maximum >= 100_000:
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e3:g}k"))
            elif metric == "global_bullwhip" and maximum < 0.001:
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    if synthetic:
        fig.text(0.5, 0.035,
                 "Current synthetic data, revision 2; weeks 1-130. Each panel has its own scale.\n"
                 "Global bullwhip = product of four stage ratios. Cost excludes API charges; tokens include failed attempts.",
                 ha="center", fontsize=9, color="#4b5563")
    else:
        fig.text(0.5, 0.08 if args.intermittent_alternative else 0.035,
                 ("FOODS_2_368_TX_2_evaluation. " if args.intermittent_alternative
                  else "Intermittent: FOODS_2_368_TX_2_evaluation; other regimes: original representative series. "
                  if args.updated_intermittent else "Original representative series for each regime. ") +
                 "Each panel has its own scale. "
                 "Operational cost excludes API charges; tokens are counts, not currency.",
                 ha="center", fontsize=9, color="#4b5563")
    stem = ("intermittent_four_experiment_comparison_130_FOODS_2_368_TX_2"
            if args.intermittent_alternative else
            "four_regime_four_experiment_comparison_130_updated_intermittent"
            if args.updated_intermittent else "four_regime_four_experiment_comparison_130")
    if synthetic:
        stem = "four_regime_four_experiment_comparison_130"
    if synthetic:
        stem = 'synthetic_130_comparison'
    else:
        stem = 'm5_130_intermittent_comparison' if args.intermittent_alternative else 'm5_130_comparison'
    output_directory = ROOT / 'results' / ('figures' if synthetic or args.updated_intermittent else 'generated')
    output_directory.mkdir(parents=True, exist_ok=True)
    output_png = output_directory / f"{stem}.png"
    output_pdf = output_directory / f"{stem}.pdf"
    fig.savefig(output_png, dpi=200, facecolor="white")
    fig.savefig(output_pdf, facecolor="white")
    plt.close(fig)
    print(output_png)
    print(output_pdf)


if __name__ == "__main__":
    main()
