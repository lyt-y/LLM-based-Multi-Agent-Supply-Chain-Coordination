"""Plot frozen M5 or synthetic weekly demand series for the four regimes."""

if __package__ in (None, ''):
    from _bootstrap import configure_paths
else:
    from ._bootstrap import configure_paths
configure_paths()


from project_paths import M5_COMPARISONS, SYNTHETIC_RESULTS
from demand.synthetic_demand import load_series

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts' / 'm5' / 'representatives.json'
OUTPUT = ROOT / 'results/figures'
RECORDS = (
    ('Smooth', 'smooth', 'FOODS_1_002_WI_1_evaluation', '#2563eb'),
    ('Erratic', 'erratic', 'FOODS_1_019_CA_3_evaluation', '#e08727'),
    ('Intermittent', 'intermittent', 'FOODS_2_368_TX_2_evaluation', '#168a71'),
    ('Lumpy', 'lumpy', 'FOODS_1_037_TX_2_evaluation', '#a04ea1'),
)


def main(dataset='m5'):
    synthetic = dataset == 'synthetic'
    if synthetic:
        rows = load_series()
        representatives = {regime: [row for row in rows if row['regime'] == regime]
                           for _, regime, _, _ in RECORDS}
        records = tuple((label, regime, f'{regime}_001', color)
                        for label, regime, _, color in RECORDS)
        output = ROOT / 'results/figures'
        shared_max = max(max(row['weekly_demand']) for row in rows)
    else:
        representatives = json.loads(SOURCE.read_text(encoding='utf-8'))
        records = RECORDS
        output = OUTPUT
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True,
                             constrained_layout=True)
    title = ('Bernoulli-Gamma weekly demand by regime' if synthetic
             else 'M5 evaluation demand: weeks 1-130')
    fig.suptitle(title, fontsize=16, weight='bold')

    for ax, (label, regime, record_id, color) in zip(axes, records):
        matches = [row for row in representatives[regime] if row['id'] == record_id]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one representative for {record_id}')
        values = matches[0]['weekly_demand']
        if len(values) != 277:
            raise ValueError(f'Expected 277 weekly observations for {record_id}')
        if not synthetic:
            values = values[:130]
        periods = len(values)
        weeks = range(1, periods + 1)
        ax.bar(weeks, values, width=0.85, color=color, linewidth=0)
        mean = sum(values) / len(values)
        ax.axhline(mean, color='#333333', linestyle='--', linewidth=1,
                   label=f'Mean {mean:.1f}')
        ax.set_title(f'{label}  ·  {record_id}', loc='left', fontsize=11, weight='bold')
        ax.set_ylabel('Units / week')
        ax.set_ylim(0, max(1, (shared_max if synthetic else max(values)) * 1.12))
        ax.set_xlim(0.5, periods + 0.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.grid(axis='y', color='#dddddd', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper right', frameon=False, fontsize=9)
        annotation = f'{sum(value > 0 for value in values)}/{periods} nonzero weeks'
        if synthetic:
            stats = matches[0]['statistics']
            annotation = (f"Zero weeks: {stats['zero_weeks']}/277  |  "
                          f"ADI: {stats['adi']:.2f}  |  Positive CV squared: {stats['positive_cv2']:.2f}")
        ax.text(0.01, 0.88, annotation,
                transform=ax.transAxes, fontsize=9, color='#444444', va='top')

    axes[-1].set_xlabel('Week')
    axes[-1].set_xticks([1, 52, 104, 156, 208, 260, 277] if synthetic else [1, 26, 52, 78, 104, 130])
    output.mkdir(parents=True, exist_ok=True)
    stem = 'synthetic_demand_277' if synthetic else 'm5_demand_130'
    png = output / f'{stem}.png'
    pdf = output / f'{stem}.pdf'
    fig.savefig(png, dpi=200, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('m5', 'synthetic'), default='m5')
    main(parser.parse_args().dataset)
