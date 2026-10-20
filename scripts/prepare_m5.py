"""Explicitly preprocess M5 or export its historical forecast artifacts."""
if __package__ in (None, ''):
    from _bootstrap import configure_paths
else:
    from ._bootstrap import configure_paths
configure_paths()

import argparse
from pathlib import Path
from demand.m5_demand import DATA_PATH, ARTIFACT_PATH, preprocess, persist_demand_forecasts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, default=DATA_PATH)
    parser.add_argument('--output-dir', type=Path, default=ARTIFACT_PATH)
    parser.add_argument('--forecasts-only', action='store_true')
    args = parser.parse_args(argv)
    if args.forecasts_only:
        persist_demand_forecasts(args.output_dir)
    else:
        preprocess(args.data_path, args.output_dir)


if __name__ == '__main__':
    main()
