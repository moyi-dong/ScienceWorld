#!/usr/bin/env python3
"""Build the deterministic pea v0.6.4 four-model comparison report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aer_bench.pea_model_comparison_v064 import (  # noqa: E402
    comparison_bindings,
    summarize_comparison,
    write_comparison_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sol-baseline", type=Path, required=True)
    parser.add_argument("--sol-five-run-summary", type=Path, required=True)
    parser.add_argument("--terra", type=Path, required=True)
    parser.add_argument("--luna", type=Path, required=True)
    parser.add_argument("--gpt-5-5", dest="gpt_5_5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bindings = comparison_bindings(
        sol_baseline=args.sol_baseline.resolve(),
        terra=args.terra.resolve(),
        luna=args.luna.resolve(),
        gpt_5_5=args.gpt_5_5.resolve(),
    )
    summary, inventory = summarize_comparison(
        bindings,
        sol_five_run_summary=args.sol_five_run_summary.resolve(),
    )
    write_comparison_reports(args.output.resolve(), summary, inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
