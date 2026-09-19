#!/usr/bin/env python3
"""Build the deterministic pea v0.6.5 four-model five-run report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aer_bench.pea_model_repeat_report_v065 import (  # noqa: E402
    CONTRACT_PATH,
    summarize,
    write_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, inventory = summarize(
        repository_root=ROOT,
        contract_path=args.contract,
    )
    write_reports(args.output.resolve(), summary, inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
