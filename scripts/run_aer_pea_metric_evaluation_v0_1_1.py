#!/usr/bin/env python3
"""Operational entry point for the Codex-compatible v0.1.1 output schema."""

from __future__ import annotations

import sys
from pathlib import Path

import run_aer_pea_metric_evaluation as frozen_v0_1_0

SCHEMA = (
    frozen_v0_1_0.CASE_ROOT
    / "construction/metric-evaluation-output.schema.v0.1.1-development.json"
)


def main() -> int:
    if not any(
        argument == "--output-schema" or argument.startswith("--output-schema=")
        for argument in sys.argv[1:]
    ):
        sys.argv.extend(("--output-schema", str(Path(SCHEMA).resolve())))
    return frozen_v0_1_0.main()


if __name__ == "__main__":
    raise SystemExit(main())
