#!/usr/bin/env python3
"""Run a registered scripted-oracle matrix without exposing hidden configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_aer_pea_calibration import _safe_write_json, _sha256, _validate_split
from run_aer_pea_scripted_oracle import run_one

SCRIPT_PATH = Path(__file__).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "held_out"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=int)
    parser.add_argument("--held-out-freeze-manifest", type=Path)
    args = parser.parse_args()

    matrix: dict[str, Any] = json.loads(args.matrix.read_text(encoding="utf-8"))
    cells = matrix[args.split]["oracle_cells"]
    if args.root is not None:
        cells = [cell for cell in cells if cell["root"] == args.root]
        if not cells:
            parser.error(f"root {args.root} has no registered {args.split} cells")
    if args.output.exists():
        parser.error(f"refusing to overwrite matrix directory {args.output}")
    for cell in cells:
        _validate_split(
            args.split,
            cell["variation"],
            cell["root"],
            args.held_out_freeze_manifest,
        )

    result_class = (
        "held-out acceptance" if args.split == "held_out" else "dev calibration"
    )
    args.output.mkdir(parents=True)
    results = []
    for index, cell in enumerate(cells, start=1):
        print(
            f"CELL {index}/{len(cells)} {cell['world']} "
            f"variation={cell['variation']} root={cell['root']}",
            flush=True,
        )
        results.append(
            run_one(
                args.output,
                cell["world"],
                cell["variation"],
                cell["root"],
                result_class=result_class,
            )
        )

    payload = {
        "schema_version": "aer.pea.scripted-oracle-matrix-result.v1",
        "result_class": result_class,
        "split": args.split,
        "root_filter": args.root,
        "matrix_sha256": _sha256(args.matrix),
        "runner_sha256": _sha256(SCRIPT_PATH),
        "cell_count": len(results),
        "strict_success_count": sum(
            result["grade"]["strict_case_success"] for result in results
        ),
        "all_strict_success": all(
            result["grade"]["strict_case_success"] for result in results
        ),
        "results": results,
    }
    _safe_write_json(args.output / "matrix_summary.json", payload)
    return 0 if payload["all_strict_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
