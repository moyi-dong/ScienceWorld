#!/usr/bin/env python3
"""Run the frozen 0.3.0 held-out oracle matrix exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from run_aer_pea_calibration import _safe_write_json, _sha256
from run_aer_pea_scripted_oracle_v0_3_0 import run_one

SCRIPT_PATH = Path(__file__).resolve()


def _require_frozen_inputs(
    matrix_path: Path, freeze_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_for_held_out":
        raise ValueError("held-out freeze manifest is not frozen_for_held_out")
    if freeze.get("acceptance_spec_version") != "0.3.0":
        raise ValueError("held-out freeze manifest is not for acceptance spec 0.3.0")
    expected_matrix = freeze.get("acceptance_matrix", {})
    if expected_matrix.get("sha256") != _sha256(matrix_path):
        raise ValueError("acceptance matrix does not match the frozen SHA-256")
    expected_runner = freeze.get("implementation_provenance", {}).get(
        "oracle_matrix_runner_sha256"
    )
    if expected_runner != _sha256(SCRIPT_PATH):
        raise ValueError("oracle matrix runner does not match the frozen SHA-256")
    if matrix.get("spec_version") != "0.3.0" or matrix.get("status") != "frozen":
        raise ValueError("acceptance matrix is not the frozen 0.3.0 matrix")
    return matrix, freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--held-out-freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        matrix, freeze = _require_frozen_inputs(
            args.matrix, args.held_out_freeze_manifest
        )
    except ValueError as error:
        parser.error(str(error))
    cells = matrix["held_out"]["oracle_cells"]
    if len(cells) != freeze["registered_execution"]["oracle_cells"]:
        parser.error("held-out cell count differs from the frozen registration")
    if args.output.exists():
        parser.error(f"refusing to overwrite matrix directory {args.output}")

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
                result_class="held-out acceptance",
            )
        )

    payload = {
        "schema_version": "aer.pea.scripted-oracle-matrix-result.v1",
        "result_class": "held-out acceptance",
        "policy_version": "0.3.0-frozen",
        "split": "held_out",
        "matrix_sha256": _sha256(args.matrix),
        "freeze_manifest_sha256": hashlib.sha256(
            args.held_out_freeze_manifest.read_bytes()
        ).hexdigest(),
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
