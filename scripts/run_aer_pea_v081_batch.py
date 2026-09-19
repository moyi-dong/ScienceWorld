#!/usr/bin/env python3
"""Run the pea v0.8.1 control/mechanism model batch with bounded retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

SCIENCEWORLD_ROOT = Path(__file__).resolve().parents[1]
ROOT = SCIENCEWORLD_ROOT.parents[1]
sys.path.insert(0, str(SCIENCEWORLD_ROOT))

from scripts import run_aer_pea_v1_pilot as pilot  # isort: skip  # noqa: E402


BENCHMARK_VERSION = "pea-v0.8.1"
DEFAULT_WORKERS = 12
DEFAULT_RETRIES = 3
CONTROL_MODELS = (
    ("gpt-6", "gpt-6-astra", "low"),
    ("gpt-5.6-luna", "gpt-5.6-luna", "high"),
    ("deepseek-v4-flash", "deepseek-v4-flash", "high"),
)
MECHANISM_MODELS = (
    ("gpt-6", "gpt-6-astra", "low"),
    ("gpt-5.6-sol", "gpt-5.6-sol", "low"),
    ("gpt-5.6-terra", "gpt-5.6-terra", "high"),
    ("gpt-5.6-luna", "gpt-5.6-luna", "high"),
    ("deepseek-v4-flash", "deepseek-v4-flash", "high"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jobs(matrix: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    controls = sorted(
        (item for item in matrix.values() if item["group"] == "single_noise"),
        key=lambda item: item["id"],
    )
    mechanisms = sorted(
        (item for item in matrix.values() if item["group"] == "mechanism"),
        key=lambda item: item["id"],
    )
    if len(controls) != 3 or len(mechanisms) != 4:
        raise ValueError("v0.8.1 batch requires 3 single-noise controls and 4 mechanisms")
    jobs: list[dict[str, Any]] = []
    for configuration in controls:
        for repetition in range(1, 11):
            label, model, effort = CONTROL_MODELS[(repetition - 1) % len(CONTROL_MODELS)]
            jobs.append(
                {
                    "batch_group": "single_noise_control",
                    "configuration": configuration,
                    "repetition": repetition,
                    "model_label": label,
                    "model": model,
                    "reasoning_effort": effort,
                }
            )
    for configuration in mechanisms:
        for repetition, (label, model, effort) in enumerate(MECHANISM_MODELS, 1):
            jobs.append(
                {
                    "batch_group": "mechanism",
                    "configuration": configuration,
                    "repetition": repetition,
                    "model_label": label,
                    "model": model,
                    "reasoning_effort": effort,
                }
            )
    if len(jobs) != 50:
        raise AssertionError(f"unexpected v0.8.1 job count: {len(jobs)}")
    return jobs


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_with_retries(
    job: dict[str, Any], output_root: Path, timeout: int, step_limit: int, retries: int
) -> dict[str, Any]:
    configuration = job["configuration"]
    label = job["model_label"]
    for attempt in range(1, retries + 1):
        attempt_root = output_root / "attempts" / f"attempt-{attempt:02d}"
        try:
            print(
                f"START {job['batch_group']} {configuration['id']} "
                f"rep={job['repetition']} model={label} attempt={attempt}/{retries}",
                flush=True,
            )
            if job["model"] == "deepseek-v4-flash":
                metadata = pilot.run_deepseek_episode(
                    attempt_root,
                    world=configuration["world"],
                    repetition=job["repetition"],
                    variation=0,
                    case_root=configuration["development_root"],
                    timeout_seconds=timeout,
                    step_limit=step_limit,
                    noise_levels=configuration["noise_levels"],
                    configuration=configuration,
                    benchmark_version=BENCHMARK_VERSION,
                )
            else:
                metadata = pilot.run_episode(
                    pilot.CodexRunner(),
                    attempt_root,
                    world=configuration["world"],
                    repetition=job["repetition"],
                    variation=0,
                    case_root=configuration["development_root"],
                    timeout_seconds=timeout,
                    step_limit=step_limit,
                    noise_levels=configuration["noise_levels"],
                    configuration=configuration,
                    model=job["model"],
                    reasoning_effort=job["reasoning_effort"],
                    benchmark_version=BENCHMARK_VERSION,
                )
            artifact_dir = attempt_root / metadata["run_id"]
            metadata.update(
                {
                    "batch_group": job["batch_group"],
                    "model_label": label,
                    "benchmark_version": BENCHMARK_VERSION,
                    "attempt": attempt,
                    "requested_retries": retries,
                }
            )
            _write_json(artifact_dir / "run_metadata.json", metadata)
            if metadata.get("status") in {"completed", "succeeded"}:
                print(
                    f"DONE {configuration['id']} rep={job['repetition']} model={label} "
                    f"status={metadata['status']}",
                    flush=True,
                )
                return metadata
            error = f"model status {metadata.get('status')}"
        except Exception as exc:  # keep one failed service from cancelling the batch
            error = f"{type(exc).__name__}: {exc}"
        print(
            f"RETRY {configuration['id']} rep={job['repetition']} model={label} "
            f"attempt={attempt}: {error}",
            flush=True,
        )
        if attempt < retries:
            time.sleep(30)
    return {
        "batch_group": job["batch_group"],
        "configuration_id": configuration["id"],
        "repetition": job["repetition"],
        "model": job["model"],
        "model_label": label,
        "reasoning_effort": job["reasoning_effort"],
        "benchmark_version": BENCHMARK_VERSION,
        "status": "retry_exhausted",
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--step-limit", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.workers <= 18:
        parser.error("--workers must be between 1 and 18")
    if not 1 <= args.retries <= 5:
        parser.error("--retries must be between 1 and 5")
    if args.timeout <= 0 or args.step_limit <= 0:
        parser.error("timeout and step-limit must be positive")

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    matrix = pilot.load_hidden_configuration_matrix()
    jobs = _jobs(matrix)
    manifest = {
        "schema_version": "aer.pea.batch.v0.8.1-development",
        "benchmark_version": BENCHMARK_VERSION,
        "requested_control_runs": 30,
        "requested_mechanism_runs": 20,
        "requested_total_runs": 50,
        "workers": args.workers,
        "retries": args.retries,
        "timeout_seconds": args.timeout,
        "step_limit": args.step_limit,
        "scienceworld_jar_sha256": _sha256(
            pilot.SCIENCEWORLD_ROOT / "scienceworld" / "scienceworld.jar"
        ),
        "jobs": [
            {
                "batch_group": job["batch_group"],
                "configuration_id": job["configuration"]["id"],
                "world": job["configuration"]["world"],
                "case_root": job["configuration"]["development_root"],
                "noise_levels": job["configuration"]["noise_levels"],
                "repetition": job["repetition"],
                "model_label": job["model_label"],
                "model": job["model"],
                "reasoning_effort": job["reasoning_effort"],
                "benchmark_version": BENCHMARK_VERSION,
            }
            for job in jobs
        ],
    }
    _write_json(output_root / "batch_manifest.json", manifest)
    results: list[dict[str, Any]] = []
    results_lock = Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_with_retries, job, output_root, args.timeout, args.step_limit, args.retries
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            with results_lock:
                results.append(result)
                _write_json(output_root / "results.json", results)
    _write_json(
        output_root / "batch_summary.json",
        {
            "benchmark_version": BENCHMARK_VERSION,
            "requested": len(jobs),
            "completed": sum(item.get("status") in {"completed", "succeeded"} for item in results),
            "retry_exhausted": sum(item.get("status") == "retry_exhausted" for item in results),
            "results": results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
