#!/usr/bin/env python3
"""Build a sweep manifest from end-to-end benchmark run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUMMARY_KEYS = (
    "rollout_elapsed_s_mean",
    "rollout_elapsed_s_p50",
    "rollout_elapsed_s_p95",
    "env_server_elapsed_s_mean",
    "env_server_elapsed_s_p50",
    "env_server_elapsed_s_p95",
    "transport_overhead_s_mean",
    "transport_overhead_s_p50",
    "transport_overhead_s_p95",
    "vla_total_s_sum",
    "env_step_total_s_sum",
    "ray_dispatch_s_sum",
    "ray_collect_s_sum",
    "ray_unpack_s_sum",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_triage(best_run: dict[str, Any]) -> dict[str, Any]:
    dispatch = float(best_run.get("ray_dispatch_s_sum", 0.0) or 0.0)
    collect = float(best_run.get("ray_collect_s_sum", 0.0) or 0.0)
    unpack = float(best_run.get("ray_unpack_s_sum", 0.0) or 0.0)
    transport = float(best_run.get("transport_overhead_s_mean", 0.0) or 0.0)

    if dispatch <= 0.0 and collect <= 0.0 and unpack <= 0.0 and transport <= 0.0:
        return {
            "priority": "insufficient_data",
            "reason": (
                "Missing transport timing fields; rerun the benchmark with env-side "
                "timing summaries before backlog triage."
            ),
        }

    transport_work = collect + unpack + transport
    if dispatch > transport_work:
        return {
            "priority": "transport_then_dispatch",
            "reason": (
                "Dispatch is material, but the current workplan keeps transport "
                "reduction ahead of scheduler tweaks until transport is remeasured."
            ),
        }

    return {
        "priority": "transport_reduction",
        "reason": (
            "Transport, collect, and unpack work dominate dispatch; prioritize "
            "smaller returned observations or shared-memory transfer first."
        ),
    }


def collect_run_entry(run_dir: Path) -> dict[str, Any]:
    launcher = _load_json(run_dir / "launcher_metadata.json")
    summary = _load_json(run_dir / "artifacts" / "benchmark_summary.json")

    entry = {
        "run_dir": str(run_dir),
        "launcher_metadata_path": str(run_dir / "launcher_metadata.json"),
        "summary_path": str(run_dir / "artifacts" / "benchmark_summary.json"),
        "train_shard_count": int(launcher["train_shard_count"]),
        "train_envs_per_shard": launcher.get("train_envs_per_shard"),
        "env_server_gpus": launcher.get("env_server_gpus", []),
        "skip_val_server": bool(launcher.get("skip_val_server", False)),
        "val_server_enabled": bool(launcher.get("val_server_enabled", False)),
        "preprocess_mode": launcher.get("preprocess_mode"),
    }
    for key in SUMMARY_KEYS:
        if key in summary:
            entry[key] = summary[key]
    return entry


def build_sweep_manifest(root_dir: Path) -> dict[str, Any]:
    run_dirs = sorted(
        [path for path in root_dir.iterdir() if path.is_dir()],
        key=lambda path: collect_run_entry(path)["train_shard_count"],
    )
    runs = [collect_run_entry(run_dir) for run_dir in run_dirs]

    best_run = None
    best_rollout = None
    for run in runs:
        rollout_mean = run.get("rollout_elapsed_s_mean")
        if rollout_mean is None:
            continue
        rollout_value = float(rollout_mean)
        if best_rollout is None or rollout_value < best_rollout:
            best_rollout = rollout_value
            best_run = run

    manifest = {
        "root_dir": str(root_dir),
        "run_count": int(len(runs)),
        "runs": runs,
    }
    if best_run is not None:
        manifest["best_rollout_run"] = {
            "train_shard_count": best_run["train_shard_count"],
            "rollout_elapsed_s_mean": best_rollout,
            "summary_path": best_run["summary_path"],
        }
        manifest["post_fused_backlog_triage"] = _build_triage(best_run)
    else:
        manifest["post_fused_backlog_triage"] = {
            "priority": "insufficient_data",
            "reason": "No run exposed rollout_elapsed_s_mean; unable to rank shard counts.",
        }
    return manifest


def write_sweep_manifest(root_dir: Path) -> Path:
    manifest = build_sweep_manifest(root_dir)
    output_path = root_dir / "sweep_manifest.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    args = parser.parse_args()

    output_path = write_sweep_manifest(args.root_dir)
    print(f"Wrote sweep manifest to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
