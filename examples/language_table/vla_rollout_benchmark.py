#!/usr/bin/env python3
"""
Benchmark Language Table VLA rollouts through the remote environment server.

The server should be started with language_table.lamer.server_main using either
--policy smolvla or --policy lava. This client resets the remote vectorized env,
uses each environment's task instruction as the VLA goal, runs batched rollout
steps, and reports wall-clock plus server-side timing when available.

Usage:
    python examples/language_table/vla_rollout_benchmark.py \
        --remote_address localhost:50053 \
        --num_batches 5 \
        --max_turns 1 \
        --output results/vla_benchmark.jsonl
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark batched Language Table VLA rollouts."
    )
    parser.add_argument(
        "--remote_address",
        required=True,
        help="HOST:PORT of the Language Table env server",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=3,
        help="Number of vectorized reset+rollout batches to run",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=1,
        help="Max outer-loop env.step calls per reset batch",
    )
    parser.add_argument(
        "--goal_source",
        choices=["task", "observation", "fixed"],
        default="task",
        help="Goal string to send to the VLA inner loop",
    )
    parser.add_argument(
        "--fixed_goal",
        default="push the blocks according to the task",
        help="Goal string used when --goal_source=fixed",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Remote socket timeout in seconds",
    )
    parser.add_argument(
        "--expected_num_envs",
        type=int,
        default=None,
        help="Fail if the remote server reports a different parallel env count",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSONL path for per-batch records and final summary",
    )
    return parser.parse_args()


def build_env(remote_address: str, timeout: float):
    from agent_system.environments.remote.client import RemoteEnvironmentManager

    return RemoteEnvironmentManager(remote_address, timeout=timeout)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _task_from_observation(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Task:"):
            task = line.split("Task:", 1)[1].strip()
            if task:
                return task
    return text.strip() or "push the blocks according to the task"


def build_goals(obs: Dict[str, Any], goal_source: str, fixed_goal: str) -> List[str]:
    text_obs = obs.get("text") or obs.get("anchor") or []
    if isinstance(text_obs, str):
        text_obs = [text_obs]

    if goal_source == "fixed":
        return [fixed_goal] * len(text_obs)
    if goal_source == "observation":
        return [str(text).strip() or fixed_goal for text in text_obs]
    return [_task_from_observation(str(text)) for text in text_obs]


def first_benchmark_payload(infos: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for info in infos:
        payload = info.get("benchmark_env") if isinstance(info, dict) else None
        if payload:
            return payload
    return None


def summarize_server_timings(records: List[Dict[str, Any]]) -> Dict[str, float]:
    totals = defaultdict(float)
    counts = defaultdict(int)

    for record in records:
        for payload in record.get("server_timings", []):
            if not payload:
                continue
            for key, value in payload.items():
                if key.endswith("_s") and isinstance(value, (int, float)):
                    totals[key] += float(value)
                    counts[key] += 1

    summary = {}
    for key, total in sorted(totals.items()):
        summary[f"{key}_sum"] = total
        summary[f"{key}_mean"] = total / counts[key]
    return summary


def run_batch(env, batch_idx: int, max_turns: int, goal_source: str, fixed_goal: str):
    reset_t0 = time.perf_counter()
    obs, infos = env.reset()
    reset_wall_s = time.perf_counter() - reset_t0

    num_envs = len(obs.get("text") or obs.get("anchor") or [])
    server_timings = []
    reset_benchmark = first_benchmark_payload(infos)
    if reset_benchmark:
        server_timings.append(_jsonable(reset_benchmark))

    total_rewards = np.zeros(num_envs, dtype=np.float64)
    done = np.zeros(num_envs, dtype=bool)
    turns_completed = 0
    step_wall_times = []

    for turn_idx in range(max_turns):
        goals = build_goals(obs, goal_source, fixed_goal)
        if len(goals) != num_envs:
            raise RuntimeError(
                f"Built {len(goals)} goals for {num_envs} environments"
            )

        step_t0 = time.perf_counter()
        obs, rewards, dones, infos = env.step(goals, phase="play")
        step_wall_s = time.perf_counter() - step_t0
        step_wall_times.append(step_wall_s)
        turns_completed += 1

        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=bool)
        total_rewards += rewards
        done |= dones

        step_benchmark = first_benchmark_payload(infos)
        if step_benchmark:
            server_timings.append(_jsonable(step_benchmark))

        print(
            f"batch={batch_idx} turn={turn_idx} "
            f"wall_s={step_wall_s:.3f} done={int(done.sum())}/{num_envs} "
            f"mean_reward={float(rewards.mean()):.3f}"
        )
        if done.all():
            break

    batch_wall_s = reset_wall_s + sum(step_wall_times)
    return {
        "batch": batch_idx,
        "num_envs": num_envs,
        "turns_completed": turns_completed,
        "reset_wall_s": reset_wall_s,
        "step_wall_s": step_wall_times,
        "batch_wall_s": batch_wall_s,
        "env_rollouts": num_envs,
        "env_turns": num_envs * turns_completed,
        "success_rate": float(done.mean()) if num_envs else 0.0,
        "mean_total_reward": float(total_rewards.mean()) if num_envs else 0.0,
        "server_timings": server_timings,
    }


def main():
    args = parse_args()
    env = build_env(args.remote_address, args.timeout)
    if args.expected_num_envs is not None and env.num_processes != args.expected_num_envs:
        env.close()
        raise SystemExit(
            f"Remote server has {env.num_processes} envs, expected "
            f"{args.expected_num_envs}"
        )

    print(
        f"Connected to {args.remote_address}: "
        f"num_envs={env.num_processes}, server_max_turns={env.max_turns}"
    )

    records = []
    total_t0 = time.perf_counter()
    try:
        for batch_idx in range(args.num_batches):
            record = run_batch(
                env=env,
                batch_idx=batch_idx,
                max_turns=min(args.max_turns, env.max_turns),
                goal_source=args.goal_source,
                fixed_goal=args.fixed_goal,
            )
            records.append(record)
        total_wall_s = time.perf_counter() - total_t0
    finally:
        env.close()

    total_env_rollouts = sum(r["env_rollouts"] for r in records)
    total_env_turns = sum(r["env_turns"] for r in records)
    summary = {
        "summary": True,
        "remote_address": args.remote_address,
        "num_batches": len(records),
        "num_envs": env.num_processes,
        "max_turns": args.max_turns,
        "goal_source": args.goal_source,
        "total_wall_s": total_wall_s,
        "total_env_rollouts": total_env_rollouts,
        "total_env_turns": total_env_turns,
        "env_rollouts_per_s": (
            total_env_rollouts / total_wall_s if total_wall_s > 0 else 0.0
        ),
        "env_turns_per_s": total_env_turns / total_wall_s if total_wall_s > 0 else 0.0,
        "mean_batch_wall_s": (
            sum(r["batch_wall_s"] for r in records) / len(records) if records else 0.0
        ),
        "mean_success_rate": (
            sum(r["success_rate"] for r in records) / len(records) if records else 0.0
        ),
        "mean_total_reward": (
            sum(r["mean_total_reward"] for r in records) / len(records)
            if records
            else 0.0
        ),
    }
    summary.update(summarize_server_timings(records))

    print("\n=== VLA Rollout Benchmark Summary ===")
    print(f"  batches           : {summary['num_batches']}")
    print(f"  parallel envs     : {summary['num_envs']}")
    print(f"  total wall time   : {summary['total_wall_s']:.3f}s")
    print(f"  env rollouts/sec  : {summary['env_rollouts_per_s']:.3f}")
    print(f"  env turns/sec     : {summary['env_turns_per_s']:.3f}")
    print(f"  mean batch wall   : {summary['mean_batch_wall_s']:.3f}s")
    print(f"  mean success rate : {summary['mean_success_rate']:.3f}")
    if "server_elapsed_s_sum" in summary:
        print(f"  server elapsed    : {summary['server_elapsed_s_sum']:.3f}s")
    if "vla_total_s_sum" in summary:
        print(f"  VLA total         : {summary['vla_total_s_sum']:.3f}s")
    if "env_step_total_s_sum" in summary:
        print(f"  env step total    : {summary['env_step_total_s_sum']:.3f}s")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(_jsonable(record)) + "\n")
            f.write(json.dumps(_jsonable(summary)) + "\n")
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
