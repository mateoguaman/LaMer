"""Standalone benchmark: throughput of parallel PyBullet environments via Ray.

Measures raw environment throughput (steps/sec) as a function of the number of
parallel workers, completely decoupled from LLM / training compute.  Use this
to find the sweet spot for ``env_num`` on your machine before committing to a
full training run.

Install prerequisites
---------------------
    pip install pybullet pybullet_envs ray[default]

    # For newer gym / pybullet-gymnasium:
    pip install pybullet pybullet-gymnasium ray[default]

    # psutil is optional but enables per-worker memory reporting:
    pip install psutil

Usage
-----
    # Quick smoke-test with CartPole (no pybullet needed):
    python -m examples.benchmark_pybullet_envs --env_id CartPole-v1

    # Benchmark AntBullet up to 64 parallel envs, 200 steps each:
    python -m examples.benchmark_pybullet_envs \\
        --env_id AntBulletEnv-v0 \\
        --max_envs 64 \\
        --num_steps 200

    # Test a custom sweep:
    python -m examples.benchmark_pybullet_envs \\
        --env_id HalfCheetahBulletEnv-v0 \\
        --env_counts 1 4 16 32 64 128 \\
        --num_steps 100 \\
        --warmup_steps 20

Output
------
    PyBullet Env Throughput Benchmark
    Environment : AntBulletEnv-v0
    Steps/worker: 200  |  Warmup: 10

     num_envs | steps/sec |  mem_MB | efficiency
    ----------+-----------+---------+-----------
            1 |     142.3 |     180 |   100.0 %
            2 |     278.1 |     360 |    97.7 %
            4 |     540.6 |     720 |    95.1 %
            8 |    1023.4 |    1440 |    90.0 %
           16 |    1854.8 |    2880 |    81.5 %
           32 |    2901.2 |    5760 |    63.7 %
           64 |    3402.7 |   11520 |    37.4 %  <- below 50% efficiency
    ----------+-----------+---------+-----------
    Recommended max_envs : 32  (last point with efficiency >= 50 %)
    Peak throughput      : 3402 steps/sec at 64 envs
"""

import argparse
import time
import sys
from typing import List, Optional, Dict, Any

import numpy as np
import ray


# ---------------------------------------------------------------------------
# Remote worker (self-contained so this script can run without installing the
# agent_system package)
# ---------------------------------------------------------------------------
@ray.remote(num_cpus=0.5)
class _BenchWorker:
    """Minimal Ray actor holding a single gym environment."""

    def __init__(self, env_id: str, seed: int, env_kwargs: Dict[str, Any]):
        # Register pybullet gym environments if the package is available
        try:
            import pybullet_envs  # noqa: F401
        except ImportError:
            pass

        import gym
        self.env = gym.make(env_id, **env_kwargs)
        self.env.reset(seed=seed) if hasattr(self.env, "reset") else self.env.reset()

    def action_space_info(self) -> Dict[str, Any]:
        import gym.spaces as spaces
        sp = self.env.action_space
        if isinstance(sp, spaces.Box):
            return {
                "type": "Box",
                "shape": list(sp.shape),
                "low": sp.low.tolist(),
                "high": sp.high.tolist(),
            }
        if isinstance(sp, spaces.Discrete):
            return {"type": "Discrete", "n": int(sp.n)}
        return {"type": type(sp).__name__}

    def reset(self, seed: Optional[int] = None):
        kwargs = {"seed": seed} if seed is not None else {}
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, _ = result
        else:
            obs = result
        return obs

    def step(self, action):
        result = self.env.step(np.array(action, dtype=np.float32))
        if len(result) == 5:
            obs, reward, terminated, truncated, _ = result
            done = terminated or truncated
        else:
            obs, reward, done, _ = result
        if done:
            # Auto-reset so benchmarks run continuously
            reset_result = self.env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        return obs, float(reward), bool(done)

    def memory_rss_mb(self) -> float:
        """Return resident-set size of this worker process in MB."""
        try:
            import psutil, os
            proc = psutil.Process(os.getpid())
            return proc.memory_info().rss / 1024 / 1024
        except ImportError:
            return float("nan")

    def close(self):
        self.env.close()


# ---------------------------------------------------------------------------
# Random-action generator (runs in the driver, not the workers)
# ---------------------------------------------------------------------------
def _random_actions(space_info: Dict, n: int) -> List:
    if space_info["type"] == "Box":
        low = np.array(space_info["low"], dtype=np.float32)
        high = np.array(space_info["high"], dtype=np.float32)
        shape = tuple(space_info["shape"])
        return [np.random.uniform(low, high, shape).tolist() for _ in range(n)]
    if space_info["type"] == "Discrete":
        return [int(np.random.randint(space_info["n"])) for _ in range(n)]
    raise NotImplementedError(f"Unsupported action space: {space_info['type']}")


# ---------------------------------------------------------------------------
# Single-point benchmark
# ---------------------------------------------------------------------------
def _benchmark_n_envs(
    env_id: str,
    n: int,
    num_steps: int,
    warmup_steps: int,
    env_kwargs: Dict,
    base_seed: int = 0,
) -> Dict[str, Any]:
    """Create n workers, run warmup + benchmark, return metrics, then kill workers."""
    workers = []
    try:
        workers = [
            _BenchWorker.remote(env_id, base_seed + i, env_kwargs)
            for i in range(n)
        ]

        # Fetch action-space info once
        space_info = ray.get(workers[0].action_space_info.remote())

        # Warmup (not timed)
        for _ in range(warmup_steps):
            actions = _random_actions(space_info, n)
            futures = [w.step.remote(a) for w, a in zip(workers, actions)]
            ray.get(futures)

        # Timed benchmark
        t0 = time.perf_counter()
        for _ in range(num_steps):
            actions = _random_actions(space_info, n)
            futures = [w.step.remote(a) for w, a in zip(workers, actions)]
            ray.get(futures)
        elapsed = time.perf_counter() - t0

        total_steps = n * num_steps
        steps_per_sec = total_steps / elapsed

        # Memory (sum across all workers)
        mem_futures = [w.memory_rss_mb.remote() for w in workers]
        mem_vals = ray.get(mem_futures)
        total_mem_mb = sum(v for v in mem_vals if not np.isnan(v))
        has_mem = not any(np.isnan(v) for v in mem_vals)

        return {
            "n": n,
            "steps_per_sec": steps_per_sec,
            "total_mem_mb": total_mem_mb if has_mem else float("nan"),
            "error": None,
        }

    except Exception as exc:
        return {"n": n, "steps_per_sec": float("nan"), "total_mem_mb": float("nan"), "error": str(exc)}

    finally:
        for w in workers:
            try:
                ray.kill(w)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark parallel PyBullet (or any gym) environment throughput with Ray.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env_id",
        default="AntBulletEnv-v0",
        help="Gym environment ID to benchmark (default: AntBulletEnv-v0). "
             "Use CartPole-v1 for a quick sanity-check without pybullet.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of timed steps per worker per trial (default: 100).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Un-timed warmup steps before measurement (default: 10).",
    )
    parser.add_argument(
        "--max_envs",
        type=int,
        default=128,
        help="Maximum number of parallel envs to try (default: 128). "
             "Sweep stops early if efficiency drops below --min_efficiency.",
    )
    parser.add_argument(
        "--env_counts",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of env counts to test, e.g. --env_counts 1 4 16 32. "
             "Overrides --max_envs and the default power-of-2 sweep.",
    )
    parser.add_argument(
        "--min_efficiency",
        type=float,
        default=0.30,
        help="Stop the sweep when throughput efficiency drops below this fraction "
             "(default: 0.30, i.e. 30%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0).",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=None,
        help="Total CPUs to allocate for Ray (default: auto-detect).",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Init Ray
    # -------------------------------------------------------------------
    ray_init_kwargs: Dict[str, Any] = {"log_to_driver": False, "ignore_reinit_error": True}
    if args.num_cpus is not None:
        ray_init_kwargs["num_cpus"] = args.num_cpus
    ray.init(**ray_init_kwargs)

    # -------------------------------------------------------------------
    # Validate that the environment can be created
    # -------------------------------------------------------------------
    print(f"\nValidating environment '{args.env_id}' ...", end=" ", flush=True)
    probe = _BenchWorker.remote(args.env_id, args.seed, {})
    try:
        space_info = ray.get(probe.action_space_info.remote())
        print(f"OK  (action space: {space_info})")
    except Exception as exc:
        print(f"\nERROR: Could not create environment '{args.env_id}'.")
        print(f"  {exc}")
        if "AntBullet" in args.env_id or "Bullet" in args.env_id:
            print(
                "\nHint: install PyBullet environments with:\n"
                "    pip install pybullet pybullet_envs\n"
                "or try: --env_id CartPole-v1"
            )
        ray.shutdown()
        sys.exit(1)
    finally:
        try:
            ray.kill(probe)
        except Exception:
            pass

    # -------------------------------------------------------------------
    # Determine sweep counts
    # -------------------------------------------------------------------
    if args.env_counts:
        counts = sorted(set(args.env_counts))
    else:
        counts = []
        c = 1
        while c <= args.max_envs:
            counts.append(c)
            c *= 2

    # -------------------------------------------------------------------
    # Print header
    # -------------------------------------------------------------------
    import os
    try:
        import psutil
        total_mem_gb = psutil.virtual_memory().total / 1024 ** 3
        mem_str = f"{total_mem_gb:.0f} GB RAM"
    except ImportError:
        mem_str = "RAM unknown (install psutil)"

    ray_cpus = ray.available_resources().get("CPU", 0)

    print(f"\nPyBullet Env Throughput Benchmark")
    print(f"  Environment  : {args.env_id}")
    print(f"  Steps/worker : {args.num_steps}  |  Warmup: {args.warmup_steps}")
    print(f"  Machine      : {ray_cpus:.0f} Ray CPUs, {mem_str}")
    print(f"  Sweep        : {counts}\n")

    col_w = 10
    header = (
        f"{'num_envs':>{col_w}} | {'steps/sec':>{col_w}} | {'mem_MB':>{col_w}} | {'efficiency':>{col_w}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    # -------------------------------------------------------------------
    # Sweep
    # -------------------------------------------------------------------
    results = []
    baseline_sps: Optional[float] = None
    recommended: Optional[int] = None
    peak_sps = 0.0
    peak_n = 0

    for n in counts:
        result = _benchmark_n_envs(
            env_id=args.env_id,
            n=n,
            num_steps=args.num_steps,
            warmup_steps=args.warmup_steps,
            env_kwargs={},
            base_seed=args.seed,
        )

        if result["error"]:
            flag = "  OOM/ERROR"
            print(f"{n:>{col_w}} | {'ERROR':>{col_w}} | {'---':>{col_w}} | {'---':>{col_w}}{flag}")
            print(f"    └─ {result['error'][:80]}")
            break

        sps = result["steps_per_sec"]
        mem = result["total_mem_mb"]
        mem_str_val = f"{mem:>{col_w}.0f}" if not np.isnan(mem) else f"{'n/a':>{col_w}}"

        if baseline_sps is None:
            baseline_sps = sps
            efficiency = 1.0
        else:
            efficiency = sps / (baseline_sps * n)

        if sps > peak_sps:
            peak_sps = sps
            peak_n = n

        if efficiency >= args.min_efficiency:
            recommended = n

        flag = ""
        if efficiency < args.min_efficiency:
            flag = f"  <- below {args.min_efficiency * 100:.0f}% efficiency"

        print(
            f"{n:>{col_w}} | {sps:>{col_w}.1f} | {mem_str_val} | {efficiency * 100:>{col_w}.1f} %{flag}"
        )

        results.append({**result, "efficiency": efficiency})

    print(sep)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print()
    if recommended is not None:
        print(f"  Recommended max_envs : {recommended}  "
              f"(last point with efficiency >= {args.min_efficiency * 100:.0f} %)")
    else:
        print(f"  No recommended value found — efficiency was below {args.min_efficiency * 100:.0f}% from the start.")

    if peak_sps > 0:
        print(f"  Peak throughput      : {peak_sps:.0f} steps/sec at {peak_n} envs")

    print()
    print("Notes:")
    print("  - 'steps/sec' = total env transitions per second across all parallel workers.")
    print("  - 'efficiency' = actual_throughput / (1-env_throughput * num_envs).")
    print("    100% = perfect linear scaling; drops with serialisation overhead,")
    print("    CPU contention, or memory pressure.")
    print("  - mem_MB is the sum of RSS for all worker processes (requires psutil).")
    print("  - For LLM training, set env_num = recommended_max_envs in your config.")

    ray.shutdown()


if __name__ == "__main__":
    main()
