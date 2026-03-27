"""
Benchmark Language Table environment scaling with VLA inner loop.

Measures:
  1. End-to-end step time (VLA inference + PyBullet sim) vs number of envs
  2. CPU utilization vs number of envs
  3. Breakdown: VLA inference time vs env.step() time
  4. Optimal number of envs for a given CPU budget

Runs in the language-table conda env (ltvenv).

Usage:
    CUDA_VISIBLE_DEVICES=0 ltvenv/bin/python -m benchmarks.benchmark_envs \
        --checkpoint_dir /path/to/checkpoints/ \
        --env_counts 4,8,16,32,64,128 \
        --inner_steps 20 --cpus_per_gpu 8
"""

import argparse
import gc
import logging
import os
import time

import numpy as np
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_gpu_memory_mb():
    """Get current GPU memory usage via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits",
             f"--id={os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().split(",")
            return int(used.strip()), int(total.strip())
    except Exception:
        pass
    return -1, -1


def decode_instruction(instruction_array):
    arr = np.asarray(instruction_array)
    non_zero = arr[arr != 0]
    if non_zero.shape[0] == 0:
        return ""
    return bytes(non_zero.tolist()).decode("utf-8")


def benchmark_env_scaling(num_envs, checkpoint_dir, checkpoint_prefix,
                          inner_steps, block_mode, seed):
    """Run a full benchmark for a given number of environments."""
    import ray
    if not ray.is_initialized():
        ray.init()

    from language_table.environments.rewards.block2block import BlockToBlockReward
    from language_table.lamer.envs import LanguageTableMultiProcessEnv
    from language_table.lamer.lava_policy import LAVAPolicy

    # Create envs
    logger.info("Creating %d environments...", num_envs)
    t0 = time.time()
    envs = LanguageTableMultiProcessEnv(
        num_envs=num_envs,
        block_mode=block_mode,
        reward_factory_cls=BlockToBlockReward,
        seed=seed,
        group_n=1,
        return_full_state=True,
        render_obs=True,
    )
    env_create_time = time.time() - t0
    logger.info("Envs created in %.1fs", env_create_time)

    # Load VLA policy
    logger.info("Loading LAVA policy...")
    t0 = time.time()
    policy = LAVAPolicy(
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
    )
    policy_load_time = time.time() - t0

    # Reset
    logger.info("Resetting environments...")
    t0 = time.time()
    obs_list, infos = envs.reset()
    reset_time = time.time() - t0

    instructions = []
    for obs in obs_list:
        instr = decode_instruction(obs.get("instruction", []))
        instructions.append(instr)

    policy.reset(num_envs=num_envs)
    active_mask = np.ones(num_envs, dtype=bool)

    # Run inner loop with detailed timing
    vla_times = []
    env_step_times = []
    total_step_times = []
    cpu_percents = []

    for step in range(inner_steps):
        cpu_before = psutil.cpu_percent(interval=None)

        t_total = time.perf_counter()

        # VLA inference
        t_vla = time.perf_counter()
        actions = policy.predict(instructions, obs_list, active_mask)
        vla_time = time.perf_counter() - t_vla

        # Environment step
        t_env = time.perf_counter()
        obs_list, rewards, dones, step_infos = envs.step(actions, active_mask=active_mask)
        env_time = time.perf_counter() - t_env

        total_time = time.perf_counter() - t_total
        cpu_after = psutil.cpu_percent(interval=None)

        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=bool)
        active_mask &= ~dones

        # Skip first step (JIT warmup)
        if step > 0:
            vla_times.append(vla_time)
            env_step_times.append(env_time)
            total_step_times.append(total_time)
            cpu_percents.append(cpu_after)

        if not active_mask.any():
            break

    envs.close()

    gpu_used, gpu_total = get_gpu_memory_mb()

    return {
        "num_envs": num_envs,
        "env_create_time_s": env_create_time,
        "reset_time_s": reset_time,
        "steps_completed": len(total_step_times),
        "vla_ms_mean": np.mean(vla_times) * 1000 if vla_times else 0,
        "vla_ms_p95": np.percentile(vla_times, 95) * 1000 if vla_times else 0,
        "env_step_ms_mean": np.mean(env_step_times) * 1000 if env_step_times else 0,
        "env_step_ms_p95": np.percentile(env_step_times, 95) * 1000 if env_step_times else 0,
        "total_ms_mean": np.mean(total_step_times) * 1000 if total_step_times else 0,
        "total_ms_p95": np.percentile(total_step_times, 95) * 1000 if total_step_times else 0,
        "throughput_envsteps_per_sec": num_envs / np.mean(total_step_times) if total_step_times else 0,
        "cpu_percent_mean": np.mean(cpu_percents) if cpu_percents else 0,
        "gpu_used_mb": gpu_used,
        "gpu_total_mb": gpu_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Language Table env scaling")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoint_prefix", type=str,
                        default="bc_resnet_sim_checkpoint_")
    parser.add_argument("--env_counts", type=str, default="4,8,16,32,64,128",
                        help="Comma-separated env counts to test")
    parser.add_argument("--inner_steps", type=int, default=20,
                        help="Inner-loop steps per benchmark run")
    parser.add_argument("--block_mode", type=str, default="BLOCK_4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpus_per_gpu", type=int, default=8,
                        help="CPU budget per GPU (for analysis)")
    parser.add_argument("--time_budget_s", type=float, default=30.0,
                        help="Max acceptable outer-step time in seconds")
    args = parser.parse_args()

    env_counts = [int(x) for x in args.env_counts.split(",")]

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    total_cpus = psutil.cpu_count(logical=True)
    logger.info("System CPUs: %d (budget per GPU: %d)", total_cpus, args.cpus_per_gpu)

    results = []

    print("\n" + "=" * 130)
    print("ENVIRONMENT SCALING BENCHMARK")
    print("=" * 130)
    print(f"{'Envs':>6} | {'Create':>8} | {'Reset':>8} | "
          f"{'VLA Infer':>12} | {'Env Step':>12} | {'Total Step':>12} | "
          f"{'Throughput':>12} | {'CPU %':>7} | {'GPU MB':>8} | "
          f"{'5-step':>8} | {'100-step':>9}")
    print(f"{'':>6} | {'(s)':>8} | {'(s)':>8} | "
          f"{'(ms)':>12} | {'(ms)':>12} | {'(ms)':>12} | "
          f"{'(env-step/s)':>12} | {'':>7} | {'':>8} | "
          f"{'(s)':>8} | {'(s)':>9}")
    print("-" * 130)

    for n in env_counts:
        try:
            r = benchmark_env_scaling(
                n, args.checkpoint_dir, args.checkpoint_prefix,
                args.inner_steps, args.block_mode, args.seed,
            )
            results.append(r)

            time_5 = r["total_ms_mean"] * 5 / 1000
            time_100 = r["total_ms_mean"] * 100 / 1000
            status = "OK" if time_5 < args.time_budget_s else "SLOW"

            print(f"{n:>6} | {r['env_create_time_s']:>8.1f} | {r['reset_time_s']:>8.1f} | "
                  f"{r['vla_ms_mean']:>8.1f}±{r['vla_ms_p95'] - r['vla_ms_mean']:>3.0f}"
                  f" | {r['env_step_ms_mean']:>8.1f}±{r['env_step_ms_p95'] - r['env_step_ms_mean']:>3.0f}"
                  f" | {r['total_ms_mean']:>8.1f}±{r['total_ms_p95'] - r['total_ms_mean']:>3.0f}"
                  f" | {r['throughput_envsteps_per_sec']:>12.1f}"
                  f" | {r['cpu_percent_mean']:>6.1f}%"
                  f" | {r['gpu_used_mb']:>8}"
                  f" | {time_5:>8.1f} | {time_100:>9.1f}  {status}")

        except Exception as e:
            logger.error("Failed at num_envs=%d: %s", n, e)
            print(f"{n:>6} | FAILED: {e}")

        gc.collect()

    print("=" * 130)

    # Analysis for the training setup
    print("\n" + "=" * 100)
    print("ANALYSIS FOR TRAINING SETUP")
    print("=" * 100)

    if results:
        print(f"\nCPU budget per GPU: {args.cpus_per_gpu}")
        print(f"Time budget per outer step: {args.time_budget_s}s")
        print()

        # With current config: 5 inner steps
        for inner_steps in [5, 10, 25, 50, 100]:
            print(f"\n  With max_inner_steps={inner_steps}:")
            for r in results:
                outer_step_time = r["total_ms_mean"] * inner_steps / 1000
                within_budget = outer_step_time < args.time_budget_s
                marker = "<<< RECOMMENDED" if (within_budget and
                    (r == results[-1] or results[results.index(r)+1]["total_ms_mean"] * inner_steps / 1000 >= args.time_budget_s)
                ) else ""
                print(f"    {r['num_envs']:>4} envs: {outer_step_time:>6.1f}s per outer step "
                      f"({'OK' if within_budget else 'SLOW'}) {marker}")

        # Find bottleneck
        print("\n  Bottleneck analysis (what dominates step time):")
        for r in results:
            vla_pct = r["vla_ms_mean"] / r["total_ms_mean"] * 100 if r["total_ms_mean"] > 0 else 0
            env_pct = r["env_step_ms_mean"] / r["total_ms_mean"] * 100 if r["total_ms_mean"] > 0 else 0
            bottleneck = "VLA" if vla_pct > env_pct else "ENV"
            print(f"    {r['num_envs']:>4} envs: VLA={vla_pct:.0f}% Env={env_pct:.0f}% → bottleneck={bottleneck}")


if __name__ == "__main__":
    main()
