"""
Benchmark the low-level VLA (LAVA) policy on a single GPU.

Measures:
  1. Max batch size before GPU OOM
  2. Inference latency vs batch size (with JIT warmup)
  3. GPU memory usage vs batch size
  4. Preprocessing (TF image ops + CLIP tokenization) vs JAX forward time

Runs in the language-table conda env (ltvenv).

Usage:
    CUDA_VISIBLE_DEVICES=0 ltvenv/bin/python -m benchmarks.benchmark_vla \
        --checkpoint_dir /path/to/checkpoints/ \
        --batch_sizes 1,2,4,8,16,32,64,128,256,512 \
        --num_warmup 3 --num_iters 20
"""

import argparse
import gc
import logging
import os
import sys
import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

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


def get_jax_memory_mb():
    """Get JAX device memory stats."""
    try:
        devices = jax.devices()
        if devices:
            stats = devices[0].memory_stats()
            if stats:
                used = stats.get("bytes_in_use", 0) / 1024**2
                limit = stats.get("bytes_limit", 0) / 1024**2
                peak = stats.get("peak_bytes_in_use", 0) / 1024**2
                return used, limit, peak
    except Exception:
        pass
    return -1, -1, -1


def make_dummy_obs(batch_size, height=180, width=320):
    """Create dummy observations matching Language Table output."""
    obs_list = []
    for _ in range(batch_size):
        obs_list.append({
            "rgb": np.random.randint(0, 255, (height, width, 3), dtype=np.uint8),
        })
    return obs_list


def benchmark_vla_inference(policy, batch_size, num_warmup, num_iters):
    """Benchmark VLA inference at a given batch size.

    Returns dict with timing and memory stats, or None on OOM.
    """
    goals = [f"push the red block to the blue block"] * batch_size
    active_mask = np.ones(batch_size, dtype=bool)
    obs_list = make_dummy_obs(batch_size)

    policy.reset(num_envs=batch_size)

    # Warmup (includes JIT compilation on first call)
    for i in range(num_warmup):
        try:
            obs_list_iter = make_dummy_obs(batch_size)
            _ = policy.predict(goals, obs_list_iter, active_mask)
        except Exception as e:
            if "out of memory" in str(e).lower() or "resource" in str(e).lower():
                return None
            raise

    # Timed iterations
    preprocess_times = []
    forward_times = []
    total_times = []

    for i in range(num_iters):
        obs_list_iter = make_dummy_obs(batch_size)

        # --- Total time (preprocess + forward) ---
        t_total_start = time.perf_counter()

        # --- Preprocess: build batch (TF image ops + CLIP tokenization) ---
        t_pre_start = time.perf_counter()
        observation = policy._build_batch(goals, obs_list_iter, active_mask)
        t_pre_end = time.perf_counter()

        # --- Forward: JAX JIT inference ---
        t_fwd_start = time.perf_counter()
        actions = policy._forward_jit(policy.variables, observation)
        # Force synchronization (JAX is async by default)
        actions.block_until_ready()
        t_fwd_end = time.perf_counter()

        t_total_end = time.perf_counter()

        preprocess_times.append(t_pre_end - t_pre_start)
        forward_times.append(t_fwd_end - t_fwd_start)
        total_times.append(t_total_end - t_total_start)

    jax_used, jax_limit, jax_peak = get_jax_memory_mb()
    gpu_used, gpu_total = get_gpu_memory_mb()

    return {
        "batch_size": batch_size,
        "preprocess_ms_mean": np.mean(preprocess_times) * 1000,
        "preprocess_ms_std": np.std(preprocess_times) * 1000,
        "forward_ms_mean": np.mean(forward_times) * 1000,
        "forward_ms_std": np.std(forward_times) * 1000,
        "total_ms_mean": np.mean(total_times) * 1000,
        "total_ms_std": np.std(total_times) * 1000,
        "throughput_samples_per_sec": batch_size / np.mean(total_times),
        "gpu_used_mb": gpu_used,
        "gpu_total_mb": gpu_total,
        "jax_used_mb": jax_used,
        "jax_peak_mb": jax_peak,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark LAVA VLA inference")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing the LAVA Flax checkpoint")
    parser.add_argument("--checkpoint_prefix", type=str,
                        default="bc_resnet_sim_checkpoint_")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8,16,32,64,128,256,512",
                        help="Comma-separated batch sizes to test")
    parser.add_argument("--num_warmup", type=int, default=3,
                        help="Number of warmup iterations (includes JIT compile)")
    parser.add_argument("--num_iters", type=int, default=20,
                        help="Number of timed iterations per batch size")
    parser.add_argument("--xla_mem_fraction", type=float, default=0.9,
                        help="XLA_PYTHON_CLIENT_MEM_FRACTION (default 0.9 for benchmarking)")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Configure JAX memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_mem_fraction)

    # Suppress TF noise
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    from language_table.lamer.lava_policy import LAVAPolicy

    gpu_used_init, gpu_total = get_gpu_memory_mb()
    logger.info("GPU: %d MB used / %d MB total (before loading model)", gpu_used_init, gpu_total)

    logger.info("Loading LAVA policy from %s ...", args.checkpoint_dir)
    t0 = time.time()
    policy = LAVAPolicy(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
    )
    load_time = time.time() - t0
    gpu_used_model, _ = get_gpu_memory_mb()
    logger.info("Policy loaded in %.1fs (GPU: %d MB used)", load_time, gpu_used_model)

    # Print JAX device info
    devices = jax.devices()
    logger.info("JAX devices: %s", devices)
    logger.info("JAX default backend: %s", jax.default_backend())

    # Run benchmarks
    results = []
    max_batch_size = 0

    print("\n" + "=" * 100)
    print("VLA (LAVA) INFERENCE BENCHMARK")
    print("=" * 100)
    print(f"{'Batch':>6} | {'Preprocess':>12} | {'Forward':>12} | {'Total':>12} | "
          f"{'Throughput':>12} | {'GPU Used':>10} | {'JAX Peak':>10} | Status")
    print(f"{'Size':>6} | {'(ms)':>12} | {'(ms)':>12} | {'(ms)':>12} | "
          f"{'(samp/s)':>12} | {'(MB)':>10} | {'(MB)':>10} |")
    print("-" * 100)

    for bs in batch_sizes:
        try:
            result = benchmark_vla_inference(policy, bs, args.num_warmup, args.num_iters)
        except Exception as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "resource" in error_str or "alloc" in error_str:
                result = None
            else:
                logger.error("Unexpected error at batch_size=%d: %s", bs, e)
                result = None

        if result is None:
            print(f"{bs:>6} | {'---':>12} | {'---':>12} | {'---':>12} | "
                  f"{'---':>12} | {'---':>10} | {'---':>10} | OOM")
            break
        else:
            max_batch_size = bs
            results.append(result)
            print(f"{bs:>6} | {result['preprocess_ms_mean']:>9.1f}±{result['preprocess_ms_std']:<3.0f}"
                  f"| {result['forward_ms_mean']:>9.1f}±{result['forward_ms_std']:<3.0f}"
                  f"| {result['total_ms_mean']:>9.1f}±{result['total_ms_std']:<3.0f}"
                  f"| {result['throughput_samples_per_sec']:>12.1f}"
                  f" | {result['gpu_used_mb']:>10}"
                  f" | {result['jax_peak_mb']:>10.0f}"
                  f" | OK")

        # Force garbage collection between sizes
        gc.collect()

    print("=" * 100)
    print(f"\nMax successful batch size: {max_batch_size}")

    if results:
        # Find optimal batch size (best throughput)
        best = max(results, key=lambda r: r["throughput_samples_per_sec"])
        print(f"Best throughput: {best['throughput_samples_per_sec']:.1f} samples/s at batch_size={best['batch_size']}")
        print(f"At best throughput: preprocess={best['preprocess_ms_mean']:.1f}ms, forward={best['forward_ms_mean']:.1f}ms")

    # Summary for the training setup
    print("\n" + "=" * 100)
    print("IMPLICATIONS FOR TRAINING")
    print("=" * 100)
    if results:
        # Current setup: 16 train envs * 8 group = 128 workers, 5 inner steps
        for target_envs in [16, 32, 64, 128, 256]:
            # Find closest batch size
            matching = [r for r in results if r["batch_size"] >= target_envs]
            if matching:
                r = min(matching, key=lambda x: x["batch_size"])
                inner_step_ms = r["total_ms_mean"] * (target_envs / r["batch_size"])
                for inner_steps in [5, 10, 25, 50, 100]:
                    total_s = inner_step_ms * inner_steps / 1000
                    print(f"  {target_envs:>4} envs × {inner_steps:>3} inner steps "
                          f"= {total_s:>6.1f}s VLA time (batch inference at bs={r['batch_size']})")
            else:
                print(f"  {target_envs:>4} envs: exceeds max tested batch size")


if __name__ == "__main__":
    main()
