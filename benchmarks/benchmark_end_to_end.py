#!/usr/bin/env python3
"""Run an end-to-end LaMer Language Table benchmark.

This benchmark intentionally uses the real remote env-server boundary from
training. It can either:

1. attach to already-running `language_table.lamer.server_main` processes, or
2. launch matching `server_main` processes locally, then run the normal
   `verl.trainer.main_ppo` entrypoint with benchmark-specific overrides.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANGTABLE_DIR = REPO_ROOT.parent / "language-table"
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

PRESETS = {
    "doc_baseline": {
        "train_num_envs": 16,
        "val_num_envs": 128,
        "group_size": 8,
        "num_attempts": 2,
        "max_turns": 4,
        "max_inner_steps": 96,
    },
    "resolved_training_config": {
        "train_num_envs": 16,
        "val_num_envs": 128,
        "group_size": 8,
        "num_attempts": 3,
        "max_turns": 4,
        "max_inner_steps": 96,
    },
}


def _hydra_list(items: list[str]) -> str:
    quoted = ",".join(json.dumps(item) for item in items)
    return f"[{quoted}]"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((host, port))
            except OSError:
                time.sleep(1.0)
                continue
            return
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def _build_server_cmd(
    args,
    port: int,
    num_envs: int,
    group_n: int,
    split: str,
) -> list[str]:
    cmd = [
        args.langtable_python,
        "-m",
        "language_table.lamer.server_main",
        "--host",
        args.host,
        "--port",
        str(port),
        "--num_envs",
        str(num_envs),
        "--group_n",
        str(group_n),
        "--block_mode",
        args.block_mode,
        "--max_inner_steps",
        str(args.max_inner_steps),
        "--num_attempts",
        str(args.num_attempts),
        "--max_turns",
        str(args.max_turns),
        "--reward_type",
        args.reward_type,
        "--preprocess_mode",
        args.preprocess_mode,
        "--split",
        split,
        "--benchmark_timing",
    ]
    if args.benchmark_trace_inner_steps:
        cmd.append("--benchmark_trace_inner_steps")
    if args.do_reflection:
        cmd.append("--do_reflection")
    if args.vla_checkpoint:
        cmd.extend(["--vla_checkpoint", args.vla_checkpoint])
    return cmd


def _spawn_server(
    args,
    cmd: list[str],
    log_path: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if args.jax_platforms:
        env["JAX_PLATFORMS"] = args.jax_platforms
    if extra_env:
        env.update(
            {key: str(value) for key, value in extra_env.items() if value is not None}
        )

    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=args.langtable_dir,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def _cleanup():
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        log_fh.close()

    atexit.register(_cleanup)
    return proc


def _build_trainer_cmd(
    args,
    output_dir: Path,
    train_addresses: list[str],
    val_addresses: list[str],
) -> list[str]:
    run_name = args.run_name or f"language_table_benchmark_{int(time.time())}"
    trainer_local_dir = output_dir / "trainer"
    benchmark_dir = output_dir / "artifacts"

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=gigpo",
        f"data.train_files={args.train_data}",
        f"data.val_files={args.val_data}",
        f"data.train_batch_size={args.train_num_envs}",
        f"data.val_batch_size={args.val_num_envs}",
        "data.max_prompt_length=4096",
        "data.max_response_length=1024",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.return_raw_chat=True",
        f"actor_rollout_ref.model.path={args.model_path}",
        "+actor_rollout_ref.model.enable_thinking=False",
        f"actor_rollout_ref.actor.optim.lr={args.learning_rate}",
        "actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.micro_batch_size}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.7",
        "actor_rollout_ref.rollout.val_kwargs.top_p=0.8",
        "actor_rollout_ref.rollout.val_kwargs.top_k=20",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "actor_rollout_ref.rollout.max_num_batched_tokens=32768",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.use_invalid_action_penalty=True",
        "actor_rollout_ref.actor.invalid_action_penalty_coef=0.5",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "algorithm.gamma=0.95",
        "+algorithm.step_gamma=0.95",
        "+algorithm.traj_gamma=0.9",
        "algorithm.gigpo.step_advantage_w=1.0",
        "algorithm.gigpo.mode=mean_norm",
        "reward_model.reward_manager=episode",
        "env.env_name=language_table",
        "env.seed=0",
        "+env.remote=True",
        f"env.rollout.n={args.group_size}",
        f"env.num_attempts={args.num_attempts}",
        f"env.max_turns={args.max_turns}",
        "+env.reflection_type=history_and_reflection",
        "trainer.critic_warmup=0",
        "trainer.logger=['console']",
        "trainer.project_name=lamer_benchmark",
        f"trainer.experiment_name={run_name}",
        f"trainer.default_local_dir={trainer_local_dir}",
        "trainer.n_gpus_per_node=4",
        "trainer.nnodes=1",
        "trainer.save_freq=0",
        "trainer.test_freq=0",
        "trainer.total_epochs=100000",
        "trainer.val_before_train=False",
        "trainer.log_val_generations=0",
        "trainer.log_train_generations=0",
        "trainer.log_train_videos=0",
        "trainer.log_val_videos=0",
        "trainer.max_actor_ckpt_to_keep=1",
        "trainer.max_critic_ckpt_to_keep=1",
        "trainer.resume_mode=disable",
        "benchmark.enabled=True",
        "benchmark.profile=compare",
        f"benchmark.preset_name={args.preset}",
        f"benchmark.output_dir={benchmark_dir}",
        f"benchmark.warmup_iterations={args.warmup_iterations}",
        f"benchmark.measured_iterations={args.measured_iterations}",
    ]

    if len(train_addresses) > 1 or len(val_addresses) > 1:
        cmd.extend(
            [
                "+env.sharded=True",
                f"+env.remote_addresses={_hydra_list(train_addresses)}",
                f"+env.remote_val_addresses={_hydra_list(val_addresses)}",
            ]
        )
    else:
        cmd.extend(
            [
                f"+env.remote_address={train_addresses[0]}",
                f"+env.remote_val_address={val_addresses[0]}",
            ]
        )
    return cmd


def _print_summary(summary_path: Path) -> None:
    if not summary_path.exists():
        return
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    print("\nBenchmark summary:")
    for key in (
        "iterations_measured",
        "iterations_warmup",
        "rollout_elapsed_s_mean",
    ):
        if key in summary:
            print(f"  {key}: {summary[key]}")
    for key in ("step", "gen", "old_log_prob", "adv", "update_actor"):
        if key in summary.get("timing_s_mean", {}):
            print(f"  timing_s_mean/{key}: {summary['timing_s_mean'][key]:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark LaMer Language Table end-to-end training"
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="resolved_training_config")
    parser.add_argument("--mode", choices=["attach", "spawn"], default="attach")
    parser.add_argument("--train-data", default=os.environ.get("TRAIN_DATA_PATH"))
    parser.add_argument("--val-data", default=os.environ.get("VAL_DATA_PATH"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-path", default="Qwen/Qwen3-4B")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument(
        "--preprocess-mode",
        choices=["original", "batched_tf", "jax_gpu"],
        default="jax_gpu",
    )
    parser.add_argument("--reward-type", default="block2block")
    parser.add_argument("--block-mode", default="BLOCK_4")
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--measured-iterations", type=int, default=5)
    parser.add_argument("--do-reflection", dest="do_reflection", action="store_true")
    parser.add_argument("--no-reflection", dest="do_reflection", action="store_false")
    parser.set_defaults(do_reflection=True)
    parser.add_argument("--benchmark-trace-inner-steps", action="store_true")
    parser.add_argument("--train-addresses", default="127.0.0.1:50051")
    parser.add_argument("--val-addresses", default="127.0.0.1:50052")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--train-port", type=int, default=50051)
    parser.add_argument("--val-port", type=int, default=50052)
    parser.add_argument("--langtable-python", default=sys.executable)
    parser.add_argument("--langtable-dir", type=Path, default=DEFAULT_LANGTABLE_DIR)
    parser.add_argument("--vla-checkpoint", default=None)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--jax-platforms", default="cuda")
    parser.add_argument("--trainer-visible-gpus", default=None)
    parser.add_argument(
        "--env-server-gpu",
        default="0",
        help="Comma-separated GPU ids for train env servers. Each shard gets one GPU.",
    )
    parser.add_argument("--train-server-mem-fraction", default="0.7")
    parser.add_argument("--val-server-mem-fraction", default="0.2")
    parser.add_argument("--ray-num-cpus", default=None)
    args = parser.parse_args()

    if not args.train_data or not args.val_data:
        parser.error(
            "--train-data and --val-data are required (or set "
            "TRAIN_DATA_PATH / VAL_DATA_PATH)"
        )

    preset = PRESETS[args.preset]
    args.train_num_envs = preset["train_num_envs"]
    args.val_num_envs = preset["val_num_envs"]
    args.group_size = preset["group_size"]
    args.num_attempts = preset["num_attempts"]
    args.max_turns = preset["max_turns"]
    args.max_inner_steps = preset["max_inner_steps"]

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_RESULTS_DIR / f"{args.preset}_{int(time.time())}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_addresses = _split_csv(args.train_addresses)
    val_addresses = _split_csv(args.val_addresses)
    env_server_gpus = _split_csv(args.env_server_gpu)
    if not env_server_gpus:
        parser.error("--env-server-gpu must specify at least one GPU id")

    spawned = []
    if args.mode == "spawn":
        if args.train_num_envs % len(env_server_gpus) != 0:
            parser.error(
                "For sharded Language Table benchmarking, TRAIN_NUM_ENVS must be "
                "divisible by the number of env-server GPUs so each shard owns "
                "the same number of prompt groups."
            )

        train_envs_per_shard = args.train_num_envs // len(env_server_gpus)
        train_addresses = []
        used_ports = set()
        for idx, gpu_id in enumerate(env_server_gpus):
            train_port = args.train_port + idx
            used_ports.add(train_port)
            train_cmd = _build_server_cmd(
                args,
                port=train_port,
                num_envs=train_envs_per_shard,
                group_n=args.group_size,
                split="train",
            )
            spawned.append(
                _spawn_server(
                    args,
                    train_cmd,
                    output_dir / f"train_env_server_{idx}.log",
                    extra_env={
                        "CUDA_VISIBLE_DEVICES": gpu_id,
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        "XLA_PYTHON_CLIENT_MEM_FRACTION": args.train_server_mem_fraction,
                    },
                )
            )
            _wait_for_port(args.host, train_port, args.startup_timeout)
            train_addresses.append(f"{args.host}:{train_port}")

        val_gpu = env_server_gpus[0]
        val_port = args.val_port
        while val_port in used_ports:
            val_port += 1
        val_cmd = _build_server_cmd(
            args,
            port=val_port,
            num_envs=args.val_num_envs,
            group_n=1,
            split="val",
        )
        spawned.append(
            _spawn_server(
                args,
                val_cmd,
                output_dir / "val_env_server.log",
                extra_env={
                    "CUDA_VISIBLE_DEVICES": val_gpu,
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": args.val_server_mem_fraction,
                },
            )
        )
        _wait_for_port(args.host, val_port, args.startup_timeout)
        val_addresses = [f"{args.host}:{val_port}"]

    trainer_cmd = _build_trainer_cmd(args, output_dir, train_addresses, val_addresses)
    print("Running trainer command:")
    print("  " + " ".join(shlex.quote(part) for part in trainer_cmd))

    with (output_dir / "launcher_command.txt").open("w", encoding="utf-8") as fh:
        fh.write(" ".join(shlex.quote(part) for part in trainer_cmd) + "\n")

    trainer_env = os.environ.copy()
    if args.trainer_visible_gpus:
        trainer_env["CUDA_VISIBLE_DEVICES"] = args.trainer_visible_gpus
    if args.ray_num_cpus:
        trainer_env["RAY_NUM_CPUS"] = str(args.ray_num_cpus)
    result = subprocess.run(trainer_cmd, cwd=REPO_ROOT, env=trainer_env)
    summary_path = output_dir / "artifacts" / "benchmark_summary.json"
    if result.returncode == 0:
        _print_summary(summary_path)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
