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

PRESET_NAMES = ("doc_baseline", "resolved_training_config")
DOC_BASELINE_PRESET = {
    "train_num_envs": 8,
    "val_num_envs": 8,
    "group_size": 16,
    "num_attempts": 3,
    "max_turns": 4,
    "max_inner_steps": 96,
}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _hydra_list(items: list[str]) -> str:
    quoted = ",".join(json.dumps(item) for item in items)
    return f"[{quoted}]"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_train_shard_layout(
    train_num_envs: int,
    group_size: int,
    shard_count: int,
) -> int:
    """Validate that prompt groups can be partitioned across shards.

    ``TRAIN_NUM_ENVS`` is the number of prompt groups. Each group expands to
    ``GROUP_SIZE`` env workers inside the language-table server via
    ``num_processes = num_envs * group_n``. To keep every contiguous group on
    one shard, we only need the number of prompt groups to divide evenly across
    the requested shard count.
    """
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if train_num_envs % shard_count != 0:
        raise ValueError(
            "TRAIN_NUM_ENVS must be divisible by the train shard count. "
            f"train_num_envs={train_num_envs}, shard_count={shard_count}"
        )
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    train_envs_per_shard = train_num_envs // shard_count
    return train_envs_per_shard


def _append_multistep_task_flags(args, cmd: list[str], split: str) -> None:
    if args.reward_type != "multistep":
        return

    if split == "train":
        locations = args.train_task_locations
        shapes = args.train_task_shapes
        colors = args.train_task_colors
        n_steps = args.train_task_n_steps
    else:
        locations = args.val_task_locations
        shapes = args.val_task_shapes
        colors = args.val_task_colors
        n_steps = args.val_task_n_steps

    if locations:
        cmd.extend(["--task_locations", locations])
    if shapes:
        cmd.extend(["--task_shapes", shapes])
    if colors:
        cmd.extend(["--task_colors", colors])
    cmd.extend(["--task_n_steps", str(n_steps)])


def _resolve_training_shape(args) -> dict[str, int]:
    if args.preset == "doc_baseline":
        return dict(DOC_BASELINE_PRESET)

    return {
        "train_num_envs": int(args.train_num_envs),
        "val_num_envs": int(args.val_num_envs),
        "group_size": int(args.group_size),
        "num_attempts": int(args.num_attempts),
        "max_turns": int(args.max_turns),
        "max_inner_steps": int(args.max_inner_steps),
    }


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
    _append_multistep_task_flags(args, cmd, split)
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
        f"algorithm.adv_estimator={args.adv_estimator}",
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
        f"actor_rollout_ref.actor.use_kl_loss={args.use_kl_loss}",
        f"actor_rollout_ref.actor.kl_loss_coef={args.kl_loss_coef}",
        f"actor_rollout_ref.actor.kl_loss_type={args.kl_loss_type}",
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
        f"algorithm.use_kl_in_reward={args.use_kl_in_reward}",
        f"algorithm.kl_ctrl.kl_coef={args.kl_reward_coef}",
        "algorithm.gamma=0.95",
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

    if args.adv_estimator == "gigpo":
        cmd.extend(
            [
                "+algorithm.step_gamma=0.95",
                "+algorithm.traj_gamma=0.9",
                "algorithm.gigpo.step_advantage_w=1.0",
                "algorithm.gigpo.mode=mean_norm",
            ]
        )

    if args.skip_val_server:
        cmd.append("+env.skip_val_env=True")

    if len(train_addresses) > 1 or len(val_addresses) > 1:
        cmd.extend(["+env.sharded=True", f"+env.remote_addresses={_hydra_list(train_addresses)}"])
        if val_addresses:
            cmd.append(f"+env.remote_val_addresses={_hydra_list(val_addresses)}")
    else:
        cmd.append(f"+env.remote_address={train_addresses[0]}")
        if val_addresses:
            cmd.append(f"+env.remote_val_address={val_addresses[0]}")
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
    parser.add_argument("--preset", choices=PRESET_NAMES, default="doc_baseline")
    parser.add_argument("--mode", choices=["attach", "spawn"], default="attach")
    parser.add_argument("--train-data", default=os.environ.get("TRAIN_DATA_PATH"))
    parser.add_argument("--val-data", default=os.environ.get("VAL_DATA_PATH"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME"))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B"))
    parser.add_argument("--adv-estimator", default=os.environ.get("ADV_ESTIMATOR", "gigpo"))
    parser.add_argument("--learning-rate", type=float, default=_env_float("LEARNING_RATE", 1e-6))
    parser.add_argument("--batch-size", type=int, default=_env_int("BATCH_SIZE", 64))
    parser.add_argument("--micro-batch-size", type=int, default=_env_int("MICRO_BATCH_SIZE", 16))
    parser.add_argument("--use-kl-loss", default=os.environ.get("USE_KL_LOSS", "False"))
    parser.add_argument("--kl-loss-coef", type=float, default=_env_float("KL_LOSS_COEF", 0.001))
    parser.add_argument("--kl-loss-type", default=os.environ.get("KL_LOSS_TYPE", "low_var_kl"))
    parser.add_argument("--use-kl-in-reward", default=os.environ.get("USE_KL_IN_REWARD", "False"))
    parser.add_argument("--kl-reward-coef", type=float, default=_env_float("KL_REWARD_COEF", 0.001))
    parser.add_argument(
        "--preprocess-mode",
        choices=["original", "batched_tf", "jax_gpu", "jax_fused"],
        default=os.environ.get("PREPROCESS_MODE", "jax_gpu"),
    )
    parser.add_argument("--reward-type", default=os.environ.get("REWARD_TYPE", "block2block"))
    parser.add_argument("--block-mode", default="BLOCK_4")
    parser.add_argument("--train-num-envs", type=int, default=_env_int("TRAIN_NUM_ENVS", 4))
    parser.add_argument("--val-num-envs", type=int, default=_env_int("VAL_NUM_ENVS", 4))
    parser.add_argument("--group-size", type=int, default=_env_int("GROUP_SIZE", 8))
    parser.add_argument("--max-inner-steps", type=int, default=_env_int("MAX_INNER_STEPS", 5))
    parser.add_argument("--num-attempts", type=int, default=_env_int("NUM_ATTEMPTS", 2))
    parser.add_argument("--max-turns", type=int, default=_env_int("MAX_TURNS", 2))
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--measured-iterations", type=int, default=3)
    parser.add_argument("--do-reflection", dest="do_reflection", action="store_true")
    parser.add_argument("--no-reflection", dest="do_reflection", action="store_false")
    parser.set_defaults(do_reflection=True)
    parser.add_argument("--skip-val-server", dest="skip_val_server", action="store_true")
    parser.add_argument("--keep-val-server", dest="skip_val_server", action="store_false")
    parser.set_defaults(
        skip_val_server=_env_bool("BENCH_SKIP_VAL_SERVER", False)
    )
    parser.add_argument("--benchmark-trace-inner-steps", action="store_true")
    parser.add_argument("--train-task-locations", default=os.environ.get("TRAIN_TASK_LOCATIONS"))
    parser.add_argument("--train-task-shapes", default=os.environ.get("TRAIN_TASK_SHAPES"))
    parser.add_argument("--train-task-colors", default=os.environ.get("TRAIN_TASK_COLORS"))
    parser.add_argument("--train-task-n-steps", type=int, default=_env_int("TRAIN_TASK_N_STEPS", 2))
    parser.add_argument("--val-task-locations", default=os.environ.get("VAL_TASK_LOCATIONS"))
    parser.add_argument("--val-task-shapes", default=os.environ.get("VAL_TASK_SHAPES"))
    parser.add_argument("--val-task-colors", default=os.environ.get("VAL_TASK_COLORS"))
    parser.add_argument("--val-task-n-steps", type=int, default=_env_int("VAL_TASK_N_STEPS", 3))
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

    resolved_shape = _resolve_training_shape(args)
    args.train_num_envs = resolved_shape["train_num_envs"]
    args.val_num_envs = resolved_shape["val_num_envs"]
    args.group_size = resolved_shape["group_size"]
    args.num_attempts = resolved_shape["num_attempts"]
    args.max_turns = resolved_shape["max_turns"]
    args.max_inner_steps = resolved_shape["max_inner_steps"]

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
    train_envs_per_shard = None
    if args.mode == "spawn":
        try:
            train_envs_per_shard = _validate_train_shard_layout(
                train_num_envs=args.train_num_envs,
                group_size=args.group_size,
                shard_count=len(env_server_gpus),
            )
        except ValueError as exc:
            parser.error(str(exc))
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

        if args.skip_val_server:
            val_addresses = []
        else:
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
    elif args.skip_val_server:
        val_addresses = []

    launcher_metadata = {
        "preset": args.preset,
        "mode": args.mode,
        "preprocess_mode": args.preprocess_mode,
        "train_num_envs": int(args.train_num_envs),
        "val_num_envs": int(args.val_num_envs),
        "group_size": int(args.group_size),
        "train_shard_count": int(len(train_addresses)),
        "train_envs_per_shard": (
            int(train_envs_per_shard)
            if train_envs_per_shard is not None
            else (
                int(args.train_num_envs // len(train_addresses))
                if train_addresses and args.train_num_envs % len(train_addresses) == 0
                else None
            )
        ),
        "env_server_gpus": env_server_gpus[: len(train_addresses)],
        "skip_val_server": bool(args.skip_val_server),
        "val_server_enabled": bool(val_addresses),
        "train_addresses": list(train_addresses),
        "val_addresses": list(val_addresses),
    }
    with (output_dir / "launcher_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(launcher_metadata, fh, indent=2)

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
