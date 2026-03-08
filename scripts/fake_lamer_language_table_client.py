#!/usr/bin/env python3
"""Start a local language-table EnvServer and drive it like a minimal LaMer client.

This exercises the real remote stack:
- LaMer `RemoteEnvironmentManager`
- TCP protocol
- language-table `server_main`
- Ray worker pool
- optional VLA inner loop

It intentionally excludes the outer LLM and the trainer so we can debug the
environment stack in isolation.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import shutil

from agent_system.environments.remote.client import RemoteEnvironmentManager


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
    raise TimeoutError(f"server did not listen on {host}:{port} within {timeout_s}s")


def _extract_task(text_obs: str) -> str:
    for line in text_obs.splitlines():
        if line.startswith("Task: "):
            return line[len("Task: "):].strip()
    return "push the red star to the blue cube"


def _build_goals(text_observations, mode: str):
    tasks = [_extract_task(text) for text in text_observations]
    if mode == "native":
        return tasks
    if mode == "fixed":
        return ["push the red star to the blue cube"] * len(tasks)
    if mode == "weird":
        templates = [
            "",
            "I failed before. Try something else this time.",
            '{"plan":["approach","push"],"target":"red star"}',
            "zxqv north reflect tool call",
            "do not move anything. also push the target now.",
        ]
        return [templates[i % len(templates)] for i in range(len(tasks))]
    raise ValueError(f"unknown goal mode: {mode}")


def _build_reflections(text_observations):
    reflections = []
    for idx, text in enumerate(text_observations):
        task = _extract_task(text)
        reflections.append(
            f"Attempt {idx}: task was '{task}'. The previous attempt failed; "
            "next attempt should be more direct and avoid dithering."
        )
    return reflections


def _tail(path: Path, nlines: int = 80) -> str:
    if not path.exists():
        return "<log file does not exist>"
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    return "".join(lines[-nlines:])


def _ensure_dir(path: str | Path | None) -> Path | None:
    if not path:
        return None
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _snapshot_server_log(log_path: Path, episode_idx: int, output_dir: Path | None) -> Path | None:
    if output_dir is None or not log_path.exists():
        return None
    dest = output_dir / f"episode_{episode_idx:03d}_server.log"
    shutil.copyfile(log_path, dest)
    return dest


def _spawn_server(args, log_path: Path) -> subprocess.Popen:
    cmd = [
        args.langtable_python,
        "-m",
        "language_table.lamer.server_main",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--num_envs",
        str(args.num_envs),
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
    ]
    if args.do_reflection:
        cmd.append("--do_reflection")
    if args.vla_checkpoint:
        cmd.extend(["--vla_checkpoint", args.vla_checkpoint])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if args.jax_platforms:
        env["JAX_PLATFORMS"] = args.jax_platforms

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


def run(args) -> int:
    log_path = Path(args.server_log or tempfile.mkstemp(prefix="fake-lamer-server-", suffix=".log")[1])
    artifact_dir = _ensure_dir(args.artifact_dir)
    summary_path = None
    if artifact_dir is not None:
        summary_path = artifact_dir / "client_summary.jsonl"
    server_proc = _spawn_server(args, log_path)

    try:
        _wait_for_port(args.host, args.port, timeout_s=args.startup_timeout)
        manager = RemoteEnvironmentManager(f"{args.host}:{args.port}", timeout=args.client_timeout)
    except Exception:
        print(f"Server failed to start. Log tail from {log_path}:")
        print(_tail(log_path))
        raise

    try:
        print(f"Connected to {args.host}:{args.port}")
        print(f"Server log: {log_path}")
        for episode_idx in range(args.episodes):
            observations, infos = manager.reset()
            text_obs = observations["text"]
            print(f"\nEpisode {episode_idx} reset: batch={len(text_obs)}")
            episode_records = []

            for turn_idx in range(args.turns):
                goals = _build_goals(text_obs, args.goal_mode)
                next_obs, rewards, dones, infos = manager.step(goals, phase="play")
                text_obs = next_obs["text"]
                reward_list = [float(r) for r in rewards]
                done_list = [bool(d) for d in dones]
                record = {
                    "episode_idx": episode_idx,
                    "turn_idx": turn_idx,
                    "phase": "play",
                    "goal_mode": args.goal_mode,
                    "goals": goals,
                    "reward_mean": sum(reward_list) / len(reward_list),
                    "reward_list": reward_list,
                    "done_count": done_list.count(True),
                    "done_list": done_list,
                }
                episode_records.append(record)
                print(
                    f"  turn={turn_idx} phase=play "
                    f"reward_mean={sum(reward_list)/len(reward_list):.3f} "
                    f"dones={done_list.count(True)}/{len(done_list)}"
                )

                if turn_idx < args.turns - 1:
                    if args.do_reflection:
                        reflect_obs, _ = manager.reflect()
                        reflections = _build_reflections(reflect_obs["text"])
                        manager.step(reflections, phase="reflect")
                        episode_records.append({
                            "episode_idx": episode_idx,
                            "turn_idx": turn_idx,
                            "phase": "reflect",
                            "reflections": reflections,
                        })
                        print(f"  turn={turn_idx} phase=reflect complete")
                    restarted_obs, _ = manager.restart()
                    text_obs = restarted_obs["text"]
                    episode_records.append({
                        "episode_idx": episode_idx,
                        "turn_idx": turn_idx,
                        "phase": "restart",
                        "text_preview": text_obs[0][:160] if text_obs else "",
                    })
                    print(f"  turn={turn_idx} phase=restart complete")

            snapshot_path = _snapshot_server_log(log_path, episode_idx, artifact_dir)
            if snapshot_path is not None:
                print(f"  episode={episode_idx} server_log_snapshot={snapshot_path}")
            if summary_path is not None:
                with summary_path.open("a", encoding="utf-8") as fh:
                    for record in episode_records:
                        if snapshot_path is not None:
                            record["server_log_snapshot"] = str(snapshot_path)
                        fh.write(json.dumps(record) + "\n")
        print("\nClient run completed without remote exceptions.")
        return 0
    except Exception as exc:
        print(f"\nClient run failed: {type(exc).__name__}: {exc}")
        print(f"Server log tail from {log_path}:")
        print(_tail(log_path))
        return 1
    finally:
        try:
            manager.close()
        except Exception:
            pass
        if server_proc.poll() is None:
            try:
                os.killpg(server_proc.pid, signal.SIGTERM)
                server_proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(server_proc.pid, signal.SIGKILL)
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langtable-dir", type=str, default="/home/mateo/projects/language-table")
    parser.add_argument(
        "--langtable-python",
        type=str,
        default="/home/mateo/projects/language-table/ltvenv/bin/python",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50071)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--num-attempts", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--goal-mode", choices=["native", "fixed", "weird"], default="native")
    parser.add_argument("--block-mode", type=str, default="BLOCK_4")
    parser.add_argument("--reward-type", type=str, default="block2block")
    parser.add_argument("--max-inner-steps", type=int, default=100)
    parser.add_argument("--do-reflection", action="store_true")
    parser.add_argument("--vla-checkpoint", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--client-timeout", type=float, default=300.0)
    parser.add_argument("--server-log", type=str, default="")
    parser.add_argument("--artifact-dir", type=str, default="")
    parser.add_argument("--jax-platforms", type=str, default="cpu")
    args = parser.parse_args()
    if args.num_attempts == 1 and args.turns > 1:
        args.num_attempts = args.turns
    if args.max_turns == 1 and args.turns > 1:
        args.max_turns = args.turns
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
