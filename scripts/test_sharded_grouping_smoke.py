#!/usr/bin/env python3
"""
Smoke tests for grouped sharding invariants.

These checks answer three concrete questions:
1. Do shard boundaries land only on whole-group boundaries?
2. Does the Language Table env wiring enforce that invariant outside the benchmark?
3. When the invariant holds, does contiguous slicing keep each full group on one shard?

Usage
-----
python scripts/test_sharded_grouping_smoke.py --all
python scripts/test_sharded_grouping_smoke.py --test grouping_layout
python scripts/test_sharded_grouping_smoke.py --test live_remote_roundtrip
LANGTABLE_PYTHON=/path/to/ltvenv/bin/python python scripts/test_sharded_grouping_smoke.py --test live_remote_roundtrip
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO_ROOT = Path(__file__).resolve().parents[1]
LANGTABLE_DIR = Path(
    os.environ.get("LANGTABLE_DIR", str(REPO_ROOT.parent / "language-table"))
)
LANGTABLE_PYTHON = os.environ.get("LANGTABLE_PYTHON", sys.executable)


class SkippedTest(RuntimeError):
    """Raised when a smoke test cannot run in the current local environment."""


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]


def _read_log_excerpt(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return ""
    excerpt = "\n".join(lines[-40:])
    return f"\n--- server log tail ---\n{excerpt}"


def _wait_for_port(
    host: str,
    port: int,
    timeout_s: float,
    proc: subprocess.Popen | None = None,
    log_path: Path | None = None,
):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"Live Language Table server exited before opening {host}:{port}"
                f"{_read_log_excerpt(log_path) if log_path is not None else ''}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((host, port))
            except OSError:
                time.sleep(1.0)
                continue
            return

    raise TimeoutError(
        f"Timed out waiting for {host}:{port}"
        f"{_read_log_excerpt(log_path) if log_path is not None else ''}"
    )


def _start_live_language_table_server(
    port: int,
    num_envs: int,
    group_n: int,
    logs_dir: Path,
    split: str = "train",
    startup_timeout_s: float = 180.0,
):
    if not LANGTABLE_DIR.exists():
        raise SkippedTest(f"LANGTABLE_DIR does not exist: {LANGTABLE_DIR}")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    pythonpath_entries = [str(LANGTABLE_DIR), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_entries)

    log_path = logs_dir / f"live_server_{port}.log"
    cmd = [
        LANGTABLE_PYTHON,
        "-m",
        "language_table.lamer.server_main",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--num_envs",
        str(num_envs),
        "--group_n",
        str(group_n),
        "--split",
        split,
        "--num_attempts",
        "1",
        "--max_turns",
        "1",
        "--max_inner_steps",
        "1",
        "--reward_type",
        "block2block",
        "--preprocess_mode",
        "original",
        "--no_render",
    ]
    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=LANGTABLE_DIR,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _wait_for_port("127.0.0.1", port, startup_timeout_s, proc=proc, log_path=log_path)
    return proc, log_fh, log_path


def _stop_live_process(proc: subprocess.Popen | None, log_fh=None):
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if log_fh is not None:
        log_fh.close()


def _build_fake_sharded_manager(shard_sizes):
    import agent_system.environments.remote.sharded_client as sharded_module

    addresses = [f"shard{idx}" for idx in range(len(shard_sizes))]
    size_by_address = dict(zip(addresses, shard_sizes))
    instances = {}

    class FakeRemoteEnvironmentManager:
        def __init__(self, address: str, timeout: float = 300.0):
            self.address = address
            self.timeout = timeout
            self.num_processes = int(size_by_address[address])
            self.num_attempts = 3
            self.max_turns = 4
            self.do_reflection = True
            self.step_calls = []
            instances[address] = self

        def reset(self):
            obs = {
                "text": [f"{self.address}:reset:{i}" for i in range(self.num_processes)],
                "image": None,
                "anchor": [self.address] * self.num_processes,
            }
            infos = [{"shard": self.address, "local_idx": i} for i in range(self.num_processes)]
            return obs, infos

        def step(self, text_actions, phase="play"):
            self.step_calls.append((list(text_actions), phase))
            obs = {
                "text": [f"{self.address}:{action}" for action in text_actions],
                "image": None,
                "anchor": list(text_actions),
            }
            rewards = np.zeros(len(text_actions), dtype=np.float32)
            dones = np.zeros(len(text_actions), dtype=bool)
            infos = [
                {"shard": self.address, "action": action, "phase": phase}
                for action in text_actions
            ]
            return obs, rewards, dones, infos

        def restart(self):
            return self.reset()

        def reflect(self):
            return self.reset()

        def close(self):
            return None

        def get_last_benchmark_stats(self):
            return {}

    with mock.patch.object(
        sharded_module,
        "RemoteEnvironmentManager",
        FakeRemoteEnvironmentManager,
    ):
        manager = sharded_module.ShardedRemoteEnvironmentManager(addresses)

    return manager, instances


class _FakeValidatedShardedRemote:
    def __init__(self, shard_sizes):
        self._shard_sizes = list(shard_sizes)
        self.num_processes = sum(self._shard_sizes)
        self.num_attempts = 3
        self.max_turns = 4
        self.do_reflection = True

    def validate_group_partition(
        self,
        group_size: int,
        require_equal_groups_per_shard: bool = False,
    ):
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.num_processes % group_size != 0:
            raise ValueError("total_num_processes must be divisible by group_size")

        groups_per_shard = []
        for shard_size in self._shard_sizes:
            if shard_size % group_size != 0:
                raise ValueError("Shard boundary splits a contiguous env group")
            groups_per_shard.append(shard_size // group_size)

        if require_equal_groups_per_shard and len(set(groups_per_shard)) != 1:
            raise ValueError("Shards must own the same number of groups")

        return {
            "group_size": group_size,
            "groups_per_shard": groups_per_shard,
        }

    def reset(self):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def restart(self):
        raise NotImplementedError

    def reflect(self):
        raise NotImplementedError

    def close(self):
        return None


def test_grouping_layout():
    print("\n=== TEST: grouping layout guarantee ===")
    group_size = 8
    manager, _ = _build_fake_sharded_manager([32, 32, 32, 32])
    try:
        summary = manager.validate_group_partition(
            group_size,
            require_equal_groups_per_shard=True,
        )
        assert summary["groups_per_shard"] == [4, 4, 4, 4]

        for group_start in range(0, manager.num_processes, group_size):
            shard_idx = manager.shard_index_for_process(group_start)
            for process_idx in range(group_start, group_start + group_size):
                assert manager.shard_index_for_process(process_idx) == shard_idx, (
                    f"group starting at {group_start} crosses shard boundary"
                )

        print("  every contiguous 8-member group stays on one shard: OK")
        print("PASSED\n")
    finally:
        manager.close()


def test_rejects_group_splitting():
    print("\n=== TEST: reject split group boundary ===")
    manager, _ = _build_fake_sharded_manager([16, 20, 12, 16])
    try:
        try:
            manager.validate_group_partition(8, require_equal_groups_per_shard=True)
        except ValueError as exc:
            assert "splits a contiguous env group" in str(exc)
            print(f"  invalid shard layout rejected: OK ({exc})")
        else:
            raise AssertionError("Expected invalid shard layout to raise ValueError")
        print("PASSED\n")
    finally:
        manager.close()


def test_language_table_make_envs_enforces_grouping():
    print("\n=== TEST: language table make_envs guardrail ===")
    from omegaconf import OmegaConf
    from agent_system.environments.language_table.env_manager import make_envs

    config = OmegaConf.create(
        {
            "env": {
                "sharded": True,
                "remote_addresses": ["train0", "train1"],
                "remote_val_addresses": ["val0"],
                "rollout": {"n": 8},
                "reflection_type": "history_and_reflection",
            }
        }
    )

    def _fake_ctor(addresses):
        if list(addresses) == ["train0", "train1"]:
            return _FakeValidatedShardedRemote([24, 40])
        if list(addresses) == ["val0"]:
            return _FakeValidatedShardedRemote([128])
        raise AssertionError(f"Unexpected addresses: {addresses}")

    with mock.patch(
        "agent_system.environments.remote.ShardedRemoteEnvironmentManager",
        side_effect=_fake_ctor,
    ):
        try:
            make_envs(config)
        except ValueError as exc:
            assert "same number of groups" in str(exc)
            print(f"  non-benchmark env construction rejects bad layout: OK ({exc})")
        else:
            raise AssertionError("Expected make_envs() to reject invalid sharded layout")

    print("PASSED\n")


def test_step_slicing_keeps_groups_together():
    print("\n=== TEST: step slicing keeps full groups together ===")
    group_size = 2
    manager, instances = _build_fake_sharded_manager([4, 4])
    try:
        manager.validate_group_partition(group_size, require_equal_groups_per_shard=True)
        actions = [
            f"group{idx // group_size}_member{idx % group_size}"
            for idx in range(manager.num_processes)
        ]
        manager.step(actions, phase="play")

        shard0_actions, _ = instances["shard0"].step_calls[-1]
        shard1_actions, _ = instances["shard1"].step_calls[-1]
        assert shard0_actions == [
            "group0_member0",
            "group0_member1",
            "group1_member0",
            "group1_member1",
        ]
        assert shard1_actions == [
            "group2_member0",
            "group2_member1",
            "group3_member0",
            "group3_member1",
        ]
        print("  contiguous slicing preserved whole groups on each shard: OK")
        print("PASSED\n")
    finally:
        manager.close()


def test_live_remote_language_table_roundtrip():
    print("\n=== TEST: live remote language-table sharded roundtrip ===")
    from agent_system.environments.remote import ShardedRemoteEnvironmentManager

    group_size = 2
    per_shard_num_envs = 1
    ports = [_find_free_port(), _find_free_port()]
    servers = []
    client = None

    with tempfile.TemporaryDirectory(prefix="langtable-sharded-smoke-") as tmpdir:
        logs_dir = Path(tmpdir)
        try:
            try:
                for port in ports:
                    servers.append(
                        _start_live_language_table_server(
                            port=port,
                            num_envs=per_shard_num_envs,
                            group_n=group_size,
                            logs_dir=logs_dir,
                        )
                    )
            except RuntimeError as exc:
                if "ModuleNotFoundError" in str(exc) or "No module named" in str(exc):
                    raise SkippedTest(
                        "Live remote smoke test requires a Language Table Python "
                        "environment. Set LANGTABLE_PYTHON to the interpreter "
                        "that can run `language_table.lamer.server_main`."
                    ) from exc
                raise

            addresses = [f"127.0.0.1:{port}" for port in ports]
            client = ShardedRemoteEnvironmentManager(addresses, timeout=120.0)

            summary = client.validate_group_partition(
                group_size,
                require_equal_groups_per_shard=True,
            )
            assert summary["groups_per_shard"] == [1, 1]

            obs, infos = client.reset()
            assert len(obs["text"]) == client.num_processes
            assert len(infos) == client.num_processes

            for group_start in range(0, client.num_processes, group_size):
                shard_idx = client.shard_index_for_process(group_start)
                for process_idx in range(group_start, group_start + group_size):
                    assert client.shard_index_for_process(process_idx) == shard_idx

            actions = [
                f"group{idx // group_size}_member{idx % group_size}"
                for idx in range(client.num_processes)
            ]
            next_obs, rewards, dones, infos = client.step(actions, phase="play")
            assert len(next_obs["text"]) == client.num_processes
            assert rewards.shape == (client.num_processes,)
            assert dones.shape == (client.num_processes,)
            assert len(infos) == client.num_processes

            restart_obs, restart_infos = client.restart()
            assert len(restart_obs["text"]) == client.num_processes
            assert len(restart_infos) == client.num_processes

            print("  live servers accepted grouped sharded reset/step/restart: OK")
            print("PASSED\n")
        finally:
            if client is not None:
                client.close()
            for proc, log_fh, _ in servers:
                _stop_live_process(proc, log_fh)


ALL_TESTS = {
    "grouping_layout": test_grouping_layout,
    "rejects_group_splitting": test_rejects_group_splitting,
    "make_envs_guardrail": test_language_table_make_envs_enforces_grouping,
    "step_slicing": test_step_slicing_keeps_groups_together,
    "live_remote_roundtrip": test_live_remote_language_table_roundtrip,
}


def main():
    parser = argparse.ArgumentParser(
        description="Smoke tests for grouped sharded env invariants."
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help=f"Run a specific test: {list(ALL_TESTS)}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all grouping smoke tests",
    )
    args = parser.parse_args()

    if not args.all and args.test is None:
        parser.print_help()
        sys.exit(1)

    tests_to_run = ALL_TESTS if args.all else {args.test: ALL_TESTS[args.test]}
    passed = 0
    failed = 0
    skipped = 0
    for name, fn in tests_to_run.items():
        try:
            fn()
            passed += 1
        except SkippedTest as exc:
            print(f"SKIPPED: {name} - {exc}")
            skipped += 1
        except Exception as exc:
            import traceback

            print(f"FAILED: {name} - {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
