#!/usr/bin/env python3
"""
Test suite & benchmarks for the remote environment server.

Validates that a RemoteEnvironmentManager produces identical outcomes to a
local SokobanEnvironmentManager, and measures network overhead.

Usage
-----
# Run all tests (starts server automatically in a subprocess)
python scripts/test_remote_env.py --all

# Run a single test
python scripts/test_remote_env.py --test correctness
python scripts/test_remote_env.py --test metarl
python scripts/test_remote_env.py --test latency
python scripts/test_remote_env.py --test throughput
python scripts/test_remote_env.py --test large_batch
python scripts/test_remote_env.py --test reconnection
"""

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Ensure project root is on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

CONFIG_PATH = "verl/trainer/config/eval/sokoban.yaml"
SEED = 8192
PORT_BASE = 50061  # avoid collisions with user-facing servers


def _load_config(batch_size=2):
    from omegaconf import OmegaConf
    config = OmegaConf.load(CONFIG_PATH)
    config.data.train_batch_size = batch_size
    config.data.val_batch_size = batch_size
    return config


def _make_local_envs(config):
    """Create local (in-process) Sokoban envs."""
    from agent_system.environments.sokoban import make_envs
    return make_envs(config)


def _fixed_actions(num_processes, num_actions_per_turn=3):
    """Return a deterministic list of text actions."""
    directions = ["up", "down", "left", "right"]
    actions = []
    for i in range(num_processes):
        acts = ",".join(
            directions[(i + j) % len(directions)]
            for j in range(num_actions_per_turn)
        )
        actions.append(f"<action>{acts}</action>")
    return actions


def _fixed_reflections(num_processes):
    """Return a deterministic list of reflection actions."""
    return [
        "<remark>In my previous trial, I pushed boxes into corners. "
        "I should plan ahead to avoid dead states.</remark>"
    ] * num_processes


def _start_server_process(config, port, is_train=False):
    """Start an EnvServer in a child process. Returns the Process."""
    def _run():
        import ray
        from agent_system.environments.sokoban import make_envs
        from agent_system.environments.remote.server import EnvServer

        if not ray.is_initialized():
            ray.init(log_to_driver=False)
        train_envs, val_envs = make_envs(config)
        env = train_envs if is_train else val_envs
        server = EnvServer(env, host="127.0.0.1", port=port)
        server.serve()

    proc = multiprocessing.Process(target=_run, daemon=True)
    proc.start()
    # Give the server a moment to bind.
    time.sleep(3)
    return proc


def _connect_client(port, timeout=60.0):
    from agent_system.environments.remote.client import RemoteEnvironmentManager
    return RemoteEnvironmentManager(f"127.0.0.1:{port}", timeout=timeout)


def _arrays_equal(a, b, label=""):
    """Assert two numpy-like objects are identical."""
    a = np.asarray(a)
    b = np.asarray(b)
    if not np.array_equal(a, b):
        diff_mask = a != b
        raise AssertionError(
            f"Mismatch in {label}: "
            f"{np.count_nonzero(diff_mask)}/{a.size} elements differ.\n"
            f"  local : {a}\n"
            f"  remote: {b}"
        )


def _obs_equal(local_obs, remote_obs):
    """Compare observation dicts field by field."""
    for key in ("text", "image", "anchor"):
        lv = local_obs.get(key)
        rv = remote_obs.get(key)
        if lv is None and rv is None:
            continue
        if isinstance(lv, list) and isinstance(rv, list):
            assert lv == rv, f"obs['{key}'] text mismatch"
        elif isinstance(lv, np.ndarray) or isinstance(rv, np.ndarray):
            _arrays_equal(lv, rv, label=f"obs['{key}']")
        else:
            assert lv == rv, f"obs['{key}'] mismatch: {lv!r} vs {rv!r}"


def _infos_equal(local_infos, remote_infos):
    """Compare info dict lists."""
    assert len(local_infos) == len(remote_infos), "infos length mismatch"
    for i, (li, ri) in enumerate(zip(local_infos, remote_infos)):
        for key in set(li) | set(ri):
            lv = li.get(key)
            rv = ri.get(key)
            if isinstance(lv, np.ndarray) or isinstance(rv, np.ndarray):
                _arrays_equal(lv, rv, label=f"infos[{i}]['{key}']")
            else:
                assert lv == rv, (
                    f"infos[{i}]['{key}'] mismatch: {lv!r} vs {rv!r}"
                )


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

def test_correctness():
    """
    Run identical reset/step sequences on local and remote env managers
    with the same seed. Verify bit-identical results.
    """
    print("\n=== TEST: correctness ===")
    batch_size = 2
    config = _load_config(batch_size)
    num_steps = 5

    import ray
    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    # Local env
    _, local_env = _make_local_envs(config)

    # Remote env (server in subprocess)
    port = PORT_BASE
    server_proc = _start_server_process(config, port, is_train=False)
    try:
        remote_env = _connect_client(port)

        # --- Reset ---
        local_obs, local_infos = local_env.reset()
        remote_obs, remote_infos = remote_env.reset()
        _obs_equal(local_obs, remote_obs)
        _infos_equal(local_infos, remote_infos)
        print(f"  reset: OK")

        # --- Steps ---
        for step in range(num_steps):
            actions = _fixed_actions(batch_size)
            local_result = local_env.step(actions, phase="play")
            remote_result = remote_env.step(actions, phase="play")

            local_next_obs, local_rewards, local_dones, local_infos = local_result
            remote_next_obs, remote_rewards, remote_dones, remote_infos = remote_result

            _obs_equal(local_next_obs, remote_next_obs)
            _arrays_equal(local_rewards, remote_rewards, f"rewards[step={step}]")
            _arrays_equal(local_dones, remote_dones, f"dones[step={step}]")
            _infos_equal(local_infos, remote_infos)
            print(f"  step {step}: OK")

        remote_env.close()
        print("PASSED\n")
    finally:
        server_proc.kill()
        server_proc.join(timeout=5)


def test_metarl():
    """
    Full MetaRL loop: 3 attempts with reflection, 7 turns per attempt.
    Verify all intermediate observations match.
    """
    print("\n=== TEST: metarl (multi-attempt with reflection) ===")
    batch_size = 2
    config = _load_config(batch_size)

    import ray
    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    _, local_env = _make_local_envs(config)

    port = PORT_BASE + 1
    server_proc = _start_server_process(config, port, is_train=False)
    try:
        remote_env = _connect_client(port)

        num_attempts = local_env.num_attempts
        max_turns = local_env.max_turns

        assert remote_env.num_attempts == num_attempts
        assert remote_env.max_turns == max_turns
        assert remote_env.do_reflection == local_env.do_reflection
        assert remote_env.num_processes == local_env.num_processes
        print(f"  properties match: num_attempts={num_attempts}, "
              f"max_turns={max_turns}")

        # --- Reset ---
        local_obs, local_infos = local_env.reset()
        remote_obs, remote_infos = remote_env.reset()
        _obs_equal(local_obs, remote_obs)
        _infos_equal(local_infos, remote_infos)
        print(f"  reset: OK")

        for attempt_idx in range(num_attempts):
            if attempt_idx >= 1:
                # Reflect
                local_robs, local_rinfos = local_env.reflect()
                remote_robs, remote_rinfos = remote_env.reflect()
                _obs_equal(local_robs, remote_robs)
                _infos_equal(local_rinfos, remote_rinfos)

                reflections = _fixed_reflections(batch_size)
                local_result = local_env.step(reflections, phase="reflect")
                remote_result = remote_env.step(reflections, phase="reflect")
                _arrays_equal(local_result[1], remote_result[1],
                              "reflect rewards")
                _arrays_equal(local_result[2], remote_result[2],
                              "reflect dones")
                print(f"  attempt {attempt_idx} reflect: OK")

                # Restart
                local_obs, local_infos = local_env.restart()
                remote_obs, remote_infos = remote_env.restart()
                _obs_equal(local_obs, remote_obs)
                _infos_equal(local_infos, remote_infos)
                print(f"  attempt {attempt_idx} restart: OK")

            # Play
            for step in range(max_turns):
                actions = _fixed_actions(batch_size)
                local_result = local_env.step(actions, phase="play")
                remote_result = remote_env.step(actions, phase="play")

                _obs_equal(local_result[0], remote_result[0])
                _arrays_equal(local_result[1], remote_result[1],
                              f"rewards[a={attempt_idx},s={step}]")
                _arrays_equal(local_result[2], remote_result[2],
                              f"dones[a={attempt_idx},s={step}]")
                _infos_equal(local_result[3], remote_result[3])

            print(f"  attempt {attempt_idx} play ({max_turns} turns): OK")

        # --- success_evaluator ---
        # Build minimal total_batch_list and total_infos for success eval
        # (just validate the call works and returns identical structure)
        dummy_batch = [[{
            "active_masks": True, "traj_idx": 0, "phase": "play"
        }]] * batch_size
        dummy_infos = [[{"won": False}]] * batch_size
        dummy_lengths = np.array([1] * batch_size, dtype=np.int32)

        local_success = local_env.success_evaluator(
            total_infos=dummy_infos,
            total_batch_list=dummy_batch,
            episode_lengths=dummy_lengths,
        )
        remote_success = remote_env.success_evaluator(
            total_infos=dummy_infos,
            total_batch_list=dummy_batch,
            episode_lengths=dummy_lengths,
        )
        for key in local_success:
            _arrays_equal(local_success[key], remote_success[key],
                          f"success['{key}']")
        print(f"  success_evaluator: OK")

        remote_env.close()
        print("PASSED\n")
    finally:
        server_proc.kill()
        server_proc.join(timeout=5)


def test_latency():
    """
    Measure per-operation round-trip time for local vs remote.
    """
    print("\n=== TEST: latency benchmark ===")
    batch_size = 4
    config = _load_config(batch_size)
    n_trials = 50

    import ray
    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    _, local_env = _make_local_envs(config)

    port = PORT_BASE + 2
    server_proc = _start_server_process(config, port, is_train=False)
    try:
        remote_env = _connect_client(port)

        results = {}

        for label, env in [("local", local_env), ("remote", remote_env)]:
            times = {"reset": [], "step": []}

            for _ in range(n_trials):
                t0 = time.perf_counter()
                env.reset()
                times["reset"].append(time.perf_counter() - t0)

                actions = _fixed_actions(batch_size)
                t0 = time.perf_counter()
                env.step(actions, phase="play")
                times["step"].append(time.perf_counter() - t0)

            results[label] = times

        print(f"  {'Operation':<12} {'Local (ms)':<20} {'Remote (ms)':<20} "
              f"{'Overhead (ms)':<15}")
        print("  " + "-" * 67)
        for op in ("reset", "step"):
            local_arr = np.array(results["local"][op]) * 1000
            remote_arr = np.array(results["remote"][op]) * 1000
            overhead = np.median(remote_arr) - np.median(local_arr)
            print(
                f"  {op:<12} "
                f"p50={np.median(local_arr):6.2f}  "
                f"p95={np.percentile(local_arr, 95):6.2f}  "
                f"p50={np.median(remote_arr):6.2f}  "
                f"p95={np.percentile(remote_arr, 95):6.2f}  "
                f"{overhead:+.2f}"
            )

        remote_env.close()
        print("PASSED\n")
    finally:
        server_proc.kill()
        server_proc.join(timeout=5)


def test_throughput():
    """
    Run N complete episodes through local and remote, measure wall-clock.
    """
    print("\n=== TEST: throughput benchmark ===")
    batch_size = 4
    config = _load_config(batch_size)
    n_episodes = 20
    max_turns = config.env.get("max_turns", 7)

    import ray
    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    _, local_env = _make_local_envs(config)

    port = PORT_BASE + 3
    server_proc = _start_server_process(config, port, is_train=False)
    try:
        remote_env = _connect_client(port)

        for label, env in [("local", local_env), ("remote", remote_env)]:
            t0 = time.perf_counter()
            for _ in range(n_episodes):
                env.reset()
                for _ in range(max_turns):
                    actions = _fixed_actions(batch_size)
                    env.step(actions, phase="play")
            elapsed = time.perf_counter() - t0
            eps_per_sec = n_episodes / elapsed
            print(f"  {label:<8}: {n_episodes} episodes in {elapsed:.2f}s "
                  f"({eps_per_sec:.1f} ep/s)")

        remote_env.close()
        print("PASSED\n")
    finally:
        server_proc.kill()
        server_proc.join(timeout=5)


def test_large_batch():
    """
    Test with batch_size=64 to verify large-payload serialization.
    """
    print("\n=== TEST: large batch serialization ===")
    batch_size = 64
    config = _load_config(batch_size)

    import ray
    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    port = PORT_BASE + 4
    server_proc = _start_server_process(config, port, is_train=False)
    try:
        remote_env = _connect_client(port)

        assert remote_env.num_processes == batch_size, (
            f"Expected {batch_size}, got {remote_env.num_processes}"
        )

        obs, infos = remote_env.reset()
        assert len(obs["text"]) == batch_size
        assert len(infos) == batch_size

        actions = _fixed_actions(batch_size)
        next_obs, rewards, dones, infos = remote_env.step(actions, phase="play")
        assert len(next_obs["text"]) == batch_size
        assert rewards.shape == (batch_size,)
        assert dones.shape == (batch_size,)
        assert len(infos) == batch_size

        print(f"  batch_size={batch_size}: OK")

        remote_env.close()
        print("PASSED\n")
    finally:
        server_proc.kill()
        server_proc.join(timeout=5)


def test_reconnection():
    """
    Verify the client raises an error when the server dies, and can
    reconnect to a restarted server.
    """
    print("\n=== TEST: reconnection ===")
    batch_size = 2
    config = _load_config(batch_size)

    import ray
    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    port = PORT_BASE + 5

    # Start server, connect, do a step
    server_proc = _start_server_process(config, port, is_train=False)
    try:
        remote_env = _connect_client(port, timeout=10.0)
        remote_env.reset()
        actions = _fixed_actions(batch_size)
        remote_env.step(actions, phase="play")
        print("  initial connection: OK")

        # Kill server
        server_proc.kill()
        server_proc.join(timeout=5)
        print("  server killed")

        # Client should fail
        try:
            remote_env.step(actions, phase="play")
            print("  ERROR: expected ConnectionError")
        except (ConnectionError, RuntimeError, OSError):
            print("  expected error raised: OK")

    finally:
        if server_proc.is_alive():
            server_proc.kill()
            server_proc.join(timeout=5)

    # Restart server, reconnect with a new client
    server_proc2 = _start_server_process(config, port, is_train=False)
    try:
        remote_env2 = _connect_client(port)
        remote_env2.reset()
        actions = _fixed_actions(batch_size)
        remote_env2.step(actions, phase="play")
        print("  reconnected to new server: OK")
        remote_env2.close()
        print("PASSED\n")
    finally:
        server_proc2.kill()
        server_proc2.join(timeout=5)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

ALL_TESTS = {
    "correctness": test_correctness,
    "metarl": test_metarl,
    "latency": test_latency,
    "throughput": test_throughput,
    "large_batch": test_large_batch,
    "reconnection": test_reconnection,
}


def main():
    parser = argparse.ArgumentParser(
        description="Test suite for the remote environment server."
    )
    parser.add_argument(
        "--test", type=str, default=None,
        help=f"Run a specific test: {list(ALL_TESTS)}",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all tests",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if not args.all and args.test is None:
        parser.print_help()
        sys.exit(1)

    # Use 'spawn' to avoid issues with Ray fork.
    multiprocessing.set_start_method("spawn", force=True)

    tests_to_run = ALL_TESTS if args.all else {args.test: ALL_TESTS[args.test]}

    passed = 0
    failed = 0
    for name, fn in tests_to_run.items():
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"FAILED: {name} — {exc}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
