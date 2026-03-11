#!/usr/bin/env python3
"""
Smoke test for the NavigationSingle environment.

Exercises the full MetaRL rollout loop (reset → step × n → reflect → restart)
using fake 1-char LLM responses — no GPU or model required.

Run from the repo root:
    python tests/test_navigation_single.py
"""

import random
import sys
import traceback

import numpy as np
from omegaconf import OmegaConf, DictConfig


# --------------------------------------------------------------------------- #
# Config builders mirroring the two scripts                                   #
# --------------------------------------------------------------------------- #

def _base_config(
    batch_size: int = 2,
    group_size: int = 2,
    nav_n: int = 4,
    num_attempts: int = 3,
    max_steps: int = 7,
    train_disturbances: list = None,
    val_disturbances: list = None,
    reflection_type: str = "reflection_only",
    do_reflection: bool = True,
    seed: int = 42,
) -> DictConfig:
    return OmegaConf.create({
        "data": {
            "train_batch_size": batch_size,
            "val_batch_size":   batch_size,
        },
        "env": {
            "env_name":        "NavigationSingle",
            "seed":            seed,
            "rollout":         {"n": group_size},
            "num_attempts":    num_attempts,
            "do_reflection":   do_reflection,
            "max_steps":       max_steps,
            "max_turns":       1,
            "reflection_type": reflection_type,
            "navigation": {
                "n":                  nav_n,
                "train_disturbances": train_disturbances or ["NoDisturbance"],
                "val_disturbances":   val_disturbances   or ["NoDisturbance"],
            },
        },
    })


# --------------------------------------------------------------------------- #
# Fake LLM response generators                                                #
# --------------------------------------------------------------------------- #

def _fake_play(_prompt: str) -> str:
    move = random.choice("UDLR")
    return f"I think the goal is to the left.\n<action>{move}</action>"


def _fake_reflect(_prompt: str) -> str:
    return (
        "I did not reach the goal. The action mapping might be adversarial.\n"
        "<remark>I will try a different direction and observe where I end up "
        "to deduce the true action mapping.</remark>"
    )


def _fake_actions(phase: str, prompts: list) -> list:
    return [
        _fake_play(p) if phase == "play" else _fake_reflect(p)
        for p in prompts
    ]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_failures = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  [{PASS}] {name}")
    else:
        msg = f"  [{FAIL}] {name}" + (f": {detail}" if detail else "")
        print(msg)
        _failures.append(name)


def assert_check(name: str, fn):
    """Run fn(); catch any exception as a test failure."""
    try:
        fn()
        print(f"  [{PASS}] {name}")
    except Exception as exc:
        _failures.append(name)
        print(f"  [{FAIL}] {name}: {exc}")
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# Core loop: runs one full MetaRL episode and returns metrics                 #
# --------------------------------------------------------------------------- #

def run_episode(envs, num_attempts: int, max_steps: int, verbose: bool = False):
    """
    Run one MetaRL episode (reset + attempts) and return
    (episode_rewards, episode_lengths, is_won, all_prompts_flat).

    all_prompts_flat: every prompt seen by env[0] across all steps/phases.
    """
    batch_size = envs.num_processes
    is_done    = np.zeros(batch_size, dtype=bool)
    is_won     = np.zeros(batch_size, dtype=bool)
    ep_rewards = np.zeros(batch_size, dtype=np.float32)
    ep_lengths = np.zeros(batch_size, dtype=np.int32)
    all_prompts = []

    # Build phase schedule: (attempt_idx, phase)
    phase_schedule = []
    for attempt_idx in range(num_attempts):
        if attempt_idx > 0:
            if envs.do_reflection:
                phase_schedule.append((attempt_idx, "reflect"))
            phase_schedule.append((attempt_idx, "play"))
        else:
            phase_schedule.append((attempt_idx, "play"))

    obs, _ = envs.reset()

    for attempt_idx, phase in phase_schedule:
        if phase == "reflect":
            obs, _ = envs.reflect()
            is_done = is_won.copy()

        if phase == "play" and attempt_idx > 0:
            obs, _ = envs.restart()
            is_done = is_won.copy()

        for _step in range(max_steps):
            active = ~is_done
            if not active.any():
                break

            prompts = obs.get("text") or [""] * batch_size
            all_prompts.append(prompts[0])

            actions = _fake_actions(phase, prompts)
            next_obs, rewards, dones, infos = envs.step(actions, phase=phase)

            rewards_np = np.asarray(rewards, dtype=np.float32).ravel()
            dones_np   = np.asarray(dones,   dtype=bool).ravel()
            wons       = np.array([bool(info.get("won", False)) for info in infos])

            ep_rewards += rewards_np * active
            ep_lengths[active] += 1
            is_done = np.logical_or(is_done, dones_np)
            is_won  = np.logical_or(is_won,  wons)

            if verbose:
                print(f"    attempt={attempt_idx+1} phase={phase} step={_step+1} "
                      f"rewards={rewards_np.round(3).tolist()} dones={dones_np.tolist()}")

            obs = next_obs

            if phase == "play" and is_done.all():
                break

    return ep_rewards, ep_lengths, is_won, all_prompts


# --------------------------------------------------------------------------- #
# Individual test functions                                                    #
# --------------------------------------------------------------------------- #

def test_import():
    print("\n=== test_import ===")
    assert_check(
        "import navigation_single make_envs",
        lambda: __import__(
            "agent_system.environments.navigation_single",
            fromlist=["make_envs"]
        ),
    )
    assert_check(
        "import navigation_single prompt",
        lambda: __import__(
            "agent_system.environments.navigation_single.prompt",
            fromlist=["get_navigation_single_step_prompt"],
        ),
    )


def test_prompt_format():
    print("\n=== test_prompt_format ===")
    from agent_system.environments.navigation_single.prompt import (
        get_navigation_single_step_prompt,
    )

    prompt = get_navigation_single_step_prompt(
        n=4, grid_size=9, phase="play",
        turn_idx=0, traj_idx=0,
        init_observation="Agent: (4,4)  Goal: (4,0)  Steps remaining: 4\n.....",
    )

    check("prompt is non-empty", len(prompt) > 0)
    check(
        "prompt asks for 1 character",
        "1 character" in prompt or "1 char" in prompt.lower(),
        f"not found in: {prompt[:200]}",
    )
    check(
        "prompt shows single-char example",
        "<action>L</action>" in prompt or "<action>U</action>" in prompt
        or "<action>R</action>" in prompt or "<action>D</action>" in prompt,
    )
    check(
        "prompt does NOT ask for n=4 characters as the action count",
        "exactly 4 characters" not in prompt,
    )
    check(
        "prompt still mentions n=4 path length for context",
        "4 steps" in prompt or "4-step" in prompt,
        "path context missing",
    )

    # reflect phase — reuses navigation's reflect prompt unchanged
    reflect_prompt = get_navigation_single_step_prompt(
        n=4, grid_size=9, phase="reflect",
        turn_idx=0, traj_idx=0,
        init_observation="Agent: (4,4)  Goal: (4,0)  Steps remaining: 4\n.....",
    )
    check("reflect prompt non-empty", len(reflect_prompt) > 0)
    check("reflect prompt has <remark> hint", "<remark>" in reflect_prompt)


def test_projection_extracts_one_char():
    print("\n=== test_projection_extracts_one_char ===")
    from agent_system.environments.navigation.projection import navigation_projection
    from functools import partial

    proj = partial(navigation_projection, n_remaining=1)

    # Multi-char input → only first char extracted
    thoughts, actions, valids = proj(
        ["I think LUDR is the path. <action>LUDR</action>",
         "Going right. <action>R</action>",
         "No tags here, just text.",
         "Empty action: <action></action>"],
        phase="play",
    )
    check("multi-char input → 1 char extracted", actions[0] == "L",
          f"got '{actions[0]}'")
    check("single-char input → 1 char extracted", actions[1] == "R",
          f"got '{actions[1]}'")
    check("missing tags → invalid (val=0)", valids[2] == 0,
          f"got valid={valids[2]}")
    check("empty action → invalid (val=0)", valids[3] == 0,
          f"got valid={valids[3]}")
    check("valid actions are marked valid", valids[0] == 1 and valids[1] == 1,
          f"got valids={valids[:2]}")


def test_make_envs_base(cfg):
    print("\n=== test_make_envs_base (NoDisturbance) ===")
    from agent_system.environments.navigation_single import make_envs

    assert_check("make_envs returns without error", lambda: make_envs(cfg))
    envs, val_envs = make_envs(cfg)

    check("train envs have correct num_processes",
          envs.num_processes == cfg.data.train_batch_size * cfg.env.rollout.n)
    check("val envs have num_processes == val_batch_size",
          val_envs.num_processes == cfg.data.val_batch_size)
    check("n matches config", envs.n == cfg.env.navigation.n)
    check("grid_size == 2n+1", envs.grid_size == 2 * cfg.env.navigation.n + 1)
    check("num_attempts matches config", envs.num_attempts == cfg.env.num_attempts)
    check("get_traj_log returns list", isinstance(envs.get_traj_log(), list))

    envs.close()
    val_envs.close()


def test_single_step_episode(cfg, label="base"):
    print(f"\n=== test_single_step_episode [{label}] ===")
    from agent_system.environments.navigation_single import make_envs

    envs, val_envs = make_envs(cfg)
    batch_size = envs.num_processes

    ep_rewards, ep_lengths, is_won, all_prompts = run_episode(
        envs,
        num_attempts=cfg.env.num_attempts,
        max_steps=cfg.env.max_steps,
        verbose=False,
    )

    check("episode ran without crash", True)
    check("ep_rewards has correct shape", ep_rewards.shape == (batch_size,))
    check("ep_lengths has correct shape", ep_lengths.shape == (batch_size,))
    check("at least one step taken", ep_lengths.sum() > 0,
          f"ep_lengths={ep_lengths.tolist()}")
    check("rewards are finite", np.isfinite(ep_rewards).all(),
          f"rewards={ep_rewards.tolist()}")
    check("at least one prompt was seen", len(all_prompts) > 0)

    # Prompt must request a single char
    first_prompt = all_prompts[0]
    check(
        "all episode prompts ask for 1 character",
        all("1 character" in p or "1 char" in p.lower() for p in all_prompts),
    )

    # Later prompts (after first step) should contain trajectory history
    if len(all_prompts) > 1:
        check(
            "later prompts contain trajectory history (Action / Observation)",
            any("Action" in p and "Observation" in p for p in all_prompts[1:]),
            "history not found in any prompt after step 1",
        )

    check("get_traj_log returns []", envs.get_traj_log() == [])

    envs.close()
    val_envs.close()


def test_memory_accumulates(cfg):
    print("\n=== test_memory_accumulates ===")
    from agent_system.environments.navigation_single import make_envs

    envs, _ = make_envs(cfg)

    obs, _ = envs.reset()
    prompt_step0 = obs["text"][0]

    # Step once
    actions = _fake_actions("play", obs["text"])
    obs, _, _, _ = envs.step(actions, phase="play")
    prompt_step1 = obs["text"][0]

    # Step again
    actions = _fake_actions("play", obs["text"])
    obs, _, _, _ = envs.step(actions, phase="play")
    prompt_step2 = obs["text"][0]

    check("step-0 prompt has no history",
          "Action 1" not in prompt_step0 and "Observation 1" not in prompt_step0)
    check("step-1 prompt contains step-1 history",
          "Action 1" in prompt_step1 and "Observation 1" in prompt_step1,
          f"not found in: {prompt_step1[:300]}")
    check("step-2 prompt contains step-2 history",
          "Action 2" in prompt_step2 and "Observation 2" in prompt_step2,
          f"not found in: {prompt_step2[:300]}")

    envs.close()


def test_metarl_reflect_restart(cfg):
    print("\n=== test_metarl_reflect_restart ===")
    from agent_system.environments.navigation_single import make_envs

    cfg2 = OmegaConf.merge(cfg, OmegaConf.create({"env": {"num_attempts": 2}}))
    envs, _ = make_envs(cfg2)

    # Attempt 1
    obs, _ = envs.reset()
    actions = _fake_actions("play", obs["text"])
    obs, rewards1, dones1, _ = envs.step(actions, phase="play")

    check("attempt-1 rewards are finite",
          np.isfinite(np.asarray(rewards1, dtype=float)).all())

    # Reflect
    assert_check("reflect() does not crash", lambda: envs.reflect())
    obs_reflect, _ = envs.reflect()
    check("reflect obs has text", obs_reflect["text"] is not None)
    check("reflect prompt has <remark>", "<remark>" in obs_reflect["text"][0])

    reflect_actions = _fake_actions("reflect", obs_reflect["text"])
    assert_check("step(phase='reflect') does not crash",
                 lambda: envs.step(reflect_actions, phase="reflect"))

    # Restart (attempt 2)
    assert_check("restart() does not crash", lambda: envs.restart())
    obs2, _ = envs.restart()
    actions2 = _fake_actions("play", obs2["text"])
    _, rewards2, _, _ = envs.step(actions2, phase="play")

    check("attempt-2 rewards are finite",
          np.isfinite(np.asarray(rewards2, dtype=float)).all())

    # After restart curr_traj_idx should be 1 (or 2 if reflect incremented it)
    check("curr_traj_idx > 0 after restart",
          envs.curr_traj_idx > 0,
          f"curr_traj_idx={envs.curr_traj_idx}")

    envs.close()


def test_disturbances_loaded(cfg_meta):
    print("\n=== test_disturbances_loaded (FlipLeftRight+FlipUpDown) ===")
    from agent_system.environments.navigation_single import make_envs

    assert_check("make_envs with disturbances does not crash",
                 lambda: make_envs(cfg_meta))
    envs, val_envs = make_envs(cfg_meta)
    check("train envs created", envs.num_processes > 0)
    envs.close()
    val_envs.close()


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main():
    random.seed(0)
    np.random.seed(0)

    # Config for lamer_nav_base_4_step_single.sh
    cfg_base = _base_config(
        batch_size=2, group_size=2, nav_n=4,
        num_attempts=3, max_steps=7,
        train_disturbances=["NoDisturbance"],
        val_disturbances=["NoDisturbance"],
        reflection_type="reflection_only",
    )

    # Config for lamer_nav_meta_4_step_single_history_and_reflection.sh
    cfg_meta = _base_config(
        batch_size=2, group_size=2, nav_n=4,
        num_attempts=3, max_steps=7,
        train_disturbances=["FlipLeftRight", "FlipUpDown"],
        val_disturbances=["FlipBoth"],
        reflection_type="history_and_reflection",
    )

    test_import()
    test_prompt_format()
    test_projection_extracts_one_char()
    test_make_envs_base(cfg_base)
    test_single_step_episode(cfg_base, label="base/NoDisturbance")
    test_single_step_episode(cfg_meta, label="meta/FlipLR+FlipUD")
    test_memory_accumulates(cfg_base)
    test_metarl_reflect_restart(cfg_base)
    test_disturbances_loaded(cfg_meta)

    print()
    print("=" * 60)
    if _failures:
        print(f"FAILED  {len(_failures)} test(s):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
