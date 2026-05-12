#!/usr/bin/env python3
"""
Standalone Language Table rollout using a remote vLLM server (OpenAI-compatible API).

Reuses LanguageTableEnvironmentManager for prompt construction and action projection,
but calls an external vLLM server instead of running an LLM locally. Single-attempt
only (no MetaRL reflection), text-only (images ignored).

Usage:
    python examples/language_table/api_rollout.py \
        --remote_address HOST:PORT \
        --vllm_url http://HOST:PORT/v1 \
        --model MODEL_NAME \
        [--val_data path/to/val.parquet] \
        [--num_episodes N] \
        [--num_envs K] \
        [--max_turns T] \
        [--temperature 0.7] \
        [--max_tokens 1024] \
        [--output path/to/results.jsonl]
"""
import argparse
import json
import math
import os
import re
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--remote_address", required=True,
                   help="HOST:PORT of the Language Table env server")
    p.add_argument("--vllm_url", required=True,
                   help="Base URL of vLLM server, e.g. http://HOST:PORT/v1")
    p.add_argument("--model", required=True,
                   help="Model name as served by vLLM")
    p.add_argument("--val_data",
                   default=os.path.expanduser("~/data/verl-agent/text/test.parquet"),
                   help="Path to val parquet (used only to determine episode count)")
    p.add_argument("--num_episodes", type=int, default=None,
                   help="Number of episodes to run (default: all rows in parquet)")
    p.add_argument("--num_envs", type=int, default=None,
                   help="Parallel environments per batch (default: server num_processes)")
    p.add_argument("--max_turns", type=int, default=5,
                   help="Max turns per episode (additional cap on top of server limit)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--output", type=str, default=None,
                   help="Path to write JSONL results")
    p.add_argument("--video_dir", type=str, default=None,
                   help="Directory to save per-episode MP4 videos")
    p.add_argument("--no_think", action="store_true", default=True,
                   help="Append /no_think to suppress Qwen3 chain-of-thought (default: on)")
    p.add_argument("--think", dest="no_think", action="store_false",
                   help="Allow Qwen3 chain-of-thought reasoning")
    p.add_argument("--human", action="store_true", default=False,
                   help="Read action commands from stdin instead of querying the LLM; "
                        "the same command is broadcast to all active envs each step")
    return p.parse_args()


def build_env(remote_address):
    from omegaconf import OmegaConf
    from agent_system.environments.remote.client import RemoteEnvironmentManager
    from agent_system.environments.language_table.env_manager import (
        LanguageTableEnvironmentManager,
    )
    config = OmegaConf.create({"env": {"reflection_type": "history_only"}})
    remote = RemoteEnvironmentManager(remote_address, timeout=300.0)
    return LanguageTableEnvironmentManager(remote, config, prompt_state=None)


def build_client(vllm_url):
    from openai import OpenAI
    return OpenAI(base_url=vllm_url, api_key="EMPTY")


def load_num_episodes(val_data, cap):
    import pandas as pd
    df = pd.read_parquet(val_data)
    n = len(df)
    if cap is not None:
        n = min(n, cap)
    return n


def extract_action(text):
    m = re.findall(r"<action>(.*?)</action>", text, re.DOTALL)
    return m[-1].strip() if m else text.strip().splitlines()[-1][:100]


def overlay_text(frames, label):
    """Burn label text into the top-left of every frame (in-place copy)."""
    from PIL import Image, ImageDraw, ImageFont
    out = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for f in frames:
        img = Image.fromarray(f)
        draw = ImageDraw.Draw(img)
        draw.text((11, 11), label, fill=(0, 0, 0), font=font)
        draw.text((10, 10), label, fill=(255, 255, 255), font=font)
        out.append(np.array(img))
    return out


def make_grid_frame(frames_per_env, step_idx):
    """Tile one frame per env into a single grid image.

    frames_per_env: list of np.ndarray (H, W, 3), one per env.
    Returns a single np.ndarray of shape (rows*H, cols*W, 3).
    """
    from PIL import Image, ImageDraw, ImageFont

    n = len(frames_per_env)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Determine cell size from first valid frame
    h, w = frames_per_env[0].shape[:2]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    grid = Image.new("RGB", (cols * w, rows * h), color=(40, 40, 40))
    for idx, frame in enumerate(frames_per_env):
        r, c = divmod(idx, cols)
        cell = Image.fromarray(frame)
        # env label
        draw = ImageDraw.Draw(cell)
        lbl = f"env{idx}"
        draw.text((6, 6), lbl, fill=(0, 0, 0), font=font)
        draw.text((5, 5), lbl, fill=(255, 220, 0), font=font)
        grid.paste(cell, (c * w, r * h))

    return np.array(grid)


def save_video(frames, path, fps=2):
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def _llm_call(client, model, prompt, max_tokens, temperature, no_think):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": not no_think}},
    )
    return resp.choices[0].message.content


def _human_call(batch_idx, step_idx, num_envs):
    """Read one command from stdin and return it for all active envs."""
    try:
        cmd = input(f"[batch {batch_idx} step {step_idx} / {num_envs} envs] command> ").strip()
    except EOFError:
        cmd = ""
    return cmd


def run_batch(env, client, model, max_turns, temperature, max_tokens,
              video_dir, batch_idx, no_think=True, human=False):
    """Run one full episode batch across all parallel envs.

    Returns a list of per-env result dicts, each with keys:
      won, total_reward, turns_taken.
    Saves one grid video per high-level LLM step if video_dir is set.
    """
    num_envs = env.num_processes
    obs, _ = env.reset()
    effective_max = min(max_turns, env.max_turns)

    total_rewards = [0.0] * num_envs
    turns_taken = [0] * num_envs
    wons = [False] * num_envs
    active = [True] * num_envs

    all_episode_grid_frames = []

    for step_idx in range(effective_max):
        prompts = obs['text']  # list[str], length = num_envs

        text_actions = [""] * num_envs
        if human:
            cmd = _human_call(batch_idx, step_idx, num_envs)
            for i in range(num_envs):
                if active[i]:
                    text_actions[i] = cmd
        else:
            # Issue all LLM calls in parallel
            with ThreadPoolExecutor(max_workers=num_envs) as pool:
                futures = {
                    pool.submit(_llm_call, client, model, prompts[i],
                                max_tokens, temperature, no_think): i
                    for i in range(num_envs)
                    if active[i]
                }
                for fut in as_completed(futures):
                    i = futures[fut]
                    text_actions[i] = fut.result()

        for i in range(num_envs):
            if active[i]:
                goal = extract_action(text_actions[i])
                print(f"  batch {batch_idx} env {i} step {step_idx}: {goal}")

        obs, rewards, dones, infos = env.step(text_actions, phase="play")

        # Build grid frames for this LLM step and accumulate for episode video
        max_inner = max((len(infos[i].get("frames", [])) for i in range(num_envs)), default=0)
        if max_inner > 0:
            step_grid_frames = []
            for inner_idx in range(max_inner):
                inner_frames = []
                for i in range(num_envs):
                    env_frames = infos[i].get("frames", [])
                    if inner_idx < len(env_frames):
                        inner_frames.append(env_frames[inner_idx])
                    else:
                        inner_frames.append(
                            env_frames[-1] if env_frames
                            else np.zeros((256, 256, 3), dtype=np.uint8)
                        )
                step_grid_frames.append(make_grid_frame(inner_frames, inner_idx))
            all_episode_grid_frames.extend(step_grid_frames)
            if video_dir:
                vid_path = os.path.join(video_dir, f"batch{batch_idx:04d}_step{step_idx:03d}_grid.mp4")
                save_video(step_grid_frames, vid_path, fps=10)
                print(f"  step video -> {vid_path}")

        for i in range(num_envs):
            if active[i]:
                total_rewards[i] += float(rewards[i])
                turns_taken[i] += 1
                wons[i] = wons[i] or bool(infos[i].get("won", False))
                if bool(dones[i]):
                    active[i] = False

        if not any(active):
            break

    if video_dir and all_episode_grid_frames:
        vid_path = os.path.join(video_dir, f"batch{batch_idx:04d}_episode_grid.mp4")
        save_video(all_episode_grid_frames, vid_path, fps=10)
        print(f"  episode video -> {vid_path}")

    return [
        {"won": wons[i], "total_reward": total_rewards[i], "turns_taken": turns_taken[i]}
        for i in range(num_envs)
    ]


def main():
    args = parse_args()
    env = build_env(args.remote_address)
    client = build_client(args.vllm_url)
    n_episodes = load_num_episodes(args.val_data, args.num_episodes)

    num_envs = env.num_processes
    if args.num_envs is not None and args.num_envs != num_envs:
        print(f"Warning: --num_envs={args.num_envs} requested but server has "
              f"{num_envs} envs; using {num_envs}.")

    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)

    n_batches = math.ceil(n_episodes / num_envs)
    results = []
    ep_counter = 0

    try:
        for batch_idx in range(n_batches):
            batch_results = run_batch(
                env, client, args.model, args.max_turns,
                args.temperature, args.max_tokens,
                args.video_dir, batch_idx, no_think=args.no_think,
                human=args.human,
            )
            for env_i, r in enumerate(batch_results):
                if ep_counter >= n_episodes:
                    break
                r["episode"] = ep_counter
                r["batch"] = batch_idx
                r["env_slot"] = env_i
                results.append(r)
                print(f"Episode {ep_counter:>4d} (batch {batch_idx} env {env_i}): "
                      f"won={r['won']}, reward={r['total_reward']:.3f}, "
                      f"turns={r['turns_taken']}")
                ep_counter += 1
    finally:
        env.close()

    if not results:
        print("No episodes completed.")
        return

    success_rate = sum(r["won"] for r in results) / len(results)
    mean_reward = sum(r["total_reward"] for r in results) / len(results)
    mean_turns = sum(r["turns_taken"] for r in results) / len(results)

    print(f"\n=== Summary ({len(results)} episodes, {num_envs} envs/batch) ===")
    print(f"  success_rate : {success_rate:.3f}")
    print(f"  mean_reward  : {mean_reward:.3f}")
    print(f"  mean_turns   : {mean_turns:.1f}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({
                "summary": True,
                "n_episodes": len(results),
                "num_envs": num_envs,
                "success_rate": success_rate,
                "mean_reward": mean_reward,
                "mean_turns": mean_turns,
            }) + "\n")
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
