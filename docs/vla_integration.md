# Plan: Integrate Pre-trained LAVA Policy into Language Table

## Overview

Replace the dummy random-action inner-loop policy in
`language_table/lamer/env_manager.py` with the pre-trained LAVA
(Language-Augmented Visual Attention) checkpoint from the language-table
project. The VLA runs co-located with the env server on a single GPU,
performing batched inference for all environments at each inner step.

---

## Phase 1: Standalone LAVA Evaluation — COMPLETE

**Goal**: Verify the pre-trained checkpoint works in our setup before
integrating it into the training loop.

### Step 1.1: `language_table/lamer/lava_policy.py` — DONE

Self-contained wrapper that loads the LAVA checkpoint and exposes
batched `predict(goals, obs_list, active_mask) -> actions`.

Replicates the exact inference pipeline from the original eval code:
1. **ClipTokenWrapper** — CLIP-tokenizes the instruction string (cached)
2. **CentralCropImageWrapper** — `tf.image.convert_image_dtype` (uint8→float32),
   central crop (0.95), `tf.image.resize` to (180, 320) — uses TF ops for
   bit-identical results with the original eval code
3. **HistoryWrapper(4, tile_first_step_obs=True)** — per-env deque of 4 frames
4. **BCJaxPyPolicy** — batched `model.apply`, denormalize with
   `action * max(std, EPS) + mean`, clip to `[-0.1, 0.1]`

Interface:
```python
class LAVAPolicy:
    def __init__(self, checkpoint_dir, checkpoint_prefix="bc_resnet_sim_checkpoint_",
                 model_config=None, sequence_length=4, action_clip=0.1): ...
    def reset(self, num_envs): ...
    def predict(self, goals, obs_list, active_mask) -> List[np.ndarray]: ...
```

### Step 1.2: `language_table/lamer/test_lava_standalone.py` — DONE

Single-env evaluation script. Tests per reward type with configurable
episodes, max steps, and block mode. Optionally saves videos.

**Test results** (BLOCK_4, 10 episodes, 200 steps, block2block):
- Success rate: **50%** (5/10)
- Confirms checkpoint loads, preprocessing matches training, policy is meaningful

### Step 1.3: `language_table/lamer/test_lava_batched.py` — DONE

Batched evaluation using `LanguageTableMultiProcessEnv` with Ray workers.

**Test results** (BLOCK_4, 4 envs, 100 steps):
- Success rate: 25% (1/4)
- GPU throughput: **94 env-steps/s** (38.6ms/step mean, 4 envs)
- CPU throughput: 17.2 env-steps/s (for comparison)

---

## Phase 2: Integrate LAVA into Environment Manager — COMPLETE

### Step 2.1: `language_table/lamer/env_manager.py` — DONE

Added optional `vla_policy` parameter to `LanguageTableEnvironmentManager`:

```python
class LanguageTableEnvironmentManager:
    def __init__(self, envs, ..., vla_policy=None, include_rgb=False):
        self.vla = vla_policy
        self._include_rgb = include_rgb  # controls TCP to LLM client only
```

Key changes:
- `vla.reset(num_envs)` called in `reset()` and `restart()` to clear frame buffers
- `_handle_play_step` uses `vla.predict(goal_strings, obs_list, active_mask)` when
  VLA is available, falls back to random actions with a warning when not
- `_last_obs_list` cached from `reset()`/`restart()` to seed the VLA's first
  predict call (bridging the gap between env reset and inner loop start — see
  design note below)
- `_extract_images()` gated on `include_rgb` flag (only for TCP to LLM, not VLA)

**Observation flow** (matches original eval pipeline):
```
Original:  ts = env.reset()  →  policy.action(ts)  →  ts = env.step(a)  →  policy.action(ts)  → ...
Ours:      obs = envs.reset() → [cached] → LLM thinks → vla.predict(cached_obs) → obs = envs.step(a) → vla.predict(obs) → ...
```

The only difference is the gap between `reset()` and the first `predict()` where
the LLM generates a goal string. After that, it's the same tight obs→action loop.

**Tested**: reset → play → restart → play cycle with VLA producing real actions
and achieving task success (reward=100.0 on matching goals).

### Step 2.1b: `language_table/lamer/envs.py` — DONE

Cleaned up the `render_obs` / `include_rgb` separation:

| Flag | Where | Controls |
|------|-------|----------|
| `render_obs` | Ray worker (`LanguageTableWorker`) | Whether PyBullet renders RGB AND includes it in obs returned to env manager |
| `include_rgb` | Env manager (`LanguageTableEnvironmentManager`) | Whether RGB images are included in responses sent over TCP to the LLM client |

Previously `include_rgb` was on the worker level and conflated both concerns.
Now `render_obs=True` is sufficient for the VLA — it renders and returns RGB
from workers to the env manager (local Ray communication). The `include_rgb`
flag on the env manager only controls whether images go over the wire to the
remote LLM, which is typically `False` (the outer LLM doesn't need images).

Typical configurations:
- **No VLA, text-only LLM**: `render_obs=False, include_rgb=False` — fastest
- **VLA enabled, text-only LLM** (our case): `render_obs=True, include_rgb=False`
- **VLA enabled, multimodal LLM**: `render_obs=True, include_rgb=True`

### Step 2.2: `language_table/lamer/server_main.py` — DONE

Added `--vla_checkpoint` CLI flag:
```bash
--vla_checkpoint /path/to/checkpoints/bc_resnet_sim_checkpoint_955000
```

When set:
- Auto-parses `checkpoint_dir` and `checkpoint_prefix` from the full path
- Forces `render_obs=True` (VLA needs images from Ray workers)
- Loads `LAVAPolicy` and passes it to the env manager

### Step 2.2b: `requirements.txt` — DONE

Updated dependencies for GPU support:

| Package | Before | After | Why |
|---------|--------|-------|-----|
| `jax` | 0.4.17 | **0.4.30 [cuda12]** | GPU support via CUDA plugin (cuDNN 9 compatible) |
| `jaxlib` | 0.4.17 (CPU) | **0.4.30** | Matching jax version |
| `flax` | 0.6.10 | **0.8.5** | 0.6.10 incompatible with JAX 0.4.30 (`define_bool_state` removed) |
| `setuptools` | (any) | **<70** | `pkg_resources` needed by `tensorflow_hub`, removed in setuptools>=70 |

Install from scratch:
```bash
uv venv --python 3.10 ./ltvenv
uv pip install -r ./requirements.txt
uv pip install --no-deps git+https://github.com/google-research/scenic.git@ae21d9e884015aa7bc7cf1d489af53d16c249726
export PYTHONPATH=${PWD}:$PYTHONPATH
```

### Step 2.3: SLURM script — DONE

Updated `scripts/slurm/lamer_language_table.slurm`:
- Added `VLA_CHECKPOINT` config variable
- Added `--vla_checkpoint` to both train and val server launch commands
- Added `--no_render` to validation server when VLA is disabled (throughput optimization)

---

## Phase 3: End-to-End Validation

### Step 3.1: `language_table/lamer/test_integration.py`

Integration test that:
1. Starts an env server with VLA enabled (in-process, no TCP)
2. Runs the full meta-RL loop: reset → step(goal) → reflect → restart → step(new_goal)
3. Verifies:
   - VLA actions produce meaningful state changes (not random)
   - Rewards are non-trivial
   - Reflection prompts include episode outcomes
   - Different goal strings produce different behavior

### Step 3.2: Small-scale training smoke test

Run the full SLURM script with:
- 4 train envs, 8 val envs (small scale)
- VLA enabled
- 1-2 training epochs
- Verify:
  - Training loop completes without errors
  - Rewards are non-trivial (VLA actually accomplishes some tasks)
  - LLM learns to propose different goals across reflection attempts

---

## Key Design Decisions

1. **Co-located VLA (not remote)**: The VLA runs in the same process as
   the env manager. With 100 inner steps per outer step, remote VLA would
   add unacceptable network overhead (serializing 128 RGB images per step).

2. **VLA sees LLM goals, not env instructions**: The LLM's text output
   is used as the VLA's language conditioning. This is the whole point —
   the LLM can propose goals that differ from the env's original task
   (e.g., sub-goals, reworded instructions, or novel strategies after
   reflection).

3. **Frame history managed by LAVAPolicy**: The policy maintains a
   per-env deque of 4 frames. On reset/restart, buffers are cleared and
   the first frame is tiled to fill the sequence (matching the eval
   wrapper's `tile_first_step_obs=True` behavior).

4. **Graceful fallback**: When no checkpoint is provided, the existing
   random-action behavior is preserved (with a warning). This keeps the
   codebase usable for testing without a checkpoint.

5. **render_obs vs include_rgb**: `render_obs` controls whether Ray workers
   render and return RGB (for VLA, local). `include_rgb` controls whether
   images are sent over TCP to the LLM client (remote). These are independent
   concerns — the VLA needs images but the LLM typically doesn't.

---

## Resource Requirements

### Current setup (no VLA):
- 4 GPUs for LLM training (Qwen3-4B with FSDP + vLLM)
- CPU for env servers (PyBullet + Ray workers)

### With VLA:
- 4 GPUs for LLM training (unchanged)
- 1 GPU for VLA inference (LAVA is small, ~50M params)
  - Can potentially share a GPU with one of the training processes
  - Or run on CPU with acceptable throughput (JAX CPU JIT)
- CPU for env servers (PyBullet + Ray workers, unchanged)

### Measured throughput (4 envs):
- GPU: 94 env-steps/s (38.6ms/step), JIT compile ~7s first call
- CPU: 17.2 env-steps/s (230ms/step)
- With 100 inner steps per outer step: ~4s on GPU, ~13s on CPU
- Acceptable — the LLM generation step takes ~5-30s anyway

---

## File Changes Summary

### New files (in language-table repo):
```
language_table/lamer/lava_policy.py          # LAVA wrapper (~310 lines)
language_table/lamer/test_lava_standalone.py  # Single-env eval (~240 lines)
language_table/lamer/test_lava_batched.py     # Batched eval (~154 lines)
language_table/lamer/test_integration.py      # End-to-end test (Phase 3)
```

### Modified files (in language-table repo):
```
language_table/lamer/env_manager.py   # VLA in _handle_play_step, include_rgb separation
language_table/lamer/envs.py          # render_obs controls both rendering and RGB inclusion
language_table/lamer/server_main.py   # --vla_checkpoint CLI arg
requirements.txt                      # JAX 0.4.30+CUDA, Flax 0.8.5, setuptools<70
```

### Modified files (in LaMer repo):
```
scripts/slurm/lamer_language_table.slurm  # --vla_checkpoint flag, GPU allocation
```
