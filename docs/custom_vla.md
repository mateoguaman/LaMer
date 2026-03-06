# Integrating a Custom VLA (Inner-Loop Policy)

This guide explains how to replace the default LAVA policy with a different
Vision-Language-Action model for the inner loop of the Language Table
meta-RL setup.

## Architecture Overview

In the LaMer + Language Table integration, there are two loops:

- **Outer loop (LLM)**: Proposes natural-language goal strings, reflects on
  outcomes, and adapts across meta-RL attempts.
- **Inner loop (VLA)**: Receives a goal string + RGB observations from the
  environment, and produces low-level 2D displacement actions at each
  timestep. Runs for `max_inner_steps` (default 100) steps per outer step.

The VLA runs **co-located** with the environment manager (same process, same
GPU). It is called once per inner step with a batch of observations from all
parallel environments.

## The VLA Interface

The environment manager (`language_table/lamer/env_manager.py`) expects a
VLA policy object with two methods:

```python
class YourVLAPolicy:
    def reset(self, num_envs: int) -> None:
        """Called when environments reset or restart.

        Clear any internal state (frame buffers, hidden states, etc.)
        for all environments. After this call, the policy should be
        ready to start a fresh episode for `num_envs` environments.
        """
        ...

    def predict(
        self,
        goals: List[str],
        obs_list: List[Dict[str, Any]],
        active_mask: np.ndarray,
    ) -> List[np.ndarray]:
        """Produce one action per environment.

        Parameters
        ----------
        goals : list[str], length num_envs
            Natural-language goal for each environment. These come from
            the outer-loop LLM and may differ from the environment's
            original task instruction.

        obs_list : list[dict], length num_envs
            Per-environment observation dicts. When render_obs=True,
            each dict contains at minimum:
              - "rgb": np.ndarray, shape (180, 320, 3), dtype uint8
                  The camera image from the environment.
            It also contains block positions, end-effector state, and
            the environment's own instruction (as an int32 array).

        active_mask : np.ndarray, shape (num_envs,), dtype bool
            True for environments that still need actions. False for
            environments that have already terminated (done=True).
            You can skip computation for inactive environments.

        Returns
        -------
        list[np.ndarray]
            One action per environment. Each action is a (2,) float32
            array representing a 2D displacement in [-0.1, 0.1].
            Actions for inactive environments should be zeros.
        """
        ...
```

## Step-by-Step Integration

### 1. Create your policy wrapper

Create a new file (e.g., `language_table/lamer/your_policy.py`) that
implements the interface above. Key considerations:

- **Preprocessing**: The RGB images from the environment are 180x320x3 uint8.
  Your model likely expects a different format (float32, different resolution,
  normalized, etc.). Handle this in `predict()`.

- **Frame history**: If your model uses a sequence of frames (like LAVA uses 4),
  maintain per-environment frame buffers (e.g., `collections.deque`). Clear
  them in `reset()`. On the first call after reset, tile the first frame to
  fill the buffer.

- **Batched inference**: For throughput, process all active environments in a
  single forward pass rather than looping. With 128 environments and 100 inner
  steps, this is the difference between seconds and minutes.

- **Action space**: Language Table uses 2D displacement actions in `[-0.1, 0.1]`.
  If your model outputs actions in a different range or format, map them to
  this space. The environment will clip to `[-0.1, 0.1]` regardless, but
  producing out-of-range actions wastes the model's output resolution.

- **Language conditioning**: The `goals` parameter contains the LLM's
  proposed goal strings, which may differ from the environment's original
  task instruction. This is intentional — the LLM sees the env state, and
  after reflecting on failures, it may propose sub-goals ("first move the
  red block out of the way"), reworded instructions, or entirely novel
  strategies. Your model must handle arbitrary natural-language strings as
  conditioning, not just the fixed set of instructions the env generates.

### 2. Update `server_main.py`

In `language_table/lamer/server_main.py`, add your model's CLI arguments
and loading logic. The current code handles this for LAVA:

```python
if args.vla_checkpoint:
    from .lava_policy import LAVAPolicy
    vla_policy = LAVAPolicy(
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
    )
```

For a different model, you might add:

```python
parser.add_argument("--vla_type", type=str, default="lava",
                    choices=["lava", "openvla", "rt2", ...])

# Then in the loading section:
if args.vla_type == "your_model":
    from .your_policy import YourVLAPolicy
    vla_policy = YourVLAPolicy(checkpoint_path=args.vla_checkpoint, ...)
```

### 3. Ensure `render_obs=True`

Any vision-based VLA needs RGB images from the Ray workers. The server
automatically forces `render_obs=True` when `--vla_checkpoint` is provided.
If your model does NOT need images (e.g., it's a text-only policy that
reasons over block positions), you can set `render_obs=False` for faster
throughput.

### 4. GPU considerations

The VLA runs in the env server process. For GPU inference:

- **JAX models** (like LAVA): JIT-compile the forward pass for best throughput.
  The current setup uses `jax[cuda12]==0.4.30`.
- **PyTorch models**: Use `torch.no_grad()` and `.eval()` mode. PyTorch 2.10
  with CUDA is already in the venv.
- **Multi-GPU**: By default, the VLA will use whatever GPU is visible. On SLURM,
  use `CUDA_VISIBLE_DEVICES` to assign a specific GPU to the env server process
  if needed (the LLM training uses the other GPUs).

## Reference: LAVA Implementation

See `language_table/lamer/lava_policy.py` for a complete example. It
demonstrates:

- Loading a Flax checkpoint and JIT-compiling inference
- CLIP tokenization with caching
- TF-based image preprocessing (crop + resize) matching the training pipeline
- Per-environment frame history buffers with `collections.deque`
- Batched forward pass producing actions for all environments at once
- Action denormalization using statistics from the checkpoint

## Testing

Use the existing test scripts to validate your policy:

```bash
# Single-env evaluation (modify to import your policy instead of LAVAPolicy)
ltvenv/bin/python -m language_table.lamer.test_lava_standalone \
    --checkpoint_dir /path/to/your/checkpoint --num_episodes 10

# Integration test (full meta-RL loop)
ltvenv/bin/python -m language_table.lamer.test_integration \
    --checkpoint_dir /path/to/your/checkpoint --num_envs 4
```

You should see:
- Non-zero success rates (better than random ~0-5%)
- Different rewards for different goal strings
- Reasonable throughput (>10 env-steps/s on GPU for the inner loop)
