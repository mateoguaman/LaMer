# PyBullet + Ray Parallelization

This doc covers two things:

1. **The env wrapper** — `agent_system/environments/pybullet/envs.py` — how to
   plug a PyBullet environment into the LaMer training loop.
2. **The benchmark script** — `examples/benchmark_pybullet_envs.py` — how to
   find the maximum number of parallel envs your machine can sustain, completely
   decoupled from LLM training.

---

## Table of contents

- [Why PyBullet works with Ray](#why-pybullet-works-with-ray)
- [Install](#install)
- [Running the benchmark](#running-the-benchmark)
  - [Quick smoke-test](#quick-smoke-test)
  - [Full PyBullet sweep](#full-pybullet-sweep)
  - [Custom env count list](#custom-env-count-list)
  - [All CLI flags](#all-cli-flags)
- [Reading the output](#reading-the-output)
- [Assumptions and defaults](#assumptions-and-defaults)
  - [Ray worker CPU budget (`num_cpus`)](#ray-worker-cpu-budget-num_cpus)
  - [Benchmark sweep shape](#benchmark-sweep-shape)
  - [Warmup steps](#warmup-steps)
  - [Auto-reset on episode end](#auto-reset-on-episode-end)
  - [Seed ranges (train vs val)](#seed-ranges-train-vs-val)
  - [gym API compatibility](#gym-api-compatibility)
  - [Memory measurement](#memory-measurement)
- [Translating results to a training config](#translating-results-to-a-training-config)
- [Tuning guide](#tuning-guide)

---

## Why PyBullet works with Ray

PyBullet maintains one physics server **per OS process** (identified
internally by a `physicsClientId`).  Ray actors are separate OS processes, so
each worker gets its own fully isolated physics simulation — no shared state,
no locking.  This is the same reason the existing `MazeWorker` and
`SokobanWorker` actors work; the pattern is identical.

PyBullet workers must always run **headless**.  When `render=False`
(the default for `pybullet_envs` gym environments), PyBullet automatically
uses `pybullet.DIRECT`.  Never pass `render=True` to a Ray worker — it will
try to open a display and fail on a headless server.

---

## Install

```bash
# Core deps (pybullet_envs registers gym env IDs like AntBulletEnv-v0)
pip install pybullet pybullet_envs ray[default]

# For newer pybullet-gymnasium (gym >=0.26 compatible):
pip install pybullet pybullet-gymnasium ray[default]

# Optional — enables the mem_MB column in benchmark output:
pip install psutil
```

The benchmark script and env wrapper do **not** need to be installed as a
package; run them from the repo root.

---

## Running the benchmark

All commands are run from the **repo root** (`/path/to/LaMer`).

### Quick smoke-test

Tests the benchmark machinery itself using `CartPole-v1` (ships with gym,
no pybullet required):

```bash
python -m examples.benchmark_pybullet_envs --env_id CartPole-v1
```

Expected runtime: ~10 seconds.

### Full PyBullet sweep

```bash
python -m examples.benchmark_pybullet_envs \
    --env_id AntBulletEnv-v0 \
    --max_envs 128 \
    --num_steps 200
```

This sweeps `num_envs` over `[1, 2, 4, 8, 16, 32, 64, 128]` and stops early
if efficiency drops below 30 % (the default `--min_efficiency`).

### Custom env count list

Use `--env_counts` to test an arbitrary set of values instead of the default
power-of-2 sweep:

```bash
python -m examples.benchmark_pybullet_envs \
    --env_id HalfCheetahBulletEnv-v0 \
    --env_counts 1 4 16 32 64 128 \
    --num_steps 100 \
    --warmup_steps 20
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--env_id` | `AntBulletEnv-v0` | Gym-registered env ID. Any gym-compatible env works, not just PyBullet ones. |
| `--num_steps` | `100` | Number of **timed** steps per worker per trial point. Higher → more stable measurement; lower → faster sweep. |
| `--warmup_steps` | `10` | Un-timed steps run before measurement starts. Allows JIT compilation, TCP connection setup in Ray, etc. to settle. |
| `--max_envs` | `128` | Upper bound for the power-of-2 sweep. Ignored when `--env_counts` is set. |
| `--env_counts` | *(not set)* | If provided, overrides `--max_envs` and tests exactly these counts (space-separated ints). |
| `--min_efficiency` | `0.30` | The sweep stops and marks a point when efficiency falls below this fraction. `0.30` = 30 %. |
| `--seed` | `0` | Base RNG seed.  Worker `i` is seeded with `seed + i`. |
| `--num_cpus` | *(auto)* | Total CPUs to hand to Ray.  Omit to let Ray auto-detect. Pass an explicit number to simulate a smaller machine or cap resource usage. |

---

## Reading the output

```
PyBullet Env Throughput Benchmark
  Environment  : AntBulletEnv-v0
  Steps/worker : 200  |  Warmup: 10
  Machine      : 32 Ray CPUs, 64 GB RAM
  Sweep        : [1, 2, 4, 8, 16, 32, 64, 128]

  num_envs | steps/sec |    mem_MB | efficiency
--------------------------------------------------
         1 |     142.3 |       180 |   100.0 %
         2 |     278.1 |       360 |    97.7 %
         4 |     540.6 |       720 |    95.1 %
         8 |    1023.4 |      1440 |    90.0 %
        16 |    1854.8 |      2880 |    81.5 %
        32 |    2901.2 |      5760 |    63.7 %
        64 |    3402.7 |     11520 |    37.4 %  <- below 30% efficiency
--------------------------------------------------

  Recommended max_envs : 64  (last point with efficiency >= 30 %)
  Peak throughput      : 3402 steps/sec at 64 envs
```

**`steps/sec`** — Total environment transitions per second across all parallel
workers.  This is the number that matters for training throughput: a higher
value means faster data collection per wall-clock second.

**`mem_MB`** — Sum of RSS (resident set size) for all worker processes.
Requires `psutil`; shows `n/a` if it is not installed.  This is the main
hard limit: once total memory exceeds available RAM the OS will start
swapping and throughput collapses.

**`efficiency`** — `actual_steps_per_sec / (baseline_1env × num_envs)`.
100 % means perfect linear scaling.  It degrades because of:
- **Ray IPC overhead**: each `ray.get` call has ~1–5 ms round-trip latency.
  The more workers, the more time is spent waiting for the slowest one.
- **CPU contention**: if `num_envs × num_cpus_per_worker > total_cpu_cores`,
  workers compete for CPU time and slow each other down.
- **Memory pressure**: once working set approaches RAM capacity, the kernel
  pages out worker memory and stalls on reads.

**`Recommended max_envs`** — the last row where `efficiency >= --min_efficiency`.
This is a conservative starting point; you can push beyond it if throughput
(not efficiency) is your primary concern.

---

## Assumptions and defaults

### Ray worker CPU budget (`num_cpus`)

**`PyBulletWorker` declares `num_cpus=0.5`.**

This appears in `agent_system/environments/pybullet/envs.py:59`:

```python
@ray.remote(num_cpus=0.5)
class PyBulletWorker:
```

And the same value is used in the benchmark's `_BenchWorker` at
`examples/benchmark_pybullet_envs.py:69`.

`num_cpus` is Ray's **scheduling declaration**, not a hard OS limit.  It
tells Ray's scheduler how many CPU slots to reserve on the cluster, which
determines how many actors Ray will allow to run simultaneously given the
available CPU budget.

| Value | Effect |
|-------|--------|
| `0.5` | Ray schedules up to `floor(total_cpus / 0.5)` workers at once, i.e. up to `2 × core_count` workers.  Good default for mixed CPU/Python overhead workloads. |
| `1.0` | One worker per physical core.  Use this if you observe CPU contention with `0.5`. |
| `0.1` | Up to `10 × core_count` workers.  The value used by the Maze/Sokoban wrappers; reasonable for lightweight Python envs but will over-subscribe for physics-heavy PyBullet. |

To change it permanently, edit the decorator in `envs.py`.  To test a
different value without editing, pass `--num_cpus N` to the benchmark to
constrain the total Ray CPU budget and observe how the throughput curve
changes.

### Benchmark sweep shape

By default the sweep is `[1, 2, 4, 8, 16, ..., max_envs]` (powers of 2).
Each point creates fresh workers, runs the benchmark, then **kills all workers
before moving to the next count**.  Workers do not persist between points.
This means:

- Memory from previous points is fully released.
- There is no carry-over state between points.
- Startup time (actor creation + `gym.make`) is included in each point's
  wall-clock time *before* the timed section begins (it happens during
  warmup).

### Warmup steps

Default: **10 steps** (not timed).

These serve several purposes:
- Allows Ray to finish spawning worker processes before the clock starts.
- Triggers any lazy initialisation inside the gym env (e.g. pybullet's
  first `loadURDF` call is slower than subsequent ones).
- Warms up Python's JIT (if using PyPy or Numba inside the env).

If your environment has a slow first episode (e.g. loading large meshes),
increase `--warmup_steps` to cover at least one full episode reset cycle.

### Auto-reset on episode end

In the benchmark's `_BenchWorker.step`, when an episode terminates
(`done=True`) the worker immediately calls `env.reset()` and returns the
new initial observation.  This keeps all workers stepping continuously
without needing external episode management, which would add coordination
overhead and distort the throughput measurement.

The production `PyBulletWorker` in `envs.py` does **not** auto-reset — it
returns `done=True` and expects the caller (the training rollout loop) to
call `reset()` explicitly, matching the behaviour of the other env wrappers
in this codebase.

### Seed ranges (train vs val)

`PyBulletMultiProcessEnv.reset()` draws seeds from different ranges to
prevent train and val environments from sharing initial states:

| Split | Seed range |
|-------|------------|
| `is_train=True` (default) | `[0, 2¹⁶ − 1]` = `[0, 65535]` |
| `is_train=False` | `[2¹⁶, 2³² − 1]` = `[65536, 4294967295]` |

Within a GRPO/GiGPO group (`group_n > 1`), all copies of the same environment
share the same seed so they start from the same state — matching the
grouping semantics of the other env wrappers.

### gym API compatibility

Both `PyBulletWorker` and `_BenchWorker` handle both gym API versions:

| gym version | `env.step()` returns | `env.reset()` returns |
|-------------|---------------------|-----------------------|
| ≤ 0.25 | `(obs, reward, done, info)` | `obs` |
| 0.26+ (current) | `(obs, reward, terminated, truncated, info)` | `(obs, info)` |

`done` is derived as `terminated or truncated` in the 0.26+ case.

### Memory measurement

Memory (`mem_MB`) is measured by calling `psutil.Process(os.getpid()).memory_info().rss`
inside each worker **after** the timed benchmark section completes.  The
values are summed across all workers.

Notes:
- RSS includes shared libraries loaded by pybullet, so the first worker's
  footprint is larger than subsequent ones on Linux (where shared libs are
  mapped once per binary but counted per-process in RSS).
- If `psutil` is not installed the column shows `n/a` and the benchmark
  still runs normally.
- The driver process itself is not counted.

---

## Translating results to a training config

Once you have a `Recommended max_envs` value, set `train_batch_size` in
your training config to that number (or a round multiple of `group_n`):

```yaml
# verl/trainer/config/ppo_trainer.yaml  (or your env-specific config)
data:
  train_batch_size: 32   # set to recommended_max_envs from benchmark

env:
  rollout:
    n: 4                 # group_n — must divide train_batch_size evenly
```

`env_num = train_batch_size` and `total_workers = env_num × group_n`, so the
actual number of Ray actors spawned during training is `32 × 4 = 128` in the
example above.  Run the benchmark with `--env_counts 128` to confirm that
count is sustainable before starting a long training run.

---

## Tuning guide

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Efficiency drops sharply at a small count | CPU oversubscription | Increase `num_cpus` per worker from `0.5` → `1.0` in `envs.py` |
| `mem_MB` grows linearly but efficiency stays high, then crashes | OOM / swap | Reduce `train_batch_size` or add more RAM |
| Throughput plateaus early but efficiency stays high | Ray IPC is the bottleneck, not the envs | Reduce `num_steps` per `ray.get` call or batch multiple steps inside the worker |
| `ERROR` on first worker creation | `pybullet_envs` not installed | `pip install pybullet pybullet_envs` |
| Workers hang on startup | Display required (GUI mode) | Ensure `render=False` (default) or unset `$DISPLAY` |
| `mem_MB` shows `n/a` | `psutil` not installed | `pip install psutil` |
