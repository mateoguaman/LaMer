# Plan: Decouple Training from Environment via Environment Server

## Goal

Create a remote environment server protocol that allows environment managers to
run in a separate process/machine from the training loop. V0 validates
correctness by running Sokoban locally over the network and verifying identical
outcomes to the existing in-process path.

---

## Setup & Running Tests

### Prerequisites

- Python 3.10+
- A working CUDA installation (for `torch`), or CPU-only PyTorch

### 1. Install Dependencies

```bash
# Core project dependencies (from repo root)
pip install -r requirements.txt

# Additional deps needed by the Sokoban environment and tests
pip install torch omegaconf ray[default] gym==0.26.2 gym_sokoban==0.0.6 \
            matplotlib Pillow transformers numpy
```

> **Note on numpy / gym compatibility**: `gym==0.26.2` prints a deprecation
> warning about NumPy 2.0, but works correctly with `numpy<2`. If you have
> `numpy>=2`, either downgrade (`pip install "numpy<2"`) or ignore the warning —
> the tests pass either way.

> **Note on antlr4 / setuptools**: `omegaconf` requires
> `antlr4-python3-runtime==4.9.*`, which fails to build with
> `setuptools>=67`. If you hit this, run:
> ```bash
> pip install "setuptools<67"
> pip install "antlr4-python3-runtime==4.9.3"
> pip install omegaconf
> ```

### 2. Run the Test Suite

All tests are in `scripts/test_remote_env.py`. Each test automatically starts
an environment server in a subprocess and connects to it — no manual server
launch needed.

```bash
# Run all tests
python scripts/test_remote_env.py --all

# Run a single test
python scripts/test_remote_env.py --test correctness
python scripts/test_remote_env.py --test metarl
python scripts/test_remote_env.py --test latency
python scripts/test_remote_env.py --test throughput
python scripts/test_remote_env.py --test large_batch
python scripts/test_remote_env.py --test reconnection
```

### What Each Test Does

| Test | Description | Runtime |
|------|-------------|---------|
| `correctness` | Runs identical reset/step sequences on local & remote envs with the same seed; asserts bit-identical observations, rewards, dones, infos | ~60s |
| `metarl` | Full MetaRL loop: 3 attempts × 7 turns with reflection/restart between attempts; verifies properties, all intermediate obs, and `success_evaluator` | ~60s |
| `latency` | Benchmarks per-operation p50/p95 latency for `reset` and `step` (local vs remote, 50 trials each) | ~60s |
| `throughput` | Runs 20 complete episodes through both local and remote; reports episodes/second | ~30s |
| `large_batch` | Spawns a server with `batch_size=64` and verifies serialization of large observation payloads | ~60s |
| `reconnection` | Starts a server, connects, kills the server, verifies the client raises an error, restarts the server, verifies the client can reconnect | ~90s |

### 3. Launch a Standalone Environment Server (optional)

If you want to run the server and client separately (e.g. on different
machines):

```bash
# Terminal 1: Start env server
python scripts/launch_env_server.py \
    --env_name sokoban \
    --config verl/trainer/config/eval/sokoban.yaml \
    --host 0.0.0.0 \
    --port 50051 \
    --mode val

# Terminal 2: Connect from training code
# In Python:
from agent_system.environments.remote import RemoteEnvironmentManager
env = RemoteEnvironmentManager("localhost:50051")
obs, infos = env.reset()
```

### New Files (this branch)

```
agent_system/environments/remote/
├── __init__.py          # Exports RemoteEnvironmentManager
├── client.py            # RemoteEnvironmentManager (drop-in client)
├── server.py            # EnvServer (wraps any EnvironmentManager)
├── protocol.py          # Shared message types & wire format (TCP + pickle)
└── launcher.py          # CLI launcher utility

scripts/
├── launch_env_server.py # Convenience script to start a server
└── test_remote_env.py   # Test suite & benchmarks (this file)
```

---

## Architecture Overview

```
┌─────────────────────────────────────┐     ┌──────────────────────────────────┐
│         TRAINING PROCESS            │     │      ENVIRONMENT SERVER(S)       │
│                                     │     │                                  │
│  RayPPOTrainer                      │     │  EnvServer (gRPC)                │
│    └─ TrajectoryCollector           │     │    └─ SokobanEnvironmentManager  │
│         └─ RemoteEnvironmentManager │────▶│         └─ SokobanMultiProcessEnv│
│              (drop-in client)       │     │              └─ SokobanWorkers   │
│                                     │     │                                  │
│  ActorRolloutRefWorker (GPU)        │     │  (CPU-only, separate deps)       │
└─────────────────────────────────────┘     └──────────────────────────────────┘
              Network (gRPC)
```

**Key invariant**: `TrajectoryCollector` and `RayPPOTrainer` are NOT modified.
`RemoteEnvironmentManager` is a drop-in replacement that implements the same
interface as `SokobanEnvironmentManager` (or any `EnvironmentManagerBase`).

---

## File Plan

```
agent_system/environments/remote/
├── __init__.py                    # Exports RemoteEnvironmentManager
├── client.py                      # RemoteEnvironmentManager (drop-in client)
├── server.py                      # EnvServer (wraps any EnvironmentManager)
├── protocol.py                    # Shared message types & serialization
└── launcher.py                    # CLI launcher for env server process

scripts/
├── launch_env_server.py           # Convenience script to start server
└── test_remote_env.py             # Correctness & benchmark test suite
```

---

## Step 1: Define the Protocol (`protocol.py`)

Shared message types used by both client and server. Uses Python stdlib only
(socket + pickle + struct) for v0, with a clear abstraction boundary so the
transport can be swapped to gRPC later.

### Interface contract (what crosses the wire)

```python
# Request types (client → server)
@dataclass
class EnvRequest:
    request_id: str           # UUID for matching responses
    method: str               # "reset" | "step" | "restart" | "reflect" |
                              # "success_evaluator" | "get_properties" | "close"
    args: tuple = ()          # Positional args
    kwargs: dict = field(default_factory=dict)  # Keyword args

# Response types (server → client)
@dataclass
class EnvResponse:
    request_id: str
    status: str               # "ok" | "error"
    result: Any = None        # Method return value
    error_message: str = ""   # If status == "error"
```

### Wire format

```
[4 bytes: message length (big-endian uint32)] [N bytes: pickle payload]
```

We use pickle for v0 because all data is already numpy/dict/list — all
natively pickle-serializable. The 4-byte length prefix enables framing over
TCP streams.

### Helper functions

```python
def send_message(sock: socket.socket, data: Any) -> None:
    """Serialize and send with length prefix."""

def recv_message(sock: socket.socket) -> Any:
    """Receive length-prefixed message and deserialize."""
```

### Scalability note

For Slurm/K8s deployment, the protocol layer is the swap point. The
`RemoteEnvironmentManager` constructor takes a `server_address` string. In
v0 this is `"host:port"` for a single TCP connection. For scaled deployments:
- **Slurm**: Each env server is a separate Slurm job; the training job gets
  server addresses via environment variables or a service discovery file.
- **K8s**: Env servers run as a StatefulSet/Deployment with a Service; the
  training pod connects via the Service DNS name. Multiple replicas can serve
  different environment batches behind a load balancer, or each training
  worker connects to its designated env server pod.
- **Multi-server fan-out**: A `ShardedRemoteEnvironmentManager` distributes
  `num_processes` across N servers, each handling a slice of the batch. This
  is a future extension built on the same protocol.

---

## Step 2: Implement the Server (`server.py`)

A thin wrapper that:
1. Accepts a TCP connection from a client
2. Deserializes requests
3. Dispatches to the wrapped `EnvironmentManager` instance
4. Serializes and sends responses

```python
class EnvServer:
    def __init__(self, env_manager, host="0.0.0.0", port=50051):
        self.env = env_manager
        self.host = host
        self.port = port

    def serve(self):
        """Main server loop. Accepts one client (v0), handles requests."""

    def _handle_request(self, request: EnvRequest) -> EnvResponse:
        """Dispatch request to env_manager method."""
        if request.method == "get_properties":
            return EnvResponse(
                request_id=request.request_id,
                status="ok",
                result={
                    "num_attempts": self.env.num_attempts,
                    "num_processes": self.env.num_processes,
                    "max_turns": self.env.max_turns,
                    "do_reflection": self.env.do_reflection,
                }
            )
        method = getattr(self.env, request.method)
        result = method(*request.args, **request.kwargs)
        return EnvResponse(request_id=..., status="ok", result=result)
```

**Methods exposed**: `reset`, `step`, `restart`, `reflect`,
`success_evaluator`, `close`, `get_properties`.

**Error handling**: Exceptions in env methods are caught, serialized as
`EnvResponse(status="error", error_message=traceback.format_exc())`, and
sent back to the client. The client re-raises with context.

**Threading model (v0)**: Single-threaded, synchronous. The training loop is
synchronous (step → wait → generate → wait → step), so there's no benefit
to async on the server side. This also avoids any concurrency issues with
stateful env managers.

**Scalability path**: For multiple concurrent training runs or multi-server
fan-out, add a threading mode where each connection gets its own thread and
its own env manager instance. Or run multiple server processes (one per env
shard) and have the client connect to each.

---

## Step 3: Implement the Client (`client.py`)

Drop-in replacement for any `EnvironmentManagerBase` subclass.

```python
class RemoteEnvironmentManager:
    """
    Network client that implements the same interface as
    EnvironmentManagerBase subclasses. Can be passed directly to
    TrajectoryCollector.multi_turn_loop() and RayPPOTrainer.
    """

    def __init__(self, server_address: str, timeout: float = 300.0):
        """
        Connect to an environment server.
        server_address: "host:port" string
        timeout: Socket timeout in seconds (default 5 min for slow envs)
        """
        # Parse host:port, establish TCP connection
        # Call get_properties to populate num_attempts, num_processes, etc.

    # -- Properties read by TrajectoryCollector --
    @property
    def num_attempts(self) -> int: ...
    @property
    def max_turns(self) -> int: ...
    @property
    def do_reflection(self) -> bool: ...
    @property
    def num_processes(self) -> int: ...

    # -- Methods called by TrajectoryCollector --
    def reset(self):
        """Returns (observations_dict, infos_list)"""
        return self._call("reset")

    def step(self, text_actions, phase="play"):
        """Returns (next_obs_dict, rewards, dones, infos)"""
        return self._call("step", text_actions, phase=phase)

    def restart(self):
        """Returns (observations_dict, infos_list)"""
        return self._call("restart")

    def reflect(self):
        """Returns (observations_dict, infos_list)"""
        return self._call("reflect")

    def success_evaluator(self, **kwargs):
        """Returns Dict[str, np.ndarray]"""
        return self._call("success_evaluator", **kwargs)

    def close(self):
        """Gracefully shut down connection."""
        self._call("close")
        self._socket.close()

    # -- Internal --
    def _call(self, method, *args, **kwargs):
        """Send request, receive response, handle errors."""
        request = EnvRequest(request_id=uuid4().hex, method=method,
                             args=args, kwargs=kwargs)
        send_message(self._socket, request)
        response = recv_message(self._socket)
        if response.status == "error":
            raise RuntimeError(f"Remote env error in {method}: "
                               f"{response.error_message}")
        return response.result

    def __del__(self):
        try:
            self.close()
        except:
            pass
```

### Connection resilience (for Slurm/K8s)

- **Reconnection**: If `send_message` or `recv_message` raises a connection
  error, retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s).
  Between retries, re-establish the TCP connection.
- **Health check**: Optional `ping` method (server returns immediately) to
  verify connectivity before starting a training epoch.

---

## Step 4: Server Launcher (`launcher.py` and `scripts/launch_env_server.py`)

### `launcher.py` (importable)

```python
def launch_env_server(env_name: str, config_path: str, host: str, port: int,
                      is_train: bool = True, config_overrides: dict = None):
    """
    Create an environment manager from config and start serving.
    - Initializes Ray (for env workers)
    - Creates env manager via make_envs()
    - Wraps in EnvServer and calls serve()
    """
```

### `scripts/launch_env_server.py` (CLI)

```
python scripts/launch_env_server.py \
    --env_name sokoban \
    --config verl/trainer/config/eval/sokoban.yaml \
    --host 0.0.0.0 \
    --port 50051 \
    --mode train          # or "val"
    --batch_size 4        # override config batch size
```

Parses args, calls `launch_env_server()`. For Slurm, this becomes the
command in the Slurm job script. For K8s, this becomes the container
entrypoint.

---

## Step 5: Integration Point in `main_ppo.py`

Minimal change — add an `if` branch before the existing `make_envs` call:

```python
# In main_ppo.py, run_ppo():
if config.env.get('remote', False):
    from agent_system.environments.remote import RemoteEnvironmentManager
    envs = RemoteEnvironmentManager(config.env.remote_address)
    val_envs = RemoteEnvironmentManager(config.env.remote_val_address)
else:
    # existing local env creation (unchanged)
    envs, val_envs = make_envs(config)
```

Config additions (in YAML):
```yaml
env:
  remote: true
  remote_address: "localhost:50051"
  remote_val_address: "localhost:50052"
```

---

## Step 6: Testing & Benchmarking (`scripts/test_remote_env.py`)

### Test 1: Correctness — Deterministic Comparison

Run the SAME sequence of actions through both local and remote env managers
with the same seed. Verify that observations, rewards, dones, and infos are
**bit-identical**.

```python
def test_correctness():
    """
    1. Create a local SokobanEnvironmentManager with seed=42
    2. Start an EnvServer wrapping an identical SokobanEnvironmentManager
       with seed=42 in a subprocess
    3. Create a RemoteEnvironmentManager pointing to the server
    4. Run the same sequence of reset/step/reflect/restart calls on both
    5. Assert numpy.array_equal on all observations, rewards, dones
    6. Assert info dicts are identical
    7. Assert success_evaluator returns identical results
    """
```

Key design: Use a **fixed action sequence** (not random) so the comparison
is deterministic regardless of env randomness (the seeds make env init
deterministic; the actions are pre-scripted).

### Test 2: Multi-Attempt MetaRL Correctness

Same as Test 1 but exercises the full MetaRL loop:
- 3 attempts with reflection between each
- 7 turns per attempt
- Verify all intermediate observations match

### Test 3: Latency Benchmark

```python
def test_latency():
    """
    Measure round-trip time for each operation:
    - reset()
    - step() with various batch sizes (1, 4, 16, 64)
    - reflect()
    - restart()

    Report:
    - Mean, p50, p95, p99 latency per operation
    - Overhead vs local call (remote_time - local_time)
    - Throughput: steps/second
    """
```

### Test 4: Sustained Throughput Benchmark

```python
def test_throughput():
    """
    Run N complete episodes (reset → 7 steps → reflect → restart → 7 steps ...)
    through both local and remote, measure total wall-clock time.

    Report:
    - Episodes/second (local vs remote)
    - Overhead percentage
    """
```

### Test 5: Connection Resilience

```python
def test_reconnection():
    """
    1. Start server
    2. Connect client, do a few steps
    3. Kill server process
    4. Verify client raises appropriate error
    5. Restart server
    6. Verify client reconnects and can resume (new episode)
    """
```

### Test 6: Large Batch Serialization

```python
def test_large_batch():
    """
    Test with batch_size=128 (matching val config).
    Verify serialization handles large observation payloads.
    Measure serialization/deserialization time separately.
    """
```

### Test runner CLI

```
# Run all tests
python scripts/test_remote_env.py --all

# Run specific test
python scripts/test_remote_env.py --test correctness

# Run benchmark only
python scripts/test_remote_env.py --test latency --batch_sizes 1,4,16,64
```

---

## Step 7: Scalability Considerations

### Slurm Deployment

```bash
#!/bin/bash
#SBATCH --job-name=env-server
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:0

# Write our hostname:port to a shared file for the training job
echo "$(hostname):50051" > /shared/env_server_address.txt

python scripts/launch_env_server.py \
    --env_name sokoban --config config.yaml \
    --host 0.0.0.0 --port 50051 --mode train
```

Training job reads the address:
```bash
#SBATCH --gres=gpu:4
ENV_SERVER=$(cat /shared/env_server_address.txt)
python -m verl.trainer.main_ppo ... env.remote=true \
    env.remote_address=$ENV_SERVER
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: env-server
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: env-server
        image: lamer-env-server:latest
        command: ["python", "scripts/launch_env_server.py",
                  "--env_name", "sokoban", "--port", "50051"]
        ports:
        - containerPort: 50051
        resources:
          requests:
            cpu: "8"
            memory: "16Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: env-server
spec:
  selector:
    app: env-server
  ports:
  - port: 50051
```

Training pod connects to `env-server:50051`.

### Multi-Server Fan-Out (Future Extension)

For very large batch sizes or expensive environments, split the batch across
multiple servers:

```python
class ShardedRemoteEnvironmentManager:
    """
    Distributes num_processes across N env servers.
    Server 0 handles envs [0, K), Server 1 handles [K, 2K), etc.
    Aggregates responses into single batch before returning.
    """
```

This is NOT in v0 scope but the protocol supports it — each server is
stateless from the client's perspective (it manages its own env indices).

---

## Implementation Order

1. **`protocol.py`** — Message types and wire format helpers (~80 lines)
2. **`server.py`** — EnvServer class (~120 lines)
3. **`client.py`** — RemoteEnvironmentManager class (~150 lines)
4. **`launcher.py`** — Server launcher utility (~60 lines)
5. **`__init__.py`** — Exports (~5 lines)
6. **`scripts/launch_env_server.py`** — CLI entry point (~40 lines)
7. **`scripts/test_remote_env.py`** — Test suite & benchmarks (~400 lines)
8. **`main_ppo.py`** — Add remote env config branch (~10 lines)

**Total new code**: ~865 lines
**Modified existing code**: ~10 lines in `main_ppo.py`
**Changes to training loop / rollout loop**: 0

---

## What Does NOT Change

| Component | Modified? | Why |
|-----------|-----------|-----|
| `metarl_rollout_loop.py` | No | Uses only the env interface methods |
| `rollout_loop.py` | No | Same reason |
| `ray_trainer.py` | No | Stores envs as opaque objects |
| `SokobanEnvironmentManager` | No | Becomes server-side implementation |
| `SokobanMultiProcessEnv` | No | Lives inside the server |
| Projection functions | No | Called inside env manager (server-side) |
| Memory classes | No | Internal to env manager (server-side) |
| Reward manager | No | Works on trajectory data, not envs |
| Other env types (maze, etc.) | No | Can be wrapped by the same server later |

---

## Success Criteria for V0

1. `test_correctness` passes: remote Sokoban produces **identical** observations,
   rewards, dones, and success metrics as local Sokoban for the same seed and
   action sequence
2. `test_latency` shows <5ms overhead per step on localhost
3. `test_large_batch` works with batch_size=128
4. Server and client can be started/stopped independently
5. No changes to any existing training or rollout code

---

# V1: Sharded Remote Environments + VLA Inner Policy Support

Everything below documents features built on top of the V0 remote environment
system.  The V0 docs above remain accurate — sharded mode and VLA support are
additive.

---

## Sharded Remote Environments

### Overview

`ShardedRemoteEnvironmentManager` fans out a single logical batch of
environments across **N independent `EnvServer` processes** (shards), each
running on a potentially different machine.  From the training loop's
perspective it is a drop-in replacement for `RemoteEnvironmentManager` — same
interface, same properties.

```
┌──────────────────────────────────────────────────────────┐
│                    TRAINING PROCESS                       │
│                                                          │
│  RayPPOTrainer                                           │
│    └─ TrajectoryCollector                                │
│         └─ ShardedRemoteEnvironmentManager               │
│              ├─ RemoteEnvironmentManager → EnvServer 0   │──▶ envs [0, K)
│              ├─ RemoteEnvironmentManager → EnvServer 1   │──▶ envs [K, 2K)
│              └─ RemoteEnvironmentManager → EnvServer N-1 │──▶ envs [(N-1)K, N·K)
│              ThreadPoolExecutor(N workers)                │
└──────────────────────────────────────────────────────────┘
```

### How it works

1. **Constructor** connects to each server address, queries `get_properties`
   from every shard, validates that scalar properties (`num_attempts`,
   `max_turns`, `do_reflection`) agree across all shards, and pre-computes
   slice boundaries from each shard's `num_processes`.

2. **`reset()` / `restart()` / `reflect()`** — dispatched in parallel via a
   `ThreadPoolExecutor` (one thread per shard, I/O-bound socket waits).
   Results are collected in shard order and merged:
   - Observation dicts: per-key concatenation (lists are extended, ndarrays
     are `np.concatenate`d).
   - Info lists: flat concatenation.

3. **`step(text_actions, phase)`** — the flat action list is sliced at shard
   boundaries before dispatch.  Each shard receives only its slice.  Results
   (obs, rewards, dones, infos) are merged the same way as reset.

4. **`success_evaluator(**kwargs)`** — `total_infos` and `total_batch_list`
   are sliced per shard.  Per-shard result dicts are merged by concatenating
   numpy arrays per key.

5. **`close()`** — closes all shard connections and shuts down the thread
   pool.

### Error semantics

If any shard raises an exception during a parallel call, the exception
propagates out of the `ThreadPoolExecutor` future and surfaces to the
caller.  There is no automatic retry at the sharded level — the
individual `RemoteEnvironmentManager` handles reconnection with
exponential backoff (up to 4 retries).

### Configuration

```yaml
# In your training YAML:
env:
  remote: true
  sharded: true
  remote_addresses:
    - "host0:50051"
    - "host1:50051"
  remote_val_addresses:
    - "host0:50052"
    - "host1:50052"
```

The config branch in `main_ppo.py` (lines 60–68):

```python
if config.env.get('remote', False):
    if config.env.get('sharded', False):
        from agent_system.environments.remote import ShardedRemoteEnvironmentManager
        envs = ShardedRemoteEnvironmentManager(list(config.env.remote_addresses))
        val_envs = ShardedRemoteEnvironmentManager(list(config.env.remote_val_addresses))
    else:
        from agent_system.environments.remote import RemoteEnvironmentManager
        envs = RemoteEnvironmentManager(config.env.remote_address)
        val_envs = RemoteEnvironmentManager(config.env.remote_val_address)
```

### Resource Allocation

The sharded environment code does **not** allocate hardware resources.  There is
no GPU pinning, no CPU affinity setting, and no memory-limit enforcement inside
`EnvServer`, `RemoteEnvironmentManager`, or `ShardedRemoteEnvironmentManager`.
Resource isolation is entirely the **deployment layer's** responsibility.

| Scenario | What you must do |
|----------|-----------------|
| **GPU-based envs (PyBullet VLA)** | Set `CUDA_VISIBLE_DEVICES` per server process so each sees only its assigned GPU(s). The VLA `device` string (e.g. `"cuda:0"`) is always relative to the visible set. |
| **CPU-only envs (Sokoban)** | No special handling needed — each server spawns its own Sokoban instances in-process. |
| **Slurm cluster** | Use `--gres=gpu:1` (or more) per `srun` / `sbatch` task. Slurm sets `CUDA_VISIBLE_DEVICES` automatically. |
| **Kubernetes** | Use `resources.limits` with `nvidia.com/gpu` in your pod spec. The device plugin handles visibility. |

What the code *does* control (but only locally):

- **Ray actor `num_cpus` hints** — used when Ray actors are involved (local
  mode). These are soft hints, not hard limits.
- **VLA `device` string** — selects which (visible) CUDA device the model loads
  onto. Defaults to `"cuda:0"`.
- **Ray CPU cap** — `ray.init(num_cpus=N)` caps the local Ray scheduler, useful
  for preventing oversubscription on shared machines.

There is **no orchestrator** — you manually start N `EnvServer` processes
(one per shard), then point the `ShardedRemoteEnvironmentManager` at their
addresses.  The training process never SSH-es into remote machines or spawns
servers on your behalf.

### Running Sharded Locally

#### CPU-only example (Sokoban, 2 shards)

```bash
# Terminal 1 — shard 0
python -m agent_system.environments.remote.server \
    --config configs/sokoban.yaml --port 50051

# Terminal 2 — shard 1
python -m agent_system.environments.remote.server \
    --config configs/sokoban.yaml --port 50052

# Terminal 3 — training
python verl/trainer/main_ppo.py \
    env.remote=true \
    env.sharded=true \
    env.remote_addresses='["localhost:50051","localhost:50052"]' \
    env.remote_val_addresses='["localhost:50053","localhost:50054"]'
```

#### GPU example (VLA, 2 GPUs)

```bash
# Terminal 1 — shard 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m agent_system.environments.remote.server \
    --config configs/pybullet_vla.yaml --port 50051

# Terminal 2 — shard 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m agent_system.environments.remote.server \
    --config configs/pybullet_vla.yaml --port 50052

# Terminal 3 — training (same as above)
python verl/trainer/main_ppo.py \
    env.remote=true \
    env.sharded=true \
    env.remote_addresses='["localhost:50051","localhost:50052"]' \
    env.remote_val_addresses='["localhost:50053","localhost:50054"]'
```

Each server process sees only its assigned GPU because of
`CUDA_VISIBLE_DEVICES`.  The VLA model inside the server loads onto `cuda:0`,
which maps to the physical GPU you exposed.

---

## PyBullet VLA Environment Manager

### Two-loop architecture

The PyBullet VLA setup introduces a **two-loop architecture**:

```
┌─────────────────── OUTER LOOP (LLM) ────────────────────┐
│  LLM generates goal strings (natural language)           │
│    ↓                                                     │
│  PyBulletVLAEnvironmentManager.step(goal_strings)        │
│    ↓                                                     │
│  ┌────────── INNER LOOP (VLA) ──────────────────┐        │
│  │  for inner_step in range(max_inner_steps):   │        │
│  │    vla_actions = vla.predict(goals, obs, mask)│        │
│  │    obs, rew, done, info = pybullet.step(act)  │        │
│  │    accumulate rewards, update active_mask     │        │
│  └──────────────────────────────────────────────┘        │
│    ↓                                                     │
│  state_to_text(final_pybullet_state) → text obs for LLM │
└──────────────────────────────────────────────────────────┘
```

- **Outer loop**: The LLM sees text descriptions of robot/scene state and
  produces natural-language goal strings.
- **Inner loop**: A frozen VLA policy executes in the pybullet simulator for
  up to `max_inner_steps`, translating the LLM's goal into low-level actions.
- The LLM never sees raw images or joint angles — `state_to_text` converts
  pybullet state dicts into readable descriptions.

### File reference

| File | Purpose |
|------|---------|
| `agent_system/environments/pybullet_vla/__init__.py` | Exports `PyBulletVLAEnvironmentManager`, `make_envs`, `VLAPolicy`, `DummyVLAPolicy`, `state_to_text`, `batch_state_to_text` |
| `agent_system/environments/pybullet_vla/env_manager.py` | `PyBulletVLAEnvironmentManager` class + `make_envs` factory |
| `agent_system/environments/pybullet_vla/vla_policy.py` | `VLAPolicy` ABC + `DummyVLAPolicy` placeholder |
| `agent_system/environments/pybullet_vla/state_to_text.py` | `state_to_text` / `batch_state_to_text` — pybullet dict → text |

### `PyBulletVLAEnvironmentManager`

Extends `EnvironmentManagerBase`.  Constructor takes:
- `envs` — vectorised pybullet env (gym-like with `reset()`, `step()`,
  `restart()`, `close()`, `num_processes`)
- `vla_policy` — a `VLAPolicy` instance (frozen)
- `config` — OmegaConf config

Key config keys under `config.env`:
- `num_attempts` (default 1) — meta-RL attempts
- `max_turns` (default 1) — outer LLM turns per attempt
- `do_reflection` (default False) — enable reflect/restart cycle
- `max_inner_steps` (default 100) — VLA inner-loop budget
- `reward_shaping` (default `"sum"`) — how inner rewards are aggregated

**Inner loop details** (`_handle_play_step`):
1. Gets current pybullet observations (via `envs.get_obs()` if available).
2. Calls `vla.predict(goal_strings, observations, active_mask)` each inner
   step.
3. Steps pybullet envs with VLA actions.
4. Accumulates rewards for active envs, updates `active_mask` as envs finish.
5. Early-exits when all envs are done; marks timed-out envs as done.
6. Converts final pybullet state to text via `batch_state_to_text`.

### `state_to_text` expected dict format

Each pybullet state dict should contain:

```python
{
    "ee_position": (x, y, z),           # end-effector position
    "gripper_state": float,              # 0=closed, 1=open
    "joint_positions": [j0, j1, ...],    # joint angles
    "object_poses": {                    # scene objects
        "cube": ((x,y,z), (qx,qy,qz,qw)),
        "target": ((x,y,z), (qx,qy,qz,qw)),
    },
    "task_info": "optional task description",  # optional
}
```

Additional keys are silently ignored (forward-compatible).

### `VLAPolicy` ABC

Abstract base class with two methods to override:

```python
class VLAPolicy(ABC):
    def _load_model(self) -> None: ...      # Load checkpoint into self.model
    def _forward(self, goals, observations, active_mask) -> np.ndarray: ...
```

`predict()` is the public entry point — it delegates to `_forward` and
handles the zero-active-envs case.

### `DummyVLAPolicy`

Returns `np.random.uniform(-1, 1, (batch, action_dim))`.  Used when no
checkpoint is provided (testing mode).  `action_dim` defaults to 7
(typical robot arm + gripper).

### Implementing a real VLA

1. Subclass `VLAPolicy`.
2. Override `_load_model()` to load your checkpoint (RT-2, Octo, OpenVLA,
   etc.).
3. Override `_forward()` to run inference.  Receives goal strings,
   per-env observation dicts, and an active mask.  Return actions as a
   numpy array of shape `(batch, action_dim)`.
4. Either register in the `make_envs` factory or pass directly to
   `PyBulletVLAEnvironmentManager`.

### `make_envs` factory contract

```python
def make_envs(config) -> tuple[PyBulletVLAEnvironmentManager, PyBulletVLAEnvironmentManager]:
```

Expected config keys under `config.env.pybullet_vla`:
- `vla_checkpoint` — path to VLA checkpoint (empty string → `DummyVLAPolicy`)
- `vla_device` — torch device (default `"cuda"`)
- `vla_batch_size` — max VLA batch size (default 128)
- `action_dim` — action dimensions (default 7, used by `DummyVLAPolicy`)
- `env_factory` — dotted import path to a `build_envs(num_envs, is_train, **kwargs)` function
- `env_kwargs` — dict passed through to `env_factory`

The `pybullet_vla` env type is registered in `launcher.py`'s `_ENV_FACTORIES`
dict, so it can be served remotely:

```bash
python scripts/launch_env_server.py \
    --env_name pybullet_vla \
    --config your_config.yaml \
    --host 0.0.0.0 --port 50051
```

---

## Configuration Reference

### Single-server remote (V0)

```yaml
env:
  remote: true
  remote_address: "host:50051"
  remote_val_address: "host:50052"
```

### Sharded remote (V1)

```yaml
env:
  remote: true
  sharded: true
  remote_addresses:
    - "host0:50051"
    - "host1:50051"
  remote_val_addresses:
    - "host0:50052"
    - "host1:50052"
```

### PyBullet VLA

```yaml
env:
  env_name: pybullet_vla
  num_attempts: 3
  max_turns: 1
  do_reflection: true
  max_inner_steps: 100
  reward_shaping: sum
  pybullet_vla:
    vla_checkpoint: ""          # empty → DummyVLAPolicy
    vla_device: "cuda"
    vla_batch_size: 128
    action_dim: 7
    env_factory: "my_module.build_pybullet_envs"
    env_kwargs:
      task: "pick_and_place"
      render: false
```

---

## Deployment: Slurm (Sharded)

### Job array pattern

Each environment shard runs as a separate Slurm job.  Each job writes its
`hostname:port` to a shared directory that the training job reads.

```bash
#!/bin/bash
#SBATCH --job-name=env-shard
#SBATCH --array=0-3              # 4 shards
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:0

PORT=$((50051 + SLURM_ARRAY_TASK_ID))
ADDR_DIR=/shared/env_server_addresses
mkdir -p $ADDR_DIR

echo "$(hostname):${PORT}" > "${ADDR_DIR}/shard_${SLURM_ARRAY_TASK_ID}.txt"

python scripts/launch_env_server.py \
    --env_name sokoban \
    --config config.yaml \
    --host 0.0.0.0 \
    --port $PORT \
    --mode train
```

### Training job reads addresses

```bash
#!/bin/bash
#SBATCH --job-name=lamer-train
#SBATCH --gres=gpu:4
#SBATCH --dependency=afterok:<env_shard_job_id>

ADDR_DIR=/shared/env_server_addresses

# Collect all shard addresses into a comma-separated list.
ADDRESSES=""
for f in ${ADDR_DIR}/shard_*.txt; do
    addr=$(cat "$f")
    ADDRESSES="${ADDRESSES:+$ADDRESSES,}$addr"
done

python -m verl.trainer.main_ppo \
    env.remote=true \
    env.sharded=true \
    "env.remote_addresses=[$ADDRESSES]"
```

---

## Deployment: Kubernetes (Sharded)

### StatefulSet for env servers

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: env-server
spec:
  serviceName: env-server
  replicas: 4
  selector:
    matchLabels:
      app: env-server
  template:
    metadata:
      labels:
        app: env-server
    spec:
      containers:
      - name: env-server
        image: lamer-env-server:latest
        command:
          - python
          - scripts/launch_env_server.py
          - --env_name=sokoban
          - --config=/config/train.yaml
          - --host=0.0.0.0
          - --port=50051
        ports:
        - containerPort: 50051
        resources:
          requests:
            cpu: "8"
            memory: "16Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: env-server
spec:
  clusterIP: None   # headless — each pod gets its own DNS
  selector:
    app: env-server
  ports:
  - port: 50051
```

### ConfigMap for training pod

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: env-addresses
data:
  addresses: |
    env-server-0.env-server:50051
    env-server-1.env-server:50051
    env-server-2.env-server:50051
    env-server-3.env-server:50051
```

The training pod reads the ConfigMap and connects:

```yaml
env:
  remote: true
  sharded: true
  remote_addresses:
    - "env-server-0.env-server:50051"
    - "env-server-1.env-server:50051"
    - "env-server-2.env-server:50051"
    - "env-server-3.env-server:50051"
```

---

## Sharded Test Suite (`scripts/test_sharded_env.py`)

### Running tests

```bash
# Run all tests
python scripts/test_sharded_env.py --all

# Run a single test
python scripts/test_sharded_env.py --test correctness
python scripts/test_sharded_env.py --test properties
python scripts/test_sharded_env.py --test step_slicing
python scripts/test_sharded_env.py --test shard_failure
python scripts/test_sharded_env.py --test latency
```

All tests automatically start `EnvServer` subprocesses and connect to them —
no manual server launch needed.  Uses `multiprocessing.set_start_method("spawn")`
for clean subprocess isolation.

### How the tests work internally

Tests use Python's `multiprocessing` module to spawn real `EnvServer` processes
and connect to them over TCP, exercising the full client–server path.

**`_server_worker` (line 75)** — A top-level module function (not a lambda or
closure) that creates a Sokoban environment and starts an `EnvServer`.  It must
be a top-level function because `multiprocessing.Process` with
`start_method="spawn"` needs to pickle the target callable.  The OmegaConf
config is serialized to YAML *before* being passed to the subprocess (raw
OmegaConf objects aren't reliably picklable across spawn boundaries).

**`_start_server_process` (line 91)** — Spawns the subprocess via
`multiprocessing.Process(target=_server_worker, ...)`, then polls readiness by
attempting `socket.connect()` every 2 seconds for up to 60 seconds (30 attempts
× 2s).  If the process dies during startup, it raises immediately rather than
waiting for the timeout.

**`test_correctness` (line 195)** — The most comprehensive test.  Starts **4
servers** on consecutive ports: 2 for the sharded client, and 2 as independent
"reference" clients.  All 4 use the same config (same seed), so given the same
actions the reference clients produce identical output to the sharded pair.  The
test runs `reset()` + 5 `step()` calls through both paths and asserts that the
sharded output exactly equals the manual concatenation of the two individual
reference outputs (`np.concatenate` for arrays, list concatenation for infos).

**CPU-only** — All tests use Sokoban (CPU-only).  They validate the fan-out,
slicing, and merge mechanics of `ShardedRemoteEnvironmentManager`, not GPU
resource allocation.  GPU-based environments (VLA) are tested separately.

**`set_start_method("spawn")`** — Called at test startup to avoid fork-related
issues with Ray.  Forking a process that has already initialized Ray can lead to
deadlocks or corrupted state; `spawn` starts a fresh Python interpreter for each
subprocess.

### What each test does

| Test | Description | Setup |
|------|-------------|-------|
| `correctness` | Starts 4 servers (2 for sharded client, 2 as individual references). Runs reset + 5 steps through both. Asserts sharded output equals manual concatenation of individual client outputs. | 2 envs/shard, 4 total |
| `properties` | Connects a sharded client to 2 servers (3 envs each). Verifies `num_processes=6`, and that `num_attempts`, `max_turns`, `do_reflection` match across shards. | 3 envs/shard |
| `step_slicing` | Verifies that actions `[0:N]` go to shard 0 and `[N:2N]` go to shard 1 by checking output shapes match the total batch size. | 2 envs/shard |
| `shard_failure` | Connects to 2 shards, kills one server, verifies the next `step()` raises `ConnectionError` / `RuntimeError` / `OSError`. | 2 envs/shard |
| `latency` | Benchmarks 1×8 (single server) vs 2×4 (sharded) over 20 trials. Reports p50/p95 latency for reset and step. | 8 total envs |

---

## New Files Summary (this branch)

```
agent_system/environments/remote/
├── __init__.py              # Exports RemoteEnvironmentManager, EnvServer,
│                            #   ShardedRemoteEnvironmentManager
├── client.py                # RemoteEnvironmentManager (drop-in client)
├── server.py                # EnvServer (wraps any EnvironmentManager)
├── protocol.py              # Shared message types & wire format (TCP + pickle)
├── launcher.py              # CLI launcher — _ENV_FACTORIES includes pybullet_vla
└── sharded_client.py        # ShardedRemoteEnvironmentManager (fan-out to N servers)

agent_system/environments/pybullet_vla/
├── __init__.py              # Exports PyBulletVLAEnvironmentManager, make_envs,
│                            #   VLAPolicy, DummyVLAPolicy, state_to_text,
│                            #   batch_state_to_text
├── env_manager.py           # PyBulletVLAEnvironmentManager + make_envs factory
├── vla_policy.py            # VLAPolicy ABC + DummyVLAPolicy (random actions)
└── state_to_text.py         # Pybullet state dict → natural language text

scripts/
├── launch_env_server.py     # CLI entry point for starting env servers
├── test_remote_env.py       # V0 single-server test suite
└── test_sharded_env.py      # V1 sharded test suite (5 tests)
```
