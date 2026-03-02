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
