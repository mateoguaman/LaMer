Below is the instruction to **setup** and **test** the environment for 🌊LaMer:

### Install veRL

The first step is to install veRL:

```
conda create -n lamer python==3.12 -y
conda activate lamer

pip3 install vllm==0.8.5

conda install -c nvidia cuda-nvcc=12.4 cuda-toolkit=12.4

pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -r requirements.txt
```

### Environment

For Sokoban and MineSweeper, no extra packages are needed. For Webshop and ALFWorld, please follow the instructions from [verl-agent](https://github.com/langfengQ/verl-agent/tree/master?tab=readme-ov-file#install-supported-environments) which are copied below:

#### ALFWorld

Install with pip:

```
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
```

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in ~/.cache/alfworld/):

```
alfworld-download -f
Use --extra to download pre-trained checkpoints and seq2seq data.
```

Play a Textworld game:

```
alfworld-play-tw
```

#### Webshop

Webshop requires Python <=3.10, so begin by creating a new verl-agent-webshop environment

```
conda create -n webshop python==3.10 -y
conda activate webshop
```

Install Webshop

```
cd ./agent_system/environments/webshop/webshop
./setup.sh -d all
```

Note: If you encounter issues with gdown, you may need to visit [https://drive.google.com/](https://drive.google.com/), get your Google Drive cookie, and paste it into .cache/gdown/cookies.txt. Or you may need to manually download the files.

After Webshop is installed, return to the root directory of the repository and install the verl package in verl-agent:

```
cd repo_root/
pip3 install vllm==0.8.5

pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -r requirements.txt
```

Please also refer to the original [Webshop](https://github.com/princeton-nlp/WebShop?tab=readme-ov-file#-setup) repository for more detailed information.

#### Language Table (PyBullet tabletop manipulation)

Language Table uses `gym==0.23` and PyBullet, which are incompatible with the
LaMer venv. It runs in a **separate process** via the remote env server protocol.

**Step 1: Create local config files**

From the LaMer repo:

```bash
cp .env.language_table.example .env.language_table
cp .env.language_table.secrets.example .env.language_table.secrets
```

Edit `.env.language_table` for non-secret cluster-specific settings such as:

- `LANGTABLE_DIR`
- `LANGTABLE_CONDA_ENV` (env name such as `ltvenv`; use plain names so `conda create -n` honors `envs_dirs`)
- `LAMER_CONDA_ENV`
- `CONDA_PKGS_DIRS`
- `PIP_CACHE_DIR`
- `TMPDIR`
- `CHECKPOINT_ROOT`
- `RUN_NAME`
- `VLA_CHECKPOINT_DIR`

Edit `.env.language_table.secrets` for secrets such as:

- `WANDB_USERNAME`
- `WANDB_API_KEY`
- `HF_TOKEN`

**Step 2: Bootstrap both environments with conda**

This creates or updates:

- the named LaMer conda env (`LAMER_CONDA_ENV`, default `lamer`)
- the Language Table env at `LANGTABLE_CONDA_ENV`

```bash
scripts/bootstrap_language_table.sh
```

If your cluster has a tight home quota, set `CONDA_PKGS_DIRS`,
`PIP_CACHE_DIR`, and `TMPDIR` to scratch or scrubbed storage. The bootstrap
script exports those paths before installing anything.

The Slurm job uses the original LaMer text parquet locations:

- `~/data/verl-agent/text/train.parquet`
- `~/data/verl-agent/text/test.parquet`

Checkpoints and trainer logs are written under:

- `${CHECKPOINT_ROOT}/${RUN_NAME}`

**Step 3: Verify the language-table environment works standalone**

```bash
cd /path/to/language-table
export PYTHONPATH=${PWD}:$PYTHONPATH
ltvenv/bin/python -m language_table.lamer.test_standalone --num_envs 4 --num_steps 50
```

This runs the standalone tests and saves renders to `${LT_RENDER_DIR:-/tmp/lt_renders}`.

**Step 4: Start the remote env servers**

Open two terminals (or use tmux/screen) in the language-table repo:

```bash
# Terminal 1: Training server
cd /path/to/language-table
export PYTHONPATH=${PWD}:$PYTHONPATH
ltvenv/bin/python -m language_table.lamer.server_main \
    --host 0.0.0.0 --port 50051 \
    --num_envs 8 --block_mode BLOCK_4 \
    --max_inner_steps 100 --num_attempts 3

# Terminal 2: Validation server
cd /path/to/language-table
export PYTHONPATH=${PWD}:$PYTHONPATH
ltvenv/bin/python -m language_table.lamer.server_main \
    --host 0.0.0.0 --port 50052 \
    --num_envs 16 --block_mode BLOCK_4 \
    --max_inner_steps 100 --num_attempts 3
```

**Step 5: Run LaMer training** (in a third terminal, in the LaMer repo)

```bash
cd /path/to/LaMer
conda activate lamer

python3 -m verl.trainer.main_ppo \
    env.env_name=language_table \
    env.remote=True \
    env.remote_address=localhost:50051 \
    env.remote_val_address=localhost:50052 \
    env.num_attempts=3 \
    env.max_turns=1 \
    ... # other PPO/model config as usual
```

The LaMer-side code (`agent_system/environments/language_table/`) is a thin
wrapper that returns `RemoteEnvironmentManager` instances pointing at the
language-table servers. No language-table dependencies are needed in the LaMer
venv.

**Server CLI options:**


| Flag                | Default   | Description                                           |
| ------------------- | --------- | ----------------------------------------------------- |
| `--num_envs`        | 8         | Number of parallel PyBullet environments              |
| `--block_mode`      | `BLOCK_4` | Block variant (`BLOCK_1`, `BLOCK_4`, `BLOCK_8`, etc.) |
| `--max_inner_steps` | 100       | VLA inner-loop steps per outer step                   |
| `--num_attempts`    | 1         | Meta-RL attempts per episode                          |
| `--group_n`         | 1         | Group size (for GRPO/GiGPO)                           |
| `--do_reflection`   | off       | Enable reflection phase                               |
| `--no_reward`       | off       | Run without reward (for debugging)                    |
| `--seed`            | 0         | Random seed                                           |


**Step 6: Verify server connectivity (optional but recommended)**

Before running full training, verify the servers respond correctly:

```bash
# From the language-table venv:
cd /path/to/language-table
export PYTHONPATH=${PWD}:$PYTHONPATH

# Quick TCP check
ltvenv/bin/python -m language_table.lamer.test_connection \
    --host localhost --port 50051

# Full protocol test (reset, step, restart, reflect)
ltvenv/bin/python -m language_table.lamer.test_connection \
    --host localhost --port 50051 --val_port 50052 --full
```

**Local dry-run (no GPU / no model needed)**

You can test the full three-process setup locally without a real LLM by using
LaMer's data preprocessing + the `examples/test_env.py` script:

```bash
# 1. Start both env servers (two terminals, language-table venv)
#    Use small env counts for local testing
ltvenv/bin/python -m language_table.lamer.server_main --port 50051 --num_envs 2
ltvenv/bin/python -m language_table.lamer.server_main --port 50052 --num_envs 2

# 2. Run connection test (language-table venv, third terminal)
ltvenv/bin/python -m language_table.lamer.test_connection \
    --host localhost --port 50051 --val_port 50052 --full

# 3. (Optional) Run test_env.py against the remote servers (LaMer venv)
#    First edit examples/test_env.py and set env_name = 'language_table'
cd /path/to/LaMer && conda activate lamer
python -m examples.test_env
```

This exercises the full TCP protocol, env manager, and meta-RL loop (reset →
step → restart → reflect) without needing any GPU resources.

**SLURM**

A ready-to-use SLURM wrapper is provided at `scripts/submit_language_table.sh`.
It reads `.env.language_table` and `.env.language_table.secrets`, exports the
variables, and then submits `scripts/slurm/lamer_language_table.slurm`.

The SLURM job runs everything on a single node:

1. Starts train + val env servers as background processes
2. Waits for both ports to become ready (`nc -z` polling)
3. Runs the connection test to verify protocol correctness
4. Launches `verl.trainer.main_ppo` for training

Submit with:

```bash
scripts/submit_language_table.sh
```

SLURM networking notes:

- All three processes (train server, val server, LaMer trainer) run on the
**same node**, communicating via `127.0.0.1`. No inter-node networking needed.
- The script uses `trap cleanup EXIT` to kill env servers when the job ends.
- If you need multi-node setups (e.g., env servers on CPU nodes, trainer on GPU
nodes), replace `127.0.0.1` with the compute node hostname and ensure the
chosen ports are not blocked by firewall rules. Test with:
`nc -zv <compute-node> 50051`

### Test Environment

We provide the script `examples/test_env.py` for testing environments and inspecting game states, prompts, and reward signals.

```bash
python -m examples.test_env
```

You can specify the environment by modifying the `env_name`:

```python
env_name = 'sokoban'  # 'minesweeper' or 'sokoban' or 'webshop' or 'alfworld' or 'language_table' (requires remote server)
```

This script simulates random policy on the environment using only CPU and does not require any GPU resources.
