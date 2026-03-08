## Meta-RL Induces Exploration in Language Agents

[Yulun Jiang*](https://yljblues.github.io), [Liangze Jiang*](https://liangzejiang.github.io/), [Damien Teney](https://www.damienteney.info/), [Michael Moor**](https://michaelmoor.me/), [Maria Brbić**](https://brbiclab.epfl.ch/team/)

[`Project page`](https://brbiclab.epfl.ch/projects/LaMer/) | [`Paper`](https://arxiv.org/abs/2512.16848) | [`BibTeX`](#citing) 

--------------

This repo contains the source code of 🌊LaMer, a Meta-RL framework of training LLM agents to actively explore and adapt to the environment at test time (ICLR '26).

</br>
<div align="left" style="padding: 0 0pt">
<img src="assets/intro.png" style="width: auto; height: 250px;">
</div>
</br>

## Training
To train the LLM Agent with LaMer:
```
bash examples/minesweeper/lamer_minesweeper_qwen3_4b.sh
```
To train the LLM Agent with RL baselines:
```
bash examples/minesweeper/gigpo_minesweeper_qwen3_4b.sh
```
See the `examples` folder for more examples. 

</br>
<div align="left" style="padding: 0 0pt">
<img src="assets/training.png" style="width: auto; height: 250px;">
</div>
</br>

## Environment
Please follow this [note](https://github.com/mlbio-epfl/LaMer/tree/main/assets/env.md) to install and test the agent environments.

### Language Table (PyBullet tabletop manipulation)

Language Table runs in a separate process (its own Python venv with `gym==0.23` +
PyBullet) and communicates with LaMer over TCP. See the full setup instructions in
[assets/env.md](assets/env.md#language-table-pybullet-tabletop-manipulation).

**Quick start:**

```bash
# 1. Start env servers with VLA (in the language-table repo, separate terminals)
cd /path/to/language-table && export PYTHONPATH=${PWD}:$PYTHONPATH
ltvenv/bin/python -m language_table.lamer.server_main --port 50051 --num_envs 8 --num_attempts 3 \
    --vla_checkpoint /path/to/checkpoints/bc_resnet_sim_checkpoint_955000
ltvenv/bin/python -m language_table.lamer.server_main --port 50052 --num_envs 16 --num_attempts 3 \
    --vla_checkpoint /path/to/checkpoints/bc_resnet_sim_checkpoint_955000

# 2. Run LaMer training (in this repo)
python3 -m verl.trainer.main_ppo \
    env.env_name=language_table \
    env.remote=True \
    env.remote_address=localhost:50051 \
    env.remote_val_address=localhost:50052 \
    env.num_attempts=3 \
    ... # other config
```

The `--vla_checkpoint` flag loads the pre-trained LAVA policy for the inner loop.
Without it, the server falls back to random actions. The checkpoint is downloaded
automatically on first run (see the SLURM script), or manually:

```bash
wget -O /path/to/checkpoints/bc_resnet_sim_checkpoint_955000 \
    https://storage.googleapis.com/gresearch/robotics/language_table_checkpoints/bc_resnet_sim_checkpoint_955000
```

To submit the single-node SLURM job with env-var based configuration instead of
editing paths in the script:

```bash
scripts/submit_language_table.sh
```

### SLURM setup

For repeatable SLURM runs, create a local env file once and keep your
cluster-specific paths there:

```bash
cp .env.language_table.example .env.language_table
cp .env.language_table.secrets.example .env.language_table.secrets
```

Then edit `.env.language_table` and set the values for your cluster, especially:

- `LANGTABLE_DIR`
- `LANGTABLE_PYTHON`
- `LAMER_CONDA_ENV`
- `TRAIN_DATA_PATH`
- `VAL_DATA_PATH`
- `CHECKPOINT_ROOT`
- `VLA_CHECKPOINT_DIR`
- `SETUP_SCRIPT` if your cluster needs custom shell initialization

Then edit `.env.language_table.secrets` for secrets such as:

- `WANDB_API_KEY` or `WANDB_API_KEY_FILE`
- `HF_TOKEN` if you do not want to load it via `HF_TOKEN_FILE`

Bootstrap both environments with conda:

```bash
scripts/bootstrap_language_table.sh
```

Once that file is configured, submit the Language Table job with:

```bash
scripts/submit_language_table.sh
```

You can still override any variable for a single run:

```bash
RUN_NAME=lt_smoketest TRAIN_DATA_PATH=/path/to/train.parquet \
VAL_DATA_PATH=/path/to/test.parquet \
scripts/submit_language_table.sh
```

The wrapper exports the environment variables and then calls:

```bash
sbatch scripts/slurm/lamer_language_table.slurm
```

**Docs:**
- [VLA integration plan and design decisions](docs/vla_integration.md)
- [How to integrate a custom VLA](docs/custom_vla.md)


## Acknowledgements
This work is built upon [verl](https://github.com/volcengine/verl), [verl-agent](https://github.com/langfengQ/verl-agent), [reflexion](https://github.com/noahshinn/reflexion), [RAGEN](https://github.com/mll-lab-nu/RAGEN). We thank the authors and contributors of these projects for sharing their valuable work.


## Citing
If you find our code useful, please consider citing:

```
@inproceedings{jiang2026metarl,
    title={Meta-RL Induces Exploration in Language Agents},
    author={Yulun Jiang and Liangze Jiang and Damien Teney and Michael Moor and Maria Brbic},
    booktitle={International Conference on Learning Representations}
    year={2026}
}
```
