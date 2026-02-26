#!/bin/bash
#SBATCH -D /gpfs/projects/weirdlab/mateo/projects/LaMer               # TODO CHANGE: path to your LaMer repo
#SBATCH --job-name=lamer-minesweeper
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/mateogc/slurm_logs/%x-%j.out          # TODO CHANGE: your slurm log dir
#SBATCH --error=/gpfs/scrubbed/mateogc/slurm_logs/%x-%j.err           # TODO CHANGE: your slurm log dir

set -euo pipefail

source /gpfs/home/mateogc/setup_mateo.sh                              # TODO CHANGE: path to your setup script

module load conda/Miniforge3-25.3.1-3
module load gcc/13.4.0
module load cmake/3.31.8

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate lamer                                                   # TODO CHANGE: your conda env name
set -u
echo "Environment activated!"
echo "START TIME: $(date)"

######################
### Environment ######
######################
# Ray can hang on SLURM when it auto-detects all CPUs. Cap explicitly.
export RAY_NUM_CPUS=${SLURM_CPUS_PER_TASK:-50}

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_TIMEOUT=1200000
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

######################
### Run Example ######
######################
# Run the upstream example script exactly as provided.
# It hard-codes trainer.n_gpus_per_node=4, which matches --gres=gpu:4 above.
bash examples/minesweeper/lamer_minesweeper_qwen3_4b.sh

echo "END TIME: $(date)"
