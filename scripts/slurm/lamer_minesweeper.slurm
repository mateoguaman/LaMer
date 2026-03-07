#!/bin/bash
#SBATCH -D .
#SBATCH --job-name=lamer-minesweeper
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMER_DIR="${LAMER_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SETUP_SCRIPT="${SETUP_SCRIPT:-}"
LAMER_CONDA_ENV="${LAMER_CONDA_ENV:-lamer}"

if [ -n "${SETUP_SCRIPT}" ] && [ -f "${SETUP_SCRIPT}" ]; then
    source "${SETUP_SCRIPT}"
fi

module load conda/Miniforge3-25.3.1-3
module load gcc/13.4.0
module load cmake/3.31.8

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${LAMER_CONDA_ENV}"
set -u
echo "Environment activated!"
echo "START TIME: $(date)"
echo "LAMER_DIR: ${LAMER_DIR}"

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
cd "${LAMER_DIR}"
bash examples/minesweeper/lamer_minesweeper_qwen3_4b.sh

echo "END TIME: $(date)"
