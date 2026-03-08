#!/bin/bash
#SBATCH --job-name=lamer_nav_qwen3_4b
#SBATCH --array=0-1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=960G
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --chdir=.
#SBATCH --output=slurm-%x-%A-%a.out
#SBATCH --error=slurm-%x-%A-%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMER_DIR="${LAMER_DIR:-${SCRIPT_DIR}}"
LAMER_CONDA_ENV="${LAMER_CONDA_ENV:-lamer}"

module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${LAMER_CONDA_ENV}"
set -u
cd "${LAMER_DIR}"

SCRIPTS=(
    # examples/navigation/lamer_nav_base_4_step.sh
    # examples/navigation/lamer_nav_base_6_step.sh
    # examples/navigation/lamer_nav_meta_4_step.sh
    # examples/navigation/lamer_nav_meta_6_step.sh
    # examples/navigation/lamer_nav_meta_extreme_4_step.sh
    examples/navigation/lamer_nav_meta_4_step_history_and_reflection.sh
    examples/navigation/lamer_nav_meta_extreme_4_step_history_and_reflection.sh
)

bash "${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"
