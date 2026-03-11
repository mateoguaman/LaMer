#!/bin/bash
#SBATCH --job-name=lamer_nav_qwen3_4b
#SBATCH --array=0-1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=960G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --chdir=/gpfs/projects/weirdlab/memmelma/projects/LaMer
#SBATCH --output=slurm/%x-%A-%a-out.txt
#SBATCH --error=slurm/%x-%A-%a-err.txt

module purge
module load conda
conda activate lamer

SCRIPTS=(
    # examples/navigation/lamer_nav_base_4_step.sh
    # examples/navigation/lamer_nav_base_6_step.sh
    # examples/navigation/lamer_nav_meta_4_step.sh
    # examples/navigation/lamer_nav_meta_6_step.sh
    # examples/navigation/lamer_nav_meta_extreme_4_step.sh
    # examples/navigation/lamer_nav_meta_4_step_history_and_reflection.sh
    # examples/navigation/lamer_nav_meta_extreme_4_step_history_and_reflection.sh

    examples/navigation/lamer_nav_base_4_step_single.sh
    examples/navigation/lamer_nav_meta_4_step_single_history_and_reflection.sh
)

bash "${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"