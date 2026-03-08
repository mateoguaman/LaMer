#!/bin/bash
#SBATCH -D .
#SBATCH --job-name=vlm-navigation
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=50
#SBATCH --mem=1128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# safer bash options
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SETUP_SCRIPT="${SETUP_SCRIPT:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlmn}"

if [ -n "${SETUP_SCRIPT}" ] && [ -f "${SETUP_SCRIPT}" ]; then
    source "${SETUP_SCRIPT}"
fi

module load conda/Miniforge3-25.3.1-3
module load gcc/13.4.0
module load cmake/3.31.8
module load ffmpeg/7.1

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV_NAME}"
set -u
echo "Environment activated!"
echo "START TIME: $(date)"
cd "${PROJECT_DIR}"

MODEL=vlm-navigation
TASK=sft
PRECISION=lora_vamos_10pct_gpt5_mini_cocoqa_localized_narratives
ACCELERATOR=multi_gpu
OPTIONAL_ARGS=""

######################
### Environment ######
######################
export NUM_NODES=${SLURM_NNODES:-1}
export GPUS_PER_NODE=4
export WORLD_SIZE=$(( NUM_NODES * GPUS_PER_NODE ))
export CONFIG_FILE="recipes/$MODEL/$TASK/lora_configs/config_$PRECISION.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
	  echo "Config file not found: $CONFIG_FILE"
	    echo "PWD: $(pwd)"
	      ls -la "recipes/$MODEL/$TASK/" || true
	        exit 1
fi

echo "Config file contents:"
cat "$CONFIG_FILE"

# robust parse of gradient_accumulation_steps from YAML
GRAD_ACC_STEPS="$(grep -oE 'gradient_accumulation_steps:[[:space:]]*[0-9]+' "$CONFIG_FILE" | awk '{print $NF}' || true)"
if [ -z "${GRAD_ACC_STEPS}" ]; then
	  echo "Failed to extract gradient_accumulation_steps; defaulting to 4"
	    GRAD_ACC_STEPS=4
    else
	      echo "gradient_accumulation_steps: $GRAD_ACC_STEPS"
fi
export GRAD_ACC_STEPS

# allow CLI override if present in OPTIONAL_ARGS
IFS=' ' read -ra ARGS <<< "$OPTIONAL_ARGS"
for arg in "${ARGS[@]}"; do
	if [[ "$arg" == --gradient_accumulation_steps=* ]]; then
		GRAD_ACC_STEPS="${arg#*=}"
		export GRAD_ACC_STEPS
		echo "Overriding gradient_accumulation_steps from OPTIONAL_ARGS: $GRAD_ACC_STEPS"
		break
	fi
done

######################
#### Networking ######
######################
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT=49152

######################
####  Launchers  #####
######################
export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_TIMEOUT=1200000
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_ENABLE_HF_TRANSFER=1
export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info

CMD="scripts/run_${TASK}.py $CONFIG_FILE $OPTIONAL_ARGS"

LAUNCHER="accelerate launch \
	--config_file recipes/accelerate_configs/${ACCELERATOR}.yaml \
	--gradient_accumulation_steps ${GRAD_ACC_STEPS} \
	--num_machines ${NUM_NODES} \
	--num_processes ${WORLD_SIZE} \
	--main_process_ip ${MASTER_ADDR} \
	--main_process_port ${MASTER_PORT} \
	--machine_rank \$SLURM_PROCID \
	--max_restarts 1 \
	--tee 3"

# IMPORTANT: no `clear` here (it can fail without TERM)
# Start exactly one launcher per node (ntasks-per-node=1)
srun --wait=60 --kill-on-bad-exit=1 bash -lc "$LAUNCHER $CMD"

echo "END TIME: $(date)"
