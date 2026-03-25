#!/bin/bash
#SBATCH -D .
#SBATCH --job-name=lamer-bench
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=4:00:00

# NOTE: --output/--error are set by submit_benchmarks.sh via SLURM_LOG_DIR.
# If submitting directly, logs go to slurm default (cwd).

set -euo pipefail

######################
### Config ###########
######################
# LAMER_DIR must be set by the submit wrapper (or .env.language_table).
# We cannot derive it from SCRIPT_DIR because SLURM copies the script to
# /var/spool/slurmd/jobNNN/ before executing it.
if [ -z "${LAMER_DIR:-}" ]; then
    echo "ERROR: LAMER_DIR is not set."
    echo "Use benchmarks/submit_benchmarks.sh to submit this job."
    exit 1
fi

# Source .env.language_table FIRST so all derived paths pick up cluster config
if [ -f "${LAMER_DIR}/.env.language_table" ]; then
    # shellcheck disable=SC1091
    source "${LAMER_DIR}/.env.language_table"
fi

LANGTABLE_DIR="${LANGTABLE_DIR:-${LAMER_DIR}/../language-table}"
LANGTABLE_CONDA_ENV="${LANGTABLE_CONDA_ENV:-ltvenv}"
LAMER_CONDA_ENV="${LAMER_CONDA_ENV:-lamer}"

# VLA checkpoint — read from .env.language_table or env var
VLA_CHECKPOINT_DIR="${VLA_CHECKPOINT_DIR:-}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-}"

if [ -z "${VLA_CHECKPOINT_DIR}" ]; then
    echo "ERROR: VLA_CHECKPOINT_DIR is not set."
    echo "Set it in .env.language_table or export it before submitting."
    exit 1
fi

if [ -z "${VLA_CHECKPOINT}" ]; then
    VLA_CHECKPOINT="${VLA_CHECKPOINT_DIR}/bc_resnet_sim_checkpoint_955000"
fi
VLA_CHECKPOINT_DIR_ARG="$(dirname "${VLA_CHECKPOINT}")"

# Output directory
BENCH_OUTPUT_DIR="${BENCH_OUTPUT_DIR:-${LAMER_DIR}/benchmarks/results/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${BENCH_OUTPUT_DIR}"

# Which benchmarks to run (comma-separated: vla,envs,outer)
BENCHMARKS="${BENCHMARKS:-vla,envs,outer}"

# Outer-loop model
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"

######################
### Setup ############
######################
if [ -z "${CONDA_PREFIX:-}" ]; then
    module load conda/Miniforge3-25.3.1-3
fi
module load gcc/13.4.0

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${LAMER_CONDA_ENV}"
set -u
LANGTABLE_PYTHON="$(conda run -n "${LANGTABLE_CONDA_ENV}" which python)"

######################
### Environment ######
######################
export NCCL_ASYNC_ERROR_HANDLING=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TF_CPP_MIN_LOG_LEVEL=2
export GRPC_VERBOSITY=ERROR

# Create cache dirs that Tillicum needs
[ -n "${CONDA_PKGS_DIRS:-}" ] && mkdir -p "${CONDA_PKGS_DIRS}"
[ -n "${PIP_CACHE_DIR:-}" ] && mkdir -p "${PIP_CACHE_DIR}"
[ -n "${TMPDIR:-}" ] && mkdir -p "${TMPDIR}"
[ -n "${XDG_CACHE_HOME:-}" ] && mkdir -p "${XDG_CACHE_HOME}"

# Language-table needs its repo on PYTHONPATH
export PYTHONPATH="${LANGTABLE_DIR}:${LAMER_DIR}:${PYTHONPATH:-}"

echo "============================================"
echo "LaMer H200 Benchmark Suite"
echo "============================================"
echo "START TIME: $(date)"
echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-n/a}"
echo "LAMER_DIR: ${LAMER_DIR}"
echo "LANGTABLE_DIR: ${LANGTABLE_DIR}"
echo "LANGTABLE_CONDA_ENV: ${LANGTABLE_CONDA_ENV}"
echo "VLA_CHECKPOINT: ${VLA_CHECKPOINT}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "OUTPUT: ${BENCH_OUTPUT_DIR}"
echo "BENCHMARKS: ${BENCHMARKS}"
echo ""

# GPU info
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# GPU assignments
ENV_SERVER_GPU=0
TRAIN_GPUS="1,2,3,4"

########################################
### Benchmark 1: VLA Inference #########
########################################
if echo "${BENCHMARKS}" | grep -q "vla"; then
    echo ""
    echo "============================================"
    echo "BENCHMARK 1: VLA (LAVA) Inference"
    echo "============================================"
    echo "Testing batch sizes on GPU ${ENV_SERVER_GPU}..."

    CUDA_VISIBLE_DEVICES=${ENV_SERVER_GPU} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    ${LANGTABLE_PYTHON} -m language_table.lamer.benchmark_vla_standalone \
        --checkpoint_dir "${VLA_CHECKPOINT_DIR_ARG}" \
        --batch_sizes "1,2,4,8,16,32,64,128,256,512,1024,2048,4096" \
        --num_warmup 3 --num_iters 20 \
        2>&1 | tee "${BENCH_OUTPUT_DIR}/vla_inference.log"

    echo "VLA benchmark complete."
fi

########################################
### Benchmark 2: Env Scaling ###########
########################################
if echo "${BENCHMARKS}" | grep -q "envs"; then
    echo ""
    echo "============================================"
    echo "BENCHMARK 2: Environment Scaling"
    echo "============================================"
    echo "Testing environment counts on GPU ${ENV_SERVER_GPU}..."

    CUDA_VISIBLE_DEVICES=${ENV_SERVER_GPU} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    ${LANGTABLE_PYTHON} -m language_table.lamer.benchmark_envs_standalone \
        --checkpoint_dir "${VLA_CHECKPOINT_DIR_ARG}" \
        --env_counts "4,8,16,32,64,128,256,512,1024,2048" \
        --inner_steps 20 \
        --cpus_per_gpu 8 \
        --time_budget_s 120 \
        2>&1 | tee "${BENCH_OUTPUT_DIR}/env_scaling.log"

    echo "Env scaling benchmark complete."
fi

########################################
### Benchmark 3: Outer-Loop LLM ########
########################################
if echo "${BENCHMARKS}" | grep -q "outer"; then
    echo ""
    echo "============================================"
    echo "BENCHMARK 3: Outer-Loop LLM (single GPU)"
    echo "============================================"
    echo "Testing on GPU 1 (single GPU, no FSDP)..."

    # Single-GPU benchmark (no distributed)
    CUDA_VISIBLE_DEVICES=1 \
    python -m benchmarks.benchmark_outer_loop \
        --model "${MODEL_PATH}" \
        --max_prompt_length 2048 \
        --max_response_length 1024 \
        --batch_sizes "1,2,4,8,16,32,64" \
        --lora_ranks "8,16,32,64" \
        --num_warmup 2 --num_iters 5 \
        --output_json "${BENCH_OUTPUT_DIR}/outer_loop_results.json" \
        2>&1 | tee "${BENCH_OUTPUT_DIR}/outer_loop.log"

    echo "Outer-loop benchmark complete."
fi

########################################
### Summary ############################
########################################
echo ""
echo "============================================"
echo "ALL BENCHMARKS COMPLETE"
echo "============================================"
echo "Results saved to: ${BENCH_OUTPUT_DIR}"
echo "END TIME: $(date)"
ls -la "${BENCH_OUTPUT_DIR}/"
