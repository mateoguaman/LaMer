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

# Which benchmarks to run (comma-separated: sharded_smoke,benchmark_end_to_end,vla,envs,outer)
BENCHMARKS="${BENCHMARKS:-benchmark_end_to_end}"
BENCH_END_TO_END_PRESET="${BENCH_END_TO_END_PRESET:-doc_baseline}"

# End-to-end benchmark config
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$HOME/data/verl-agent/text/train.parquet}"
VAL_DATA_PATH="${VAL_DATA_PATH:-$HOME/data/verl-agent/text/test.parquet}"
BENCH_WARMUP_ITERATIONS="${BENCH_WARMUP_ITERATIONS:-2}"
BENCH_MEASURED_ITERATIONS="${BENCH_MEASURED_ITERATIONS:-3}"
BENCH_TRACE_INNER_STEPS="${BENCH_TRACE_INNER_STEPS:-0}"
BENCH_TRAIN_SHARD_COUNTS="${BENCH_TRAIN_SHARD_COUNTS:-1,2,4}"
BENCH_PROCS_PER_GPU="${BENCH_PROCS_PER_GPU:-1}"
BENCH_ENV_SERVER_GPUS="${BENCH_ENV_SERVER_GPUS:-${BENCH_ENV_SERVER_GPU:-4,5,6,7}}"
BENCH_SKIP_VAL_SERVER="${BENCH_SKIP_VAL_SERVER:-1}"
BENCH_TRAINER_VISIBLE_GPUS="${BENCH_TRAINER_VISIBLE_GPUS:-0,1,2,3}"
BENCH_RAY_NUM_CPUS="${BENCH_RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-64}}"

# Training config used by both the normal training job and the end-to-end benchmark
RUN_NAME="${RUN_NAME:-language_table_lamer_qwen3_4b}"
REWARD_TYPE="${REWARD_TYPE:-block2block}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-gigpo}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-4}"
VAL_NUM_ENVS="${VAL_NUM_ENVS:-4}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_INNER_STEPS="${MAX_INNER_STEPS:-5}"
NUM_ATTEMPTS="${NUM_ATTEMPTS:-2}"
MAX_TURNS="${MAX_TURNS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
KL_REWARD_COEF="${KL_REWARD_COEF:-0.001}"
TRAIN_TASK_LOCATIONS="${TRAIN_TASK_LOCATIONS:-}"
TRAIN_TASK_SHAPES="${TRAIN_TASK_SHAPES:-}"
TRAIN_TASK_COLORS="${TRAIN_TASK_COLORS:-}"
TRAIN_TASK_N_STEPS="${TRAIN_TASK_N_STEPS:-2}"
VAL_TASK_LOCATIONS="${VAL_TASK_LOCATIONS:-}"
VAL_TASK_SHAPES="${VAL_TASK_SHAPES:-}"
VAL_TASK_COLORS="${VAL_TASK_COLORS:-}"
VAL_TASK_N_STEPS="${VAL_TASK_N_STEPS:-3}"

# VLA-only preprocess modes for the standalone VLA benchmark
VLA_PREPROCESS_MODES="${VLA_PREPROCESS_MODES:-original}"
PREPROCESS_MODE="${PREPROCESS_MODE:-jax_gpu}"

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
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
fi
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
echo "BENCH_END_TO_END_PRESET: ${BENCH_END_TO_END_PRESET}"
echo "TRAIN_DATA_PATH: ${TRAIN_DATA_PATH}"
echo "VAL_DATA_PATH: ${VAL_DATA_PATH}"
echo "BENCH_WARMUP_ITERATIONS: ${BENCH_WARMUP_ITERATIONS}"
echo "BENCH_MEASURED_ITERATIONS: ${BENCH_MEASURED_ITERATIONS}"
echo "BENCH_TRAIN_SHARD_COUNTS: ${BENCH_TRAIN_SHARD_COUNTS}"
echo "BENCH_PROCS_PER_GPU: ${BENCH_PROCS_PER_GPU}"
echo "BENCH_ENV_SERVER_GPUS: ${BENCH_ENV_SERVER_GPUS}"
echo "BENCH_SKIP_VAL_SERVER: ${BENCH_SKIP_VAL_SERVER}"
echo "BENCH_TRAINER_VISIBLE_GPUS: ${BENCH_TRAINER_VISIBLE_GPUS}"
echo "PREPROCESS_MODE: ${PREPROCESS_MODE}"
echo "REWARD_TYPE: ${REWARD_TYPE}"
echo "TRAIN_NUM_ENVS: ${TRAIN_NUM_ENVS}"
echo "VAL_NUM_ENVS: ${VAL_NUM_ENVS}"
echo "GROUP_SIZE: ${GROUP_SIZE}"
echo "MAX_INNER_STEPS: ${MAX_INNER_STEPS}"
echo "NUM_ATTEMPTS: ${NUM_ATTEMPTS}"
echo "MAX_TURNS: ${MAX_TURNS}"
echo ""

# GPU info
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# GPU assignments
ENV_SERVER_GPU_POOL_CSV="${BENCH_ENV_SERVER_GPUS}"
ENV_SERVER_GPU="${ENV_SERVER_GPU_POOL_CSV%%,*}"
TRAIN_GPUS="${BENCH_TRAINER_VISIBLE_GPUS}"
IFS=',' read -r -a ENV_SERVER_GPU_POOL <<< "${ENV_SERVER_GPU_POOL_CSV}"
IFS=',' read -r -a TRAIN_SHARD_COUNTS <<< "${BENCH_TRAIN_SHARD_COUNTS}"

if [ "${#ENV_SERVER_GPU_POOL[@]}" -eq 0 ]; then
    echo "ERROR: BENCH_ENV_SERVER_GPUS must contain at least one GPU id."
    exit 1
fi
if [ "${#TRAIN_SHARD_COUNTS[@]}" -eq 0 ]; then
    echo "ERROR: BENCH_TRAIN_SHARD_COUNTS must contain at least one shard count."
    exit 1
fi

join_first_n_csv() {
    local count="$1"
    local slice=("${ENV_SERVER_GPU_POOL[@]:0:${count}}")
    local IFS=,
    echo "${slice[*]}"
}

# Export the same training config variables the normal training launcher consumes,
# so benchmark_end_to_end.py resolves from the live job config instead of a
# hardcoded snapshot.
export RUN_NAME REWARD_TYPE ADV_ESTIMATOR
export TRAIN_NUM_ENVS VAL_NUM_ENVS GROUP_SIZE
export MAX_INNER_STEPS NUM_ATTEMPTS MAX_TURNS
export LEARNING_RATE BATCH_SIZE MICRO_BATCH_SIZE
export USE_KL_LOSS KL_LOSS_COEF KL_LOSS_TYPE
export USE_KL_IN_REWARD KL_REWARD_COEF
export TRAIN_TASK_LOCATIONS TRAIN_TASK_SHAPES TRAIN_TASK_COLORS TRAIN_TASK_N_STEPS
export VAL_TASK_LOCATIONS VAL_TASK_SHAPES VAL_TASK_COLORS VAL_TASK_N_STEPS

########################################
### Benchmark 0: Sharded Smoke #########
########################################
if echo "${BENCHMARKS}" | grep -q "sharded_smoke"; then
    echo ""
    echo "============================================"
    echo "BENCHMARK 0: Sharded Remote Smoke Test"
    echo "============================================"
    echo "Running live sharded remote smoke test with ${LANGTABLE_PYTHON}..."

    SHARDED_SMOKE_OUTPUT_DIR="${BENCH_OUTPUT_DIR}/sharded_smoke"
    mkdir -p "${SHARDED_SMOKE_OUTPUT_DIR}"

    LANGTABLE_PYTHON="${LANGTABLE_PYTHON}" \
    LANGTABLE_DIR="${LANGTABLE_DIR}" \
    python "${LAMER_DIR}/scripts/test_sharded_grouping_smoke.py" \
        --test live_remote_roundtrip \
        2>&1 | tee "${SHARDED_SMOKE_OUTPUT_DIR}/live_remote_roundtrip.log"

    echo "Sharded remote smoke test complete."
fi

########################################
### Benchmark 1: End-to-End ###########
########################################
if echo "${BENCHMARKS}" | grep -q "benchmark_end_to_end"; then
    echo ""
    echo "============================================"
    echo "BENCHMARK 1: End-to-End Training"
    echo "============================================"
    echo "Running benchmark_end_to_end.py on trainer GPUs ${TRAIN_GPUS} with env server GPU pool ${ENV_SERVER_GPU_POOL_CSV}..."

    END_TO_END_ROOT_DIR="${BENCH_OUTPUT_DIR}/benchmark_end_to_end"
    mkdir -p "${END_TO_END_ROOT_DIR}"

    for SHARD_COUNT_RAW in "${TRAIN_SHARD_COUNTS[@]}"; do
        SHARD_COUNT="${SHARD_COUNT_RAW//[[:space:]]/}"
        if ! [[ "${SHARD_COUNT}" =~ ^[0-9]+$ ]] || [ "${SHARD_COUNT}" -lt 1 ]; then
            echo "ERROR: invalid shard count '${SHARD_COUNT_RAW}' in BENCH_TRAIN_SHARD_COUNTS=${BENCH_TRAIN_SHARD_COUNTS}"
            exit 1
        fi
        MAX_SHARDS=$(( ${#ENV_SERVER_GPU_POOL[@]} * BENCH_PROCS_PER_GPU ))
        if [ "${SHARD_COUNT}" -gt "${MAX_SHARDS}" ]; then
            echo "ERROR: shard count ${SHARD_COUNT} exceeds max shards (${#ENV_SERVER_GPU_POOL[@]} GPUs * ${BENCH_PROCS_PER_GPU} procs/GPU = ${MAX_SHARDS})"
            exit 1
        fi

        # How many physical GPUs and procs/GPU for this shard count.
        # When SHARD_COUNT <= #GPUs: use SHARD_COUNT GPUs, 1 proc each.
        # When SHARD_COUNT > #GPUs: use all GPUs, multiple procs each.
        #   We pick procs_per_gpu so that gpus * procs_per_gpu == SHARD_COUNT.
        #   SHARD_COUNT must be divisible by the number of GPUs used.
        if [ "${SHARD_COUNT}" -le "${#ENV_SERVER_GPU_POOL[@]}" ]; then
            NUM_GPUS_NEEDED="${SHARD_COUNT}"
            RUN_PROCS_PER_GPU=1
        else
            NUM_GPUS_NEEDED="${#ENV_SERVER_GPU_POOL[@]}"
            if [ $(( SHARD_COUNT % NUM_GPUS_NEEDED )) -ne 0 ]; then
                echo "ERROR: shard count ${SHARD_COUNT} is not divisible by ${NUM_GPUS_NEEDED} GPUs. Use a shard count that is a multiple of the GPU count for intra-GPU sharding."
                exit 1
            fi
            RUN_PROCS_PER_GPU=$(( SHARD_COUNT / NUM_GPUS_NEEDED ))
        fi

        SHARD_ENV_SERVER_GPUS="$(join_first_n_csv "${NUM_GPUS_NEEDED}")"
        RUN_OUTPUT_DIR="${END_TO_END_ROOT_DIR}/shards_${SHARD_COUNT}"
        mkdir -p "${RUN_OUTPUT_DIR}"

        echo ""
        echo "--- Shard sweep run: train_shards=${SHARD_COUNT} env_gpus=${SHARD_ENV_SERVER_GPUS} procs_per_gpu=${RUN_PROCS_PER_GPU} ---"

        BENCH_CMD=(
            python "${LAMER_DIR}/benchmarks/benchmark_end_to_end.py"
            --mode spawn
            --preset "${BENCH_END_TO_END_PRESET}"
            --train-data "${TRAIN_DATA_PATH}"
            --val-data "${VAL_DATA_PATH}"
            --output-dir "${RUN_OUTPUT_DIR}"
            --model-path "${MODEL_PATH}"
            --preprocess-mode "${PREPROCESS_MODE}"
            --warmup-iterations "${BENCH_WARMUP_ITERATIONS}"
            --measured-iterations "${BENCH_MEASURED_ITERATIONS}"
            --langtable-python "${LANGTABLE_PYTHON}"
            --langtable-dir "${LANGTABLE_DIR}"
            --vla-checkpoint "${VLA_CHECKPOINT}"
            --env-server-gpu "${SHARD_ENV_SERVER_GPUS}"
            --procs-per-gpu "${RUN_PROCS_PER_GPU}"
            --trainer-visible-gpus "${TRAIN_GPUS}"
            --ray-num-cpus "${BENCH_RAY_NUM_CPUS}"
        )
        if [ "${BENCH_SKIP_VAL_SERVER}" = "1" ]; then
            BENCH_CMD+=(--skip-val-server)
        fi
        if [ "${BENCH_TRACE_INNER_STEPS}" = "1" ]; then
            BENCH_CMD+=(--benchmark-trace-inner-steps)
        fi

        "${BENCH_CMD[@]}" 2>&1 | tee "${RUN_OUTPUT_DIR}/benchmark_end_to_end.log"
    done

    python -m benchmarks.end_to_end_sweep \
        --root-dir "${END_TO_END_ROOT_DIR}" \
        2>&1 | tee "${END_TO_END_ROOT_DIR}/sweep_manifest.log"

    echo "End-to-end benchmark complete."
fi

########################################
### Benchmark 2: VLA Inference #########
########################################
if echo "${BENCHMARKS}" | grep -q "vla"; then
    echo ""
    echo "============================================"
    echo "BENCHMARK 2: VLA (LAVA) Inference"
    echo "============================================"
    echo "Testing batch sizes on GPU ${ENV_SERVER_GPU}..."

    CUDA_VISIBLE_DEVICES=${ENV_SERVER_GPU} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    ${LANGTABLE_PYTHON} -m language_table.lamer.benchmark_vla_standalone \
        --checkpoint_dir "${VLA_CHECKPOINT_DIR_ARG}" \
        --batch_sizes "1,2,4,8,16,32,64,128,256,512,1024,2048,4096" \
        --num_warmup 3 --num_iters 20 \
        --modes "${VLA_PREPROCESS_MODES}" \
        2>&1 | tee "${BENCH_OUTPUT_DIR}/vla_inference.log"

    echo "VLA benchmark complete."
fi

########################################
### Benchmark 3: Env Scaling ###########
########################################
if echo "${BENCHMARKS}" | grep -q "envs"; then
    echo ""
    echo "============================================"
    echo "BENCHMARK 3: Environment Scaling"
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
### Benchmark 4: Outer-Loop LLM ########
########################################
if echo "${BENCHMARKS}" | grep -q "outer"; then
    echo ""
    echo "============================================"
    echo "BENCHMARK 4: Outer-Loop LLM (single GPU)"
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
