#!/usr/bin/env bash
# setup_vm.sh — Bootstrap a fresh Azure VM with LaMer and language-table.
#
# Run this ON each Azure VM after provisioning.
# It clones the repos, creates conda environments, and installs dependencies.
#
# Usage:
#   bash scripts/azure/setup_vm.sh              # uses git clone (needs repos set in .env.azure)
#   bash scripts/azure/setup_vm.sh --skip-clone  # if you already copied/cloned repos manually
set -euo pipefail

SKIP_CLONE=false
if [ "${1:-}" = "--skip-clone" ]; then
    SKIP_CLONE=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAMER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${BASE_LAMER_DIR}/.env.azure" ]; then
    # shellcheck disable=SC1091
    source "${BASE_LAMER_DIR}/.env.azure"
fi
if [ -f "${BASE_LAMER_DIR}/.env.language_table.secrets" ]; then
    # shellcheck disable=SC1091
    source "${BASE_LAMER_DIR}/.env.language_table.secrets"
fi

LAMER_DIR="${LAMER_DIR:-${BASE_LAMER_DIR}}"
LANGTABLE_DIR="${LANGTABLE_DIR:-$(cd "${LAMER_DIR}/.." && pwd)/language-table}"
LAMER_CONDA_ENV="${LAMER_CONDA_ENV:-lamer}"
LANGTABLE_CONDA_ENV="${LANGTABLE_CONDA_ENV:-ltvenv}"
LAMER_GIT_REPO="${LAMER_GIT_REPO:-}"
LANGTABLE_GIT_REPO="${LANGTABLE_GIT_REPO:-}"
VLA_CHECKPOINT_DIR="${VLA_CHECKPOINT_DIR:-}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-}"

echo "=== Azure VM Setup ==="
echo "  LAMER_DIR:       ${LAMER_DIR}"
echo "  LANGTABLE_DIR:   ${LANGTABLE_DIR}"
echo "  LAMER_CONDA_ENV: ${LAMER_CONDA_ENV}"
echo "  LANGTABLE_CONDA_ENV: ${LANGTABLE_CONDA_ENV}"
echo ""

######################
### Clone repos ######
######################
if [ "${SKIP_CLONE}" = false ]; then
    if [ -z "${LAMER_GIT_REPO}" ]; then
        echo "ERROR: LAMER_GIT_REPO must be set in .env.azure"
        exit 1
    fi
    if [ ! -d "${LAMER_DIR}/.git" ]; then
        echo "Cloning LaMer..."
        git clone "${LAMER_GIT_REPO}" "${LAMER_DIR}"
    else
        echo "LaMer already cloned at ${LAMER_DIR}"
    fi
fi

# Always clone language-table if missing (--skip-clone only skips LaMer,
# which provision_vms.sh already cloned)
if [ ! -d "${LANGTABLE_DIR}/.git" ]; then
    if [ -z "${LANGTABLE_GIT_REPO}" ]; then
        echo "ERROR: LANGTABLE_GIT_REPO must be set in .env.azure"
        exit 1
    fi
    echo "Cloning language-table..."
    git clone "${LANGTABLE_GIT_REPO}" "${LANGTABLE_DIR}"
else
    echo "language-table already cloned at ${LANGTABLE_DIR}"
fi

if [ ! -d "${LAMER_DIR}" ]; then
    echo "ERROR: LAMER_DIR does not exist: ${LAMER_DIR}"
    exit 1
fi
if [ ! -d "${LANGTABLE_DIR}" ]; then
    echo "ERROR: LANGTABLE_DIR does not exist: ${LANGTABLE_DIR}"
    exit 1
fi

######################
### Conda setup ######
######################
# HPC images usually have miniconda pre-installed
if ! command -v conda &>/dev/null; then
    echo "Installing Miniforge..."
    wget -q -O /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash /tmp/miniforge.sh -b -p "${HOME}/miniforge3"
    rm /tmp/miniforge.sh
    export PATH="${HOME}/miniforge3/bin:${PATH}"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

echo ""
echo "=== Setting up LaMer conda env (${LAMER_CONDA_ENV}) ==="
if ! conda env list | awk '{print $1}' | grep -qx "${LAMER_CONDA_ENV}"; then
    conda create -y -n "${LAMER_CONDA_ENV}" python=3.12
fi
set +u
conda activate "${LAMER_CONDA_ENV}"
set -u
python -m pip install --upgrade pip
python -m pip install -r "${LAMER_DIR}/requirements.txt"

echo ""
echo "=== Setting up language-table conda env (${LANGTABLE_CONDA_ENV}) ==="
if ! conda env list | awk '{print $1}' | grep -qx "${LANGTABLE_CONDA_ENV}"; then
    conda create -y -n "${LANGTABLE_CONDA_ENV}" python=3.10
fi
set +u
conda activate "${LANGTABLE_CONDA_ENV}"
set -u
python -m pip install --upgrade pip
python -m pip install -r "${LANGTABLE_DIR}/requirements.txt"
python -m pip install --no-deps \
    git+https://github.com/google-research/scenic.git@ae21d9e884015aa7bc7cf1d489af53d16c249726

######################
### VLA checkpoint ###
######################
if [ -z "${VLA_CHECKPOINT}" ] && [ -n "${VLA_CHECKPOINT_DIR}" ]; then
    VLA_CHECKPOINT="${VLA_CHECKPOINT_DIR}/bc_resnet_sim_checkpoint_955000"
fi

if [ -n "${VLA_CHECKPOINT}" ] && [ ! -f "${VLA_CHECKPOINT}" ]; then
    echo ""
    echo "=== Downloading VLA checkpoint ==="
    mkdir -p "$(dirname "${VLA_CHECKPOINT}")"
    wget -q --show-progress -O "${VLA_CHECKPOINT}" \
        "https://storage.googleapis.com/gresearch/robotics/language_table_checkpoints/bc_resnet_sim_checkpoint_955000"
    echo "Downloaded to ${VLA_CHECKPOINT}"
fi

######################
### Verify GPU #######
######################
echo ""
echo "=== GPU Verification ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

echo "Setup complete on $(hostname)."
echo ""
echo "If this is the ENV SERVER VM, next run:"
echo "  bash ${LAMER_DIR}/scripts/azure/start_env_servers.sh"
echo ""
echo "If this is a TRAINING VM, next run start_training.sh from vm-train-0."
