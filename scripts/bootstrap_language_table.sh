#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAMER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${BASE_LAMER_DIR}/.env.language_table" ]; then
    # shellcheck disable=SC1091
    source "${BASE_LAMER_DIR}/.env.language_table"
fi
if [ -f "${BASE_LAMER_DIR}/.env.language_table.secrets" ]; then
    # shellcheck disable=SC1091
    source "${BASE_LAMER_DIR}/.env.language_table.secrets"
fi
LAMER_DIR="${LAMER_DIR:-${BASE_LAMER_DIR}}"
DEFAULT_LANGTABLE_DIR="${LAMER_DIR}/../language-table"
LANGTABLE_DIR="${LANGTABLE_DIR:-${DEFAULT_LANGTABLE_DIR}}"
LAMER_CONDA_ENV="${LAMER_CONDA_ENV:-lamer}"
LANGTABLE_CONDA_ENV="${LANGTABLE_CONDA_ENV:-ltvenv}"
CONDA_CACHE_DIR="${CONDA_CACHE_DIR:-}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-}"
CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-}"
TMPDIR="${TMPDIR:-}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-}"

if [ ! -d "${LAMER_DIR}" ]; then
    echo "ERROR: LAMER_DIR does not exist: ${LAMER_DIR}"
    exit 1
fi
if [ ! -d "${LANGTABLE_DIR}" ]; then
    echo "ERROR: LANGTABLE_DIR does not exist: ${LANGTABLE_DIR}"
    echo "Set LANGTABLE_DIR in .env.language_table or export it before running."
    exit 1
fi
if [ ! -f "${LAMER_DIR}/requirements.txt" ]; then
    echo "ERROR: Missing ${LAMER_DIR}/requirements.txt"
    exit 1
fi
if [ ! -f "${LANGTABLE_DIR}/requirements.txt" ]; then
    echo "ERROR: Missing ${LANGTABLE_DIR}/requirements.txt"
    exit 1
fi

ensure_conda() {
    if command -v conda >/dev/null 2>&1; then
        return 0
    fi
    if type module >/dev/null 2>&1; then
        module load conda/Miniforge3-25.3.1-3 || true
    fi
    command -v conda >/dev/null 2>&1
}

if ! ensure_conda; then
    echo "ERROR: conda is not available. Load your cluster conda module first."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ -n "${CONDA_CACHE_DIR}" ]; then
    : "${CONDA_PKGS_DIRS:=${CONDA_CACHE_DIR}}"
    : "${CONDA_ENVS_DIRS:=${CONDA_CACHE_DIR}}"
fi

if [ -n "${CONDA_PKGS_DIRS}" ]; then
    export CONDA_PKGS_DIRS
    mkdir -p "${CONDA_PKGS_DIRS}"
fi
if [ -n "${CONDA_ENVS_DIRS}" ]; then
    export CONDA_ENVS_DIRS
    mkdir -p "${CONDA_ENVS_DIRS}"
fi
if [ -n "${PIP_CACHE_DIR}" ]; then
    export PIP_CACHE_DIR
    mkdir -p "${PIP_CACHE_DIR}"
fi
if [ -n "${TMPDIR}" ]; then
    export TMPDIR
    mkdir -p "${TMPDIR}"
fi
if [ -n "${XDG_CACHE_HOME}" ]; then
    export XDG_CACHE_HOME
    mkdir -p "${XDG_CACHE_HOME}"
fi

echo "LAMER_DIR=${LAMER_DIR}"
echo "LANGTABLE_DIR=${LANGTABLE_DIR}"
echo "LAMER_CONDA_ENV=${LAMER_CONDA_ENV}"
echo "LANGTABLE_CONDA_ENV=${LANGTABLE_CONDA_ENV}"
echo "CONDA_CACHE_DIR=${CONDA_CACHE_DIR:-<default>}"
echo "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-<default>}"
echo "PIP_CACHE_DIR=${PIP_CACHE_DIR:-<default>}"
echo "TMPDIR=${TMPDIR:-<default>}"

echo "=== Bootstrapping LaMer env (${LAMER_CONDA_ENV}) ==="
if ! conda env list | awk '{print $1}' | grep -qx "${LAMER_CONDA_ENV}"; then
    conda create -y -n "${LAMER_CONDA_ENV}" python=3.12
fi
conda activate "${LAMER_CONDA_ENV}"
python -m pip install --upgrade pip
python -m pip install -r "${LAMER_DIR}/requirements.txt"
# flash-attn compiles CUDA kernels against torch at build time,
# so it must be installed after torch (pulled in by vllm above).
python -m pip install flash-attn --no-build-isolation

echo "=== Bootstrapping language-table env (${LANGTABLE_CONDA_ENV}) ==="
if [[ "${LANGTABLE_CONDA_ENV}" == */* ]]; then
    if [ -e "${LANGTABLE_CONDA_ENV}" ] && [ ! -d "${LANGTABLE_CONDA_ENV}/conda-meta" ]; then
        echo "ERROR: ${LANGTABLE_CONDA_ENV} exists but is not a conda env."
        echo "Remove it or set LANGTABLE_CONDA_ENV to a different path."
        exit 1
    fi
    if [ ! -d "${LANGTABLE_CONDA_ENV}/conda-meta" ]; then
        conda create -y -p "${LANGTABLE_CONDA_ENV}" python=3.10
    fi
else
    if ! conda env list | awk '!/^#/ && NF {print $1}' | grep -qx "${LANGTABLE_CONDA_ENV}"; then
        conda create -y -n "${LANGTABLE_CONDA_ENV}" python=3.10
    fi
fi
conda activate "${LANGTABLE_CONDA_ENV}"
python -m pip install --upgrade pip
python -m pip install -r "${LANGTABLE_DIR}/requirements.txt"
python -m pip install --no-deps \
    git+https://github.com/google-research/scenic.git@ae21d9e884015aa7bc7cf1d489af53d16c249726

cat <<EOF

Bootstrap complete.

LaMer env:
  ${LAMER_CONDA_ENV}

language-table env:
  ${LANGTABLE_CONDA_ENV}

language-table python:
  ${LANGTABLE_CONDA_ENV}/bin/python

Next steps:
  1. cp .env.language_table.example .env.language_table
  2. cp .env.language_table.secrets.example .env.language_table.secrets
  3. Edit both files as needed
  4. scripts/submit_language_table.sh
EOF
