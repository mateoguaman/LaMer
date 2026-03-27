#!/bin/bash
# Build the LaMer + language-table Docker image and push to Amazon ECR.
#
# Usage:
#   bash scripts/sagemaker/build_and_push.sh
#
# Requires: docker, aws CLI (authenticated), and the language-table repo
# adjacent to this repo (or set LANGTABLE_DIR).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LANGTABLE_DIR="${LANGTABLE_DIR:-$(cd "${LAMER_DIR}/../language-table" && pwd)}"

# Load config
if [ -f "${LAMER_DIR}/.env.sagemaker" ]; then
    # shellcheck disable=SC1091
    source "${LAMER_DIR}/.env.sagemaker"
fi

AWS_REGION="${AWS_REGION:-us-west-2}"
ECR_REPO_NAME="${ECR_REPO_NAME:-lamer-langtable}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "=== Build & Push to ECR ==="
echo "  LAMER_DIR:     ${LAMER_DIR}"
echo "  LANGTABLE_DIR: ${LANGTABLE_DIR}"
echo "  AWS_REGION:    ${AWS_REGION}"
echo "  ECR_REPO:      ${ECR_REPO_NAME}"
echo "  IMAGE_TAG:     ${IMAGE_TAG}"
echo ""

# Verify language-table exists
if [ ! -d "${LANGTABLE_DIR}" ]; then
    echo "ERROR: language-table not found at ${LANGTABLE_DIR}"
    echo "Clone it or set LANGTABLE_DIR."
    exit 1
fi

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_IMAGE="${ECR_URI}/${ECR_REPO_NAME}:${IMAGE_TAG}"

echo "Full image URI: ${FULL_IMAGE}"
echo ""

# Create ECR repo if it doesn't exist
aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" 2>/dev/null \
    || aws ecr create-repository --repository-name "${ECR_REPO_NAME}" --region "${AWS_REGION}"

# Authenticate Docker with ECR
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_URI}"

# Copy language-table into build context (Dockerfile expects it at ./language-table)
TEMP_LT="${LAMER_DIR}/language-table"
if [ ! -d "${TEMP_LT}" ]; then
    echo "Copying language-table into build context..."
    cp -r "${LANGTABLE_DIR}" "${TEMP_LT}"
    CLEANUP_LT=true
else
    CLEANUP_LT=false
fi

cleanup() {
    if [ "${CLEANUP_LT}" = true ] && [ -d "${TEMP_LT}" ]; then
        echo "Cleaning up temporary language-table copy..."
        rm -rf "${TEMP_LT}"
    fi
}
trap cleanup EXIT

# Build
echo ""
echo "=== Building Docker image ==="
docker build \
    -t "${ECR_REPO_NAME}:${IMAGE_TAG}" \
    -f "${LAMER_DIR}/scripts/sagemaker/Dockerfile" \
    "${LAMER_DIR}"

# Tag and push
echo ""
echo "=== Pushing to ECR ==="
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${FULL_IMAGE}"
docker push "${FULL_IMAGE}"

echo ""
echo "=== Done ==="
echo "Image pushed: ${FULL_IMAGE}"
echo ""
echo "Use this URI in launch.py or .env.sagemaker:"
echo "  SAGEMAKER_IMAGE_URI=${FULL_IMAGE}"
