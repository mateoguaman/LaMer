#!/usr/bin/env bash
# provision_vms.sh — Create the 3 Azure VMs for multi-node LaMer training.
#
# Run this from your LOCAL machine (not on Azure).
#
# Usage:
#   scripts/azure/provision_vms.sh          # create all 3 VMs
#   scripts/azure/provision_vms.sh --info   # just print IPs of existing VMs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAMER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_FILE="${BASE_LAMER_DIR}/.env.azure"
if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: .env.azure not found at ${ENV_FILE}"
    echo "Copy .env.azure.example to .env.azure and fill in your values:"
    echo "  cp ${BASE_LAMER_DIR}/.env.azure.example ${ENV_FILE}"
    exit 1
fi
# shellcheck disable=SC1091
source "${ENV_FILE}"

RG="${AZURE_RESOURCE_GROUP:?'Set AZURE_RESOURCE_GROUP in .env.azure'}"
LOC="${AZURE_LOCATION:?'Set AZURE_LOCATION in .env.azure'}"
SSH_KEY="${AZURE_SSH_KEY:?'Set AZURE_SSH_KEY in .env.azure'}"
ADMIN="${AZURE_ADMIN_USER:?'Set AZURE_ADMIN_USER in .env.azure'}"
VM_TRAIN_0="${AZURE_VM_TRAIN_0:?'Set AZURE_VM_TRAIN_0 in .env.azure'}"
VM_TRAIN_1="${AZURE_VM_TRAIN_1:?'Set AZURE_VM_TRAIN_1 in .env.azure'}"
VM_ENV="${AZURE_VM_ENV:?'Set AZURE_VM_ENV in .env.azure'}"
TRAIN_SIZE="${AZURE_TRAIN_VM_SIZE:?'Set AZURE_TRAIN_VM_SIZE in .env.azure'}"
ENV_SIZE="${AZURE_ENV_VM_SIZE:?'Set AZURE_ENV_VM_SIZE in .env.azure'}"

# Push .env.azure (and .env.language_table.secrets if present) to each VM.
# Called after VM creation so that setup_vm.sh can find them without manual copying.
push_env_files() {
    local vm=$1
    local public_ip
    public_ip=$(az vm show -g "${RG}" -n "${vm}" -d --query publicIps -o tsv 2>/dev/null || true)
    if [ -z "${public_ip}" ]; then
        echo "  WARNING: could not get public IP for ${vm}, skipping env file push."
        return 0
    fi

    local ssh_opts="-i ${SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=10"
    local remote="${ADMIN}@${public_ip}"
    local remote_dir="/home/${ADMIN}/LaMer"

    # Wait for SSH to become available (VM may still be booting)
    echo "  Waiting for SSH on ${vm} (${public_ip})..."
    for attempt in $(seq 1 24); do
        if ssh ${ssh_opts} "${remote}" "mkdir -p ${remote_dir}" 2>/dev/null; then
            break
        fi
        if [ "${attempt}" -eq 24 ]; then
            echo "  WARNING: SSH not ready on ${vm} after 2 min, skipping env file push."
            return 0
        fi
        sleep 5
    done

    echo "  Pushing .env.azure to ${vm}:${remote_dir}/.env.azure"
    scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no \
        "${ENV_FILE}" "${remote}:${remote_dir}/.env.azure"

    local secrets_file="${BASE_LAMER_DIR}/.env.language_table.secrets"
    if [ -f "${secrets_file}" ]; then
        echo "  Pushing .env.language_table.secrets to ${vm}"
        scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no \
            "${secrets_file}" "${remote}:${remote_dir}/.env.language_table.secrets"
    fi
}

print_ips() {
    echo ""
    echo "=== VM IP Addresses ==="
    for vm in "${VM_TRAIN_0}" "${VM_TRAIN_1}" "${VM_ENV}"; do
        public_ip=$(az vm show -g "${RG}" -n "${vm}" -d --query publicIps -o tsv 2>/dev/null || echo "<not found>")
        private_ip=$(az vm show -g "${RG}" -n "${vm}" -d --query privateIps -o tsv 2>/dev/null || echo "<not found>")
        echo "  ${vm}: public=${public_ip}  private=${private_ip}"
    done
    echo ""
    echo "SSH commands:"
    for vm in "${VM_TRAIN_0}" "${VM_TRAIN_1}" "${VM_ENV}"; do
        public_ip=$(az vm show -g "${RG}" -n "${vm}" -d --query publicIps -o tsv 2>/dev/null || echo "<IP>")
        echo "  ssh -i ${SSH_KEY} ${ADMIN}@${public_ip}  # ${vm}"
    done
}

if [ "${1:-}" = "--info" ]; then
    print_ips
    exit 0
fi

echo "=== Provisioning Azure VMs ==="
echo "  Resource Group: ${RG}"
echo "  Location:       ${LOC}"
echo "  Training VMs:   ${TRAIN_SIZE} (2x H100 each)"
echo "  Env Server VM:  ${ENV_SIZE} (1x A100)"
echo ""

# Ensure resource group exists
az group create --name "${RG}" --location "${LOC}" -o none 2>/dev/null || true

IMAGE="microsoft-dsvm:ubuntu-hpc:2204:latest"
COMMON_FLAGS=(
    --resource-group "${RG}"
    --image "${IMAGE}"
    --admin-username "${ADMIN}"
    --ssh-key-values "${SSH_KEY}.pub"
    --public-ip-sku Standard
    --os-disk-delete-option Delete
    --data-disk-delete-option Delete
    --nic-delete-option Delete
)

create_vm() {
    local name=$1
    local size=$2
    local disk_gb=${3:-256}

    if az vm show -g "${RG}" -n "${name}" &>/dev/null; then
        echo "  ${name} already exists, skipping creation."
        return 0
    fi

    echo "Creating ${name} (${size}, ${disk_gb}GB disk)..."
    az vm create \
        "${COMMON_FLAGS[@]}" \
        --name "${name}" \
        --size "${size}" \
        --os-disk-size-gb "${disk_gb}" \
        -o none
    echo "  ${name} created."
}

# Create all 3 VMs. Training VMs get larger disks for checkpoints.
create_vm "${VM_TRAIN_0}" "${TRAIN_SIZE}" 512
create_vm "${VM_TRAIN_1}" "${TRAIN_SIZE}" 512
create_vm "${VM_ENV}"     "${ENV_SIZE}"   256

# Azure VMs in the same VNet can reach each other on all ports by default,
# so no additional NSG rules are needed for intra-VNet traffic.

# Push env files so setup_vm.sh can run without manual scp.
echo ""
echo "=== Pushing env files to VMs ==="
for vm in "${VM_TRAIN_0}" "${VM_TRAIN_1}" "${VM_ENV}"; do
    push_env_files "${vm}"
done

print_ips

echo ""
echo "Next steps:"
echo "  1. SSH into each VM and run: bash scripts/azure/setup_vm.sh"
echo "  2. On vm-env:     bash scripts/azure/start_env_servers.sh"
echo "  3. On vm-train-0: bash scripts/azure/start_training.sh <vm-env-private-ip> <vm-train-1-private-ip>"
