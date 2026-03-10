#!/usr/bin/env bash
# teardown_vms.sh — Deallocate or delete Azure VMs to stop billing.
#
# Run this from your LOCAL machine when done training.
#
# Usage:
#   scripts/azure/teardown_vms.sh              # deallocate (keep disks, stop compute billing)
#   scripts/azure/teardown_vms.sh --delete     # delete VMs entirely (disks auto-cleaned)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAMER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${BASE_LAMER_DIR}/.env.azure" ]; then
    # shellcheck disable=SC1091
    source "${BASE_LAMER_DIR}/.env.azure"
fi

RG="${AZURE_RESOURCE_GROUP:-myresourcegroup}"
VM_TRAIN_0="${AZURE_VM_TRAIN_0:-vm-train-0}"
VM_TRAIN_1="${AZURE_VM_TRAIN_1:-vm-train-1}"
VM_ENV="${AZURE_VM_ENV:-vm-env}"

ACTION="deallocate"
if [ "${1:-}" = "--delete" ]; then
    ACTION="delete"
fi

echo "=== ${ACTION^} Azure VMs ==="

if [ "${ACTION}" = "delete" ]; then
    # Delete VMs in parallel, then wait for all to finish before cleaning up
    # networking resources (NICs auto-delete with the VM and must be gone first).
    for vm in "${VM_TRAIN_0}" "${VM_TRAIN_1}" "${VM_ENV}"; do
        echo "  Deleting ${vm}..."
        az vm delete -g "${RG}" -n "${vm}" --yes --no-wait -o none 2>/dev/null || true
    done
    echo "  Waiting for VM deletions to complete..."
    for vm in "${VM_TRAIN_0}" "${VM_TRAIN_1}" "${VM_ENV}"; do
        az vm wait -g "${RG}" -n "${vm}" --deleted 2>/dev/null || true
    done
else
    for vm in "${VM_TRAIN_0}" "${VM_TRAIN_1}" "${VM_ENV}"; do
        echo "  Deallocating ${vm}..."
        az vm deallocate -g "${RG}" -n "${vm}" --no-wait -o none 2>/dev/null || true
    done
fi

if [ "${ACTION}" = "delete" ]; then
    # az vm delete only removes the VM + resources with delete-on-VM-removal
    # (disks, NICs). NSGs, public IPs, and VNets are left behind. Clean them up.
    echo ""
    echo "Cleaning up leftover networking resources..."

    delete_if_exists() {
        local name=$1 type=$2
        if az resource show -g "${RG}" -n "${name}" --resource-type "${type}" &>/dev/null; then
            echo "  Deleting ${name}..."
            az resource delete -g "${RG}" -n "${name}" --resource-type "${type}" -o none 2>/dev/null || true
        fi
    }

    for vm in "${VM_TRAIN_0}" "${VM_TRAIN_1}" "${VM_ENV}"; do
        delete_if_exists "${vm}PublicIP" "Microsoft.Network/publicIPAddresses"
        delete_if_exists "${vm}NSG"      "Microsoft.Network/networkSecurityGroups"
    done

    # Delete the VNet (created by the first az vm create, shared by all VMs)
    delete_if_exists "${VM_TRAIN_0}VNET" "Microsoft.Network/virtualNetworks"

    echo ""
    echo "All VM resources deleted."
    echo ""
    echo "Verify cleanup:"
    echo "  az resource list -g ${RG} -o table"
else
    echo ""
    echo "VMs deallocating (compute billing will stop once complete)."
    echo "Disk storage charges continue (~\$5-10/mo per VM)."
    echo ""
    echo "To restart later:"
    echo "  az vm start -g ${RG} -n ${VM_TRAIN_0}"
    echo "  az vm start -g ${RG} -n ${VM_TRAIN_1}"
    echo "  az vm start -g ${RG} -n ${VM_ENV}"
    echo ""
    echo "Then get new IPs with: scripts/azure/provision_vms.sh --info"
fi
