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

for vm in "${VM_TRAIN_0}" "${VM_TRAIN_1}" "${VM_ENV}"; do
    echo "  ${ACTION}: ${vm}..."
    if [ "${ACTION}" = "delete" ]; then
        az vm delete -g "${RG}" -n "${vm}" --yes --no-wait -o none 2>/dev/null || true
    else
        az vm deallocate -g "${RG}" -n "${vm}" --no-wait -o none 2>/dev/null || true
    fi
done

echo ""
if [ "${ACTION}" = "deallocate" ]; then
    echo "VMs deallocating (compute billing will stop once complete)."
    echo "Disk storage charges continue (~\$5-10/mo per VM)."
    echo ""
    echo "To restart later:"
    echo "  az vm start -g ${RG} -n ${VM_TRAIN_0}"
    echo "  az vm start -g ${RG} -n ${VM_TRAIN_1}"
    echo "  az vm start -g ${RG} -n ${VM_ENV}"
    echo ""
    echo "Then get new IPs with: scripts/azure/provision_vms.sh --info"
else
    echo "VMs deleting. Attached resources (disks, NICs) will auto-delete."
    echo ""
    echo "Verify cleanup:"
    echo "  az resource list -g ${RG} -o table"
fi
