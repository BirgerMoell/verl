#!/usr/bin/env bash
# Ray >=2.55 rejects ROCR_VISIBLE_DEVICES on AMD. Keep Slurm's exact GPU mask
# under the HIP variable rather than exposing unallocated devices.
set -euo pipefail
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi
unset ROCR_VISIBLE_DEVICES
exec "$@"
