#!/usr/bin/env bash
# Shared loader for the sam3_ovis mainline. Source this file. Prefers a
# machine-local configs/environment.local.sh when present and otherwise falls
# back to configs/environment.sh. To force a specific file, set
# SAM3_OVIS_ENV_FILE. Mechanism mirrors geoevi_ovis/scripts/load_environment.sh.

_ENV_LOADER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CONFIGS_DIR="$(cd "${_ENV_LOADER_DIR}/../configs" && pwd)"
if [[ -z "${SAM3_OVIS_ENV_FILE:-}" ]]; then
  if [[ -r "${_CONFIGS_DIR}/environment.local.sh" ]]; then
    SAM3_OVIS_ENV_FILE="${_CONFIGS_DIR}/environment.local.sh"
  else
    SAM3_OVIS_ENV_FILE="${_CONFIGS_DIR}/environment.sh"
  fi
fi

if [[ ! -r "${SAM3_OVIS_ENV_FILE}" ]]; then
  echo "sam3_ovis environment config is not readable: ${SAM3_OVIS_ENV_FILE}" >&2
  return 2
fi

# shellcheck disable=SC1090
source "${SAM3_OVIS_ENV_FILE}"
SAM3_OVIS_ENV_FILE="$(cd "$(dirname "${SAM3_OVIS_ENV_FILE}")" && pwd)/$(basename "${SAM3_OVIS_ENV_FILE}")"
export SAM3_OVIS_ENV_FILE

_ENV_REQUIRED=(
  PYTHON SAM3_REPO SAM3_CKPT SAM3_HUB_DIR SAM3_OVIS_DATA_ROOT
  SAM3_OVIS_LOG_ROOT SAM3_OVIS_TMP_ROOT CUDA_VISIBLE_DEVICES
)
for _ENV_NAME in "${_ENV_REQUIRED[@]}"; do
  if [[ -z "${!_ENV_NAME:-}" ]]; then
    echo "Required setting ${_ENV_NAME} is empty in ${SAM3_OVIS_ENV_FILE}" >&2
    return 2
  fi
done

unset _ENV_LOADER_DIR _CONFIGS_DIR _ENV_REQUIRED _ENV_NAME
