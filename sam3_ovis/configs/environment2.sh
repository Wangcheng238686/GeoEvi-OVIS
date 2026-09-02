#!/usr/bin/env bash
# HOST-SPECIFIC config for host "lthpc" (RTX 3090 x4). Do NOT activate on
# other machines. Same mechanism as geoevi_ovis/configs/environment2.sh:
# inert unless configs/environment.local.sh symlinks here (the symlink is
# gitignored and never leaves this machine).
#
# Values verified against this machine on 2026-09-02:
#   - /data/wangcheng/envs/sam3 (python 3.12.14; torch 2.10.0+cu128, see
#     scripts/README.md for the install log)
#   - /data/wangcheng/dataset/SCORE (SCORE HF HuangShiqi008/SCORE mirror)
#   - /data/wangcheng/pretrained-models/sam3 (facebook/sam3 HF snapshot)
# Values already exported by the outer shell keep precedence.

_ENV_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MAINLINE_ROOT="$(cd "${_ENV_CONFIG_DIR}/.." && pwd)"
_REPOSITORY_ROOT="$(cd "${_MAINLINE_ROOT}/.." && pwd)"

# Project/runtime paths.
export SAM3_OVIS_MAINLINE_ROOT="${SAM3_OVIS_MAINLINE_ROOT:-${_MAINLINE_ROOT}}"
export SAM3_OVIS_REPOSITORY_ROOT="${SAM3_OVIS_REPOSITORY_ROOT:-${_REPOSITORY_ROOT}}"
export PYTHON="${PYTHON:-/data/wangcheng/envs/sam3/bin/python}"
export SAM3_REPO="${SAM3_REPO:-${SAM3_OVIS_REPOSITORY_ROOT}/sam3}"
export SAM3_HUB_DIR="${SAM3_HUB_DIR:-/data/wangcheng/pretrained-models/sam3}"
export SAM3_CKPT="${SAM3_CKPT:-${SAM3_HUB_DIR}/sam3.pt}"
export SAM3_OVIS_DATA_ROOT="${SAM3_OVIS_DATA_ROOT:-/data/wangcheng/dataset/SCORE}"

# Writable output and temporary roots.
export SAM3_OVIS_LOG_ROOT="${SAM3_OVIS_LOG_ROOT:-${SAM3_OVIS_MAINLINE_ROOT}/logs}"
export SAM3_OVIS_TMP_ROOT="${SAM3_OVIS_TMP_ROOT:-/tmp/sam3_ovis}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SAM3_OVIS_TMP_ROOT}/mpl}"

# GPU defaults. Other projects hold GPU 1-2 during P-1; inference defaults to
# GPU 0. Override in the launching shell when that changes.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

unset _ENV_CONFIG_DIR _MAINLINE_ROOT _REPOSITORY_ROOT
