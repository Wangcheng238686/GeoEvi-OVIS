#!/usr/bin/env bash
# Shared defaults for the sam3_ovis mainline (design-v2 / P-1).
#
# Machine-specific overrides live in environment2.sh (host lthpc) and are
# activated via the configs/environment.local.sh symlink; scripts/load_environment.sh
# prefers the local file when present. Values already exported by the outer
# shell keep precedence, so one-off overrides remain possible without editing
# this file.

_ENV_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MAINLINE_ROOT="$(cd "${_ENV_CONFIG_DIR}/.." && pwd)"
_REPOSITORY_ROOT="$(cd "${_MAINLINE_ROOT}/.." && pwd)"

# Project/runtime paths (contract: machine paths concentrate here; SAM3_REPO /
# SAM3_CKPT mirror the SAM2_REPO / SAM2_CKPT mechanism of geoevi_ovis).
export SAM3_OVIS_MAINLINE_ROOT="${SAM3_OVIS_MAINLINE_ROOT:-${_MAINLINE_ROOT}}"
export SAM3_OVIS_REPOSITORY_ROOT="${SAM3_OVIS_REPOSITORY_ROOT:-${_REPOSITORY_ROOT}}"
export PYTHON="${PYTHON:-/data/wangcheng/envs/sam3/bin/python}"
export SAM3_REPO="${SAM3_REPO:-${SAM3_OVIS_REPOSITORY_ROOT}/sam3}"
# Full HF snapshot of facebook/sam3 (config.json + tokenizer files + weights);
# SAM3_CKPT points at the dense checkpoint inside it.
export SAM3_HUB_DIR="${SAM3_HUB_DIR:-/data/wangcheng/pretrained-models/sam3}"
export SAM3_CKPT="${SAM3_CKPT:-${SAM3_HUB_DIR}/sam3.pt}"
# Registered data root: every dataset lands as a COCO-instance subtree below
# this directory (see data/README.md for the per-dataset registry).
export SAM3_OVIS_DATA_ROOT="${SAM3_OVIS_DATA_ROOT:-/data/wangcheng/dataset/SCORE}"

# Writable output and temporary roots.
export SAM3_OVIS_LOG_ROOT="${SAM3_OVIS_LOG_ROOT:-${SAM3_OVIS_MAINLINE_ROOT}/logs}"
export SAM3_OVIS_TMP_ROOT="${SAM3_OVIS_TMP_ROOT:-/tmp/sam3_ovis}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SAM3_OVIS_TMP_ROOT}/mpl}"

# Current machine execution defaults. Override in the launching shell when
# GPUs are occupied (this host: RTX 3090 x4, 24G each; other jobs routinely
# hold GPU 1-2).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

unset _ENV_CONFIG_DIR _MAINLINE_ROOT _REPOSITORY_ROOT
