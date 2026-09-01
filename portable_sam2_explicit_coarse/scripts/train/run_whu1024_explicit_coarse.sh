#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=load_environment.sh
source "${PROJECT_ROOT}/scripts/load_environment.sh"

export SAM2_MODEL_SIZE=base_plus
export MAX_EPOCHS="${MAX_EPOCHS:-80}"
export EXPLICIT_PROMPT_MODE="${EXPLICIT_PROMPT_MODE:-points_box_dense}"
export P2_BOUNDARY_REFINER_ENABLED="${P2_BOUNDARY_REFINER_ENABLED:-0}"
export TRAIN_SUBSET_RATIO="${TRAIN_SUBSET_RATIO:-0.2}"
export VAL_SUBSET_RATIO="${VAL_SUBSET_RATIO:-1.0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PORTABLE_SAM2_CHECKPOINT_ROOT}/${EXPLICIT_PROMPT_MODE}_p2br${P2_BOUNDARY_REFINER_ENABLED}}"
export RUN_TAG="${RUN_TAG:-explicit_${EXPLICIT_PROMPT_MODE}_p2br${P2_BOUNDARY_REFINER_ENABLED}}"

# Image-embedding resolution is fixed to the official stride-16 contract.
case "${EXPLICIT_PROMPT_MODE}:${P2_BOUNDARY_REFINER_ENABLED}" in
  points_box:0)
    WRAPPER="scripts/train/ablations/r1_c3_pafpn_coarse_points_box_emb64.sh"
    ;;
  points_box_dense:0)
    WRAPPER="scripts/train/ablations/r1_c4_pafpn_coarse_points_box_dense_emb64.sh"
    ;;
  points_box_dense:1)
    WRAPPER="scripts/train/ablations/c5v2_pafpn_coarse_p2_boundary_refiner_emb64.sh"
    ;;
  *)
    echo "Unsupported prompt/refiner combination: ${EXPLICIT_PROMPT_MODE}/${P2_BOUNDARY_REFINER_ENABLED}" >&2
    exit 2
    ;;
esac

exec bash "${WRAPPER}" "$@"
