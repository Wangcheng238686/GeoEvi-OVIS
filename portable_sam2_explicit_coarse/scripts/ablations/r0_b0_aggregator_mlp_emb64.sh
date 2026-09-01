#!/usr/bin/env bash
set -Eeuo pipefail
export ABLATION_ID="r0_b0_aggregator_mlp_emb64"
export RUN_TAG="${RUN_TAG:-r0_b0_aggregator_mlp_emb64}"
export NECK_TYPE="aggregator"
export PROMPT_ROUTE="mlp"
export P2_BOUNDARY_REFINER_ENABLED=0
export FINAL_MASK_COORDINATE_MODE="roi_local"
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
