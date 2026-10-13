#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Set MODEL_DIR to the checkpoint directory produced by myScripts/train.sh.
MODEL_DIR="${MODEL_DIR:-}"
if [[ -z "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR must point to a trained checkpoint directory" >&2
  exit 2
fi

DATASET="cirr"
QUERY_FEATURES="${QUERY_FEATURES:-$(config_value --feature "${DATASET}" val query)}"
GALLERY_FEATURES="${GALLERY_FEATURES:-$(config_value --feature "${DATASET}" val gallery)}"
PREDICTION_DIR="${PREDICTION_DIR:-$(config_value FEATURE_ROOT)/${DATASET}/predictions}"
EPOCHS="${EPOCHS:-10 15 20 25 30}"
END_POINTS="${END_POINTS:-0.5 0.3 0.4 0.55 0.6}"
GPU_ID="${GPU_ID:-0}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-12}"
DIM="${DIM:-$(config_value DIM)}"
EXP_NAME="${EXP_NAME:-cirr_v1_val}"

mkdir -p "${PREDICTION_DIR}"

for EPOCH in ${EPOCHS}; do
  for END_POINT in ${END_POINTS}; do
    PREDICTION_PATH="${PREDICTION_DIR}/cirr_predictions_val_${EPOCH}_${END_POINT}.pth"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" "${PROJECT_DIR}/predict_fm_v2.py" \
      --data_dir "${QUERY_FEATURES}" \
      --model_dir "${MODEL_DIR}/model_${EPOCH}1.pth" \
      --save_dir "${PREDICTION_PATH}" \
      --steps 1 \
      --end_point "${END_POINT}" \
      --num_res_blocks "${NUM_RES_BLOCKS}" \
      --dim "${DIM}" \
      --val \
      --dataset "${DATASET}" \
      --model_version v1

    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" "${PROJECT_DIR}/validate.py" \
      --dataset_path "${PREDICTION_PATH}" \
      --dataset_index_path "${GALLERY_FEATURES}" \
      --retrieval_type fm_edited \
      --dataset "${DATASET}" \
      --exp_name "${EXP_NAME}"
  done
done
