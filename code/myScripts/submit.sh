#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Set MODEL_DIR to the checkpoint directory produced by myScripts/train.sh.
MODEL_DIR="${MODEL_DIR:-}"
if [[ -z "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR must point to a trained checkpoint directory" >&2
  exit 2
fi

DATASET="${DATASET:-cirr}"
case "${DATASET}" in
  cirr) DEFAULT_MODEL_VERSION="v1_neg" ;;
  circo) DEFAULT_MODEL_VERSION="v1" ;;
  *) echo "DATASET must be 'cirr' or 'circo'" >&2; exit 2 ;;
esac

EPOCH="${EPOCH:-15}"
END_POINT="${END_POINT:-0.55}"
STEPS="${STEPS:-1}"
SPACEVLM_T="${SPACEVLM_T:-0.92}"
GPU_ID="${GPU_ID:-0}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-12}"
DIM="${DIM:-$(config_value DIM)}"
MODEL_VERSION="${MODEL_VERSION:-${DEFAULT_MODEL_VERSION}}"
QUERY_FEATURES="${QUERY_FEATURES:-$(config_value --feature "${DATASET}" test query)}"
GALLERY_FEATURES="${GALLERY_FEATURES:-$(config_value --feature "${DATASET}" test gallery)}"
PREDICTION_DIR="${PREDICTION_DIR:-$(config_value FEATURE_ROOT)/${DATASET}/predictions}"
PREDICTION_PATH="${PREDICTION_PATH:-${PREDICTION_DIR}/${DATASET}_predictions_${EPOCH}_${END_POINT}.pth}"
SUBMISSION_NAME="${SUBMISSION_NAME:-flowcir_${DATASET}}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" "${PROJECT_DIR}/predict_fm_v2_multiposnet.py" \
  --data_dir "${QUERY_FEATURES}" \
  --model_dir "${MODEL_DIR}/model_${EPOCH}1.pth" \
  --save_dir "${PREDICTION_PATH}" \
  --steps "${STEPS}" \
  --end_point "${END_POINT}" \
  --num_res_blocks "${NUM_RES_BLOCKS}" \
  --dim "${DIM}" \
  --dataset "${DATASET}" \
  --model_version "${MODEL_VERSION}" \
  --t "${SPACEVLM_T}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" "${PROJECT_DIR}/generate_submission_vlm2vec.py" \
  --dataset_path "${PREDICTION_PATH}" \
  --dataset_index_path "${GALLERY_FEATURES}" \
  --retrieval_type fm_edited \
  --dataset "${DATASET}" \
  --submission_name "${SUBMISSION_NAME}"
