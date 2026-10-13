#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
CUDA_VISIBLE_DEVICES=0 python "${PROJECT_DIR}/extract_features/hqEdit_extractFeat_clip+inverse.py" \
  --data_dir "${DATA_DIR:-$(config_value HQEDIT_DIR)}" \
  --out_dir "${OUT_DIR:-$(config_value HQEDIT_FEATURE_DIR)}" \
  --batch_size 1024 \
  --clip_model_name "$(config_value CLIP_MODEL)" "$@"
