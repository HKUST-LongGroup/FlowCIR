#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Usage examples:
#   DATASET=cirr bash myScripts/extract_cir_feat.sh
#   DATASET=circo SPLITS="test" bash myScripts/extract_cir_feat.sh
# Paths default to config.py and can be overridden with environment variables.
DATASET="${DATASET:-cirr}"
SPLITS="${SPLITS:-test val}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
CLIP_MODEL="${CLIP_MODEL:-$(config_value CLIP_MODEL)}"

case "${DATASET}" in
  cirr) DATASET_PATH="${DATASET_PATH:-$(config_value CIRR_DIR)}" ;;
  circo) DATASET_PATH="${DATASET_PATH:-$(config_value CIRCO_DIR)}" ;;
  *) echo "DATASET must be 'cirr' or 'circo'" >&2; exit 2 ;;
esac

for SPLIT in ${SPLITS}; do
  case "${SPLIT}" in
    test) EXTRACTOR="${PROJECT_DIR}/extract_features/extracted_features_clip.py" ;;
    val) EXTRACTOR="${PROJECT_DIR}/extract_features/extracted_features_clip_val.py" ;;
    *) echo "SPLITS may contain only 'test' and 'val'" >&2; exit 2 ;;
  esac

  "${PYTHON}" "${EXTRACTOR}" \
    --dataset_path "${DATASET_PATH}" \
    --dataset_name "${DATASET}" \
    --save_query_path "$(config_value --feature "${DATASET}" "${SPLIT}" query)" \
    --save_gallery_path "$(config_value --feature "${DATASET}" "${SPLIT}" gallery)" \
    --clip_model "${CLIP_MODEL}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}"
done
