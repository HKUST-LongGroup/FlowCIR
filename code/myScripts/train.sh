#!/usr/bin/env bash
set -euo pipefail

# 加载项目目录、Python 命令及配置读取函数
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# 从 config.py 读取路径；外部同名环境变量优先
DATA_DIR="${DATA_DIR:-$(config_value HQEDIT_FEATURE_DIR)}"
SAVE_ROOT="${SAVE_ROOT:-$(config_value SAVE_ROOT)}"
LOG_DIR="${LOG_DIR:-$(config_value LOG_DIR)}"
GPU_ID="${GPU_ID:-2}"

# 训练参数
BATCH_SIZE=4096
LEARNING_RATE=0.00005
EPOCHS=500
NEG_K=512
INFONCE_START_EPOCH=5
INFONCE_WARMUP_EPOCHS=100
EPS=0.025
NUM_RES_BLOCKS=12
DIM="$(config_value DIM)"
SEED=0
LAMBDA_INFONCE=0.5
TOPK_START_EPOCH=15
EPS_PRE=0.05

# 实验名称、权重目录和日志文件
EXP_NAME="flow_cir_clipl14_lr${LEARNING_RATE}_b${BATCH_SIZE}_ep${EPOCHS}_blk${NUM_RES_BLOCKS}_elip_negK${NEG_K}_eps${EPS}_seed${SEED}_infonce${LAMBDA_INFONCE}_warm${INFONCE_WARMUP_EPOCHS}_infonce_start_epoch${INFONCE_START_EPOCH}"

EXP_DIR="${SAVE_ROOT}/${EXP_NAME}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

mkdir -p "${EXP_DIR}" "${LOG_DIR}"

echo "[run] GPU=${GPU_ID}"
echo "[run] data=${DATA_DIR}"
echo "[run] lambda_infoNce=${LAMBDA_INFONCE}"
echo "[run] output=${EXP_DIR}"
echo "[run] log=${LOG_FILE}"

# 启动一次训练
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
"${PYTHON}" "${PROJECT_DIR}/train/train.py" \
  --data_dir "${DATA_DIR}" \
  --n 0 \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LEARNING_RATE}" \
  --num_res_blocks "${NUM_RES_BLOCKS}" \
  --lambda_infoNce "${LAMBDA_INFONCE}" \
  --save_dir "${EXP_DIR}" \
  --dim "${DIM}" \
  --negK "${NEG_K}" \
  --seed "${SEED}" \
  --exclude_too_hard \
  --eps "${EPS}" \
  --infonce_start_epochs "${INFONCE_START_EPOCH}" \
  --infonce_warmup_epochs "${INFONCE_WARMUP_EPOCHS}" \
  --infonce_max_weight "${LAMBDA_INFONCE}" \
  --topk_start_epoch "${TOPK_START_EPOCH}" \
  --eps_pre "${EPS_PRE}" \
  2>&1 | tee "${LOG_FILE}"
