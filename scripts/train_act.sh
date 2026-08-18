#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DP_ROOT="$PROJECT_ROOT/diffusion_policy-main"

DATASET_PATH="${DATASET_PATH:-data_local/glue_motion}"
ACT_EPOCHS="${ACT_EPOCHS:-50}"
ACT_BATCH_SIZE="${ACT_BATCH_SIZE:-16}"
ACT_NUM_WORKERS="${ACT_NUM_WORKERS:-0}"
ACT_CHUNK_SIZE="${ACT_CHUNK_SIZE:-50}"
ACT_TARGET_FREQUENCY="${ACT_TARGET_FREQUENCY:-20}"
ACT_STATE_MODE="${ACT_STATE_MODE:-eef}"
ACT_CACHE_DIR="${ACT_CACHE_DIR:-}"
ACT_CHECKPOINT_EVERY="${ACT_CHECKPOINT_EVERY:-10}"
ACT_DEVICE="${ACT_DEVICE:-auto}"
ACT_VAL_RATIO="${ACT_VAL_RATIO:-0.1}"
ACT_SEED="${ACT_SEED:-42}"
ACT_IMAGE_WIDTH="${ACT_IMAGE_WIDTH:-320}"
ACT_IMAGE_HEIGHT="${ACT_IMAGE_HEIGHT:-240}"
ACT_LR="${ACT_LR:-1e-5}"
ACT_LR_BACKBONE="${ACT_LR_BACKBONE:-1e-5}"
ACT_KL_WEIGHT="${ACT_KL_WEIGHT:-10}"
ACT_HIDDEN_DIM="${ACT_HIDDEN_DIM:-256}"
ACT_DIM_FEEDFORWARD="${ACT_DIM_FEEDFORWARD:-2048}"
ACT_ENC_LAYERS="${ACT_ENC_LAYERS:-4}"
ACT_DEC_LAYERS="${ACT_DEC_LAYERS:-7}"
ACT_NHEADS="${ACT_NHEADS:-8}"
ACT_BACKBONE="${ACT_BACKBONE:-resnet18}"
ACT_PRETRAINED_BACKBONE="${ACT_PRETRAINED_BACKBONE:-0}"
ACT_PIN_MEMORY="${ACT_PIN_MEMORY:-0}"
ACT_PERSISTENT_WORKERS="${ACT_PERSISTENT_WORKERS:-0}"
ACT_USE_CACHE="${ACT_USE_CACHE:-1}"

RUN_NAME="${RUN_NAME:-$(basename "$DATASET_PATH")_act_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/logs/$RUN_NAME"
ACT_CKPT_DIR="${ACT_CKPT_DIR:-$PROJECT_ROOT/act_outputs/${RUN_NAME}}"

mkdir -p "$LOG_DIR" "$ACT_CKPT_DIR"

set +u
source "$PROJECT_ROOT/activate_arx5_env.sh"
set -u

if [[ "$DATASET_PATH" = /* ]]; then
  DATASET_ABS="$DATASET_PATH"
else
  DATASET_ABS="$DP_ROOT/$DATASET_PATH"
fi

echo "========== ACT training started =========="
echo "Run name: $RUN_NAME"
echo "Dataset: $DATASET_ABS"
echo "Logs: $LOG_DIR"
echo "ACT output: $ACT_CKPT_DIR"
echo "Epochs: $ACT_EPOCHS"
echo "Batch size: $ACT_BATCH_SIZE"
echo "Num workers: $ACT_NUM_WORKERS"
echo "Chunk size: $ACT_CHUNK_SIZE"
echo "Target frequency: $ACT_TARGET_FREQUENCY"
echo "State mode: $ACT_STATE_MODE"
echo "Cache dir: ${ACT_CACHE_DIR:-<dataset default>}"
echo "Device: $ACT_DEVICE"
echo "Use cache: $ACT_USE_CACHE"
echo

cd "$PROJECT_ROOT"

EXTRA_ARGS=()
if [[ "$ACT_PRETRAINED_BACKBONE" = "1" ]]; then
  EXTRA_ARGS+=(--pretrained-backbone)
fi
if [[ "$ACT_PIN_MEMORY" = "1" ]]; then
  EXTRA_ARGS+=(--pin-memory)
fi
if [[ "$ACT_PERSISTENT_WORKERS" = "1" ]]; then
  EXTRA_ARGS+=(--persistent-workers)
fi
if [[ "$ACT_USE_CACHE" != "1" ]]; then
  EXTRA_ARGS+=(--no-cache)
fi
if [[ -n "$ACT_CACHE_DIR" ]]; then
  EXTRA_ARGS+=(--cache-dir "$ACT_CACHE_DIR")
fi

python -m arx5_act.train_act \
  --dataset-path "$DATASET_ABS" \
  --ckpt-dir "$ACT_CKPT_DIR" \
  --num-epochs "$ACT_EPOCHS" \
  --batch-size "$ACT_BATCH_SIZE" \
  --num-workers "$ACT_NUM_WORKERS" \
  --chunk-size "$ACT_CHUNK_SIZE" \
  --target-frequency "$ACT_TARGET_FREQUENCY" \
  --state-mode "$ACT_STATE_MODE" \
  --val-ratio "$ACT_VAL_RATIO" \
  --seed "$ACT_SEED" \
  --image-width "$ACT_IMAGE_WIDTH" \
  --image-height "$ACT_IMAGE_HEIGHT" \
  --lr "$ACT_LR" \
  --lr-backbone "$ACT_LR_BACKBONE" \
  --kl-weight "$ACT_KL_WEIGHT" \
  --hidden-dim "$ACT_HIDDEN_DIM" \
  --dim-feedforward "$ACT_DIM_FEEDFORWARD" \
  --enc-layers "$ACT_ENC_LAYERS" \
  --dec-layers "$ACT_DEC_LAYERS" \
  --nheads "$ACT_NHEADS" \
  --backbone "$ACT_BACKBONE" \
  --checkpoint-every "$ACT_CHECKPOINT_EVERY" \
  --device "$ACT_DEVICE" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/act_train.log"

echo
echo "========== ACT training finished =========="
echo "ACT log: $LOG_DIR/act_train.log"
echo "ACT ckpt: $ACT_CKPT_DIR"
