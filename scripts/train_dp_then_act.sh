#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DP_ROOT="$PROJECT_ROOT/diffusion_policy-main"

DATASET_PATH="${DATASET_PATH:-data_local/glue_motion}"
TARGET_FREQUENCY="${TARGET_FREQUENCY:-20}"

DP_EPOCHS="${DP_EPOCHS:-200}"
DP_BATCH_SIZE="${DP_BATCH_SIZE:-16}"
DP_NUM_WORKERS="${DP_NUM_WORKERS:-4}"
DP_CHECKPOINT_EVERY="${DP_CHECKPOINT_EVERY:-25}"

ACT_EPOCHS="${ACT_EPOCHS:-50}"
ACT_BATCH_SIZE="${ACT_BATCH_SIZE:-8}"
ACT_NUM_WORKERS="${ACT_NUM_WORKERS:-0}"
ACT_CHUNK_SIZE="${ACT_CHUNK_SIZE:-50}"
ACT_TARGET_FREQUENCY="${ACT_TARGET_FREQUENCY:-$TARGET_FREQUENCY}"
ACT_CHECKPOINT_EVERY="${ACT_CHECKPOINT_EVERY:-10}"
ACT_DEVICE="${ACT_DEVICE:-auto}"
SKIP_DP="${SKIP_DP:-0}"
SKIP_ACT="${SKIP_ACT:-0}"

RUN_NAME="${RUN_NAME:-$(basename "$DATASET_PATH")_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/logs/$RUN_NAME"
ACT_CKPT_DIR="$PROJECT_ROOT/act_outputs/${RUN_NAME}_act"

mkdir -p "$LOG_DIR" "$ACT_CKPT_DIR"

set +u
source "$PROJECT_ROOT/activate_arx5_env.sh"
set -u

echo "Run name: $RUN_NAME"
echo "Dataset: $DATASET_PATH"
echo "Logs: $LOG_DIR"
echo "ACT output: $ACT_CKPT_DIR"

if [[ "$SKIP_DP" != "1" ]]; then
  echo
  echo "========== DP training started =========="
  cd "$DP_ROOT"
  python train.py \
    --config-name=train_diffusion_unet_arx5_hybrid_workspace \
    task.dataset_path="$DATASET_PATH" \
    task.dataset.target_frequency="$TARGET_FREQUENCY" \
    training.num_epochs="$DP_EPOCHS" \
    dataloader.batch_size="$DP_BATCH_SIZE" \
    val_dataloader.batch_size="$DP_BATCH_SIZE" \
    dataloader.num_workers="$DP_NUM_WORKERS" \
    val_dataloader.num_workers="$DP_NUM_WORKERS" \
    training.checkpoint_every="$DP_CHECKPOINT_EVERY" \
    logging.mode=offline \
    2>&1 | tee "$LOG_DIR/dp_train.log"
  echo
  echo "========== DP training finished =========="
else
  echo
  echo "========== DP training skipped =========="
fi

if [[ "$SKIP_ACT" != "1" ]]; then
  echo "========== ACT training started =========="
  cd "$PROJECT_ROOT"
  python -m arx5_act.train_act \
    --dataset-path "$DP_ROOT/$DATASET_PATH" \
    --ckpt-dir "$ACT_CKPT_DIR" \
    --num-epochs "$ACT_EPOCHS" \
    --batch-size "$ACT_BATCH_SIZE" \
    --num-workers "$ACT_NUM_WORKERS" \
    --chunk-size "$ACT_CHUNK_SIZE" \
    --target-frequency "$ACT_TARGET_FREQUENCY" \
    --checkpoint-every "$ACT_CHECKPOINT_EVERY" \
    --device "$ACT_DEVICE" \
    2>&1 | tee "$LOG_DIR/act_train.log"
else
  echo "========== ACT training skipped =========="
fi

echo
echo "========== All training finished =========="
echo "DP log: $LOG_DIR/dp_train.log"
echo "ACT log: $LOG_DIR/act_train.log"
echo "ACT ckpt: $ACT_CKPT_DIR"
