#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DP_ROOT="$PROJECT_ROOT/diffusion_policy-main"

RUN_NAME="${RUN_NAME:-glue_motion_edge_v2_three_$(date +%Y%m%d_%H%M%S)}"
DATASET_PATH="${DATASET_PATH:-data_local/data_pro/glue_motion_edge_v2}"
TARGET_FREQUENCY="${TARGET_FREQUENCY:-20}"
CACHE_ROOT="${CACHE_ROOT:-$DP_ROOT/data_local/cache/glue_motion_edge_v2}"
DP_CACHE_DIR="${DP_CACHE_DIR:-$CACHE_ROOT/dp}"
DP_CACHE_PATH="${DP_CACHE_PATH:-}"
ACT_CACHE_DIR="${ACT_CACHE_DIR:-$CACHE_ROOT/act}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_DP="${SKIP_DP:-0}"
SKIP_ACT_EEF="${SKIP_ACT_EEF:-0}"
SKIP_ACT_JOINT="${SKIP_ACT_JOINT:-0}"

DP_EPOCHS="${DP_EPOCHS:-200}"
DP_BATCH_SIZE="${DP_BATCH_SIZE:-16}"
DP_NUM_WORKERS="${DP_NUM_WORKERS:-4}"
DP_CHECKPOINT_EVERY="${DP_CHECKPOINT_EVERY:-25}"
DP_VAL_EVERY="${DP_VAL_EVERY:-1000000}"
DP_SAMPLE_EVERY="${DP_SAMPLE_EVERY:-1000000}"
DP_DEVICE="${DP_DEVICE:-cuda:0}"

ACT_EPOCHS="${ACT_EPOCHS:-200}"
ACT_BATCH_SIZE="${ACT_BATCH_SIZE:-16}"
ACT_NUM_WORKERS="${ACT_NUM_WORKERS:-0}"
ACT_CHUNK_SIZE="${ACT_CHUNK_SIZE:-50}"
ACT_TARGET_FREQUENCY="${ACT_TARGET_FREQUENCY:-$TARGET_FREQUENCY}"
ACT_CHECKPOINT_EVERY="${ACT_CHECKPOINT_EVERY:-25}"
ACT_DEVICE="${ACT_DEVICE:-cuda}"
ACT_VAL_RATIO="${ACT_VAL_RATIO:-0.1}"
ACT_VAL_EVERY="${ACT_VAL_EVERY:-0}"
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

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/$RUN_NAME}"
DP_OUTPUT_DIR="${DP_OUTPUT_DIR:-$DP_ROOT/data/outputs/manual/$RUN_NAME/dp}"
ACT_EEF_CKPT_DIR="${ACT_EEF_CKPT_DIR:-$PROJECT_ROOT/act_outputs/$RUN_NAME/eef}"
ACT_JOINT_CKPT_DIR="${ACT_JOINT_CKPT_DIR:-$PROJECT_ROOT/act_outputs/$RUN_NAME/joint}"

if [[ "$DATASET_PATH" = /* ]]; then
  DATASET_ABS="$DATASET_PATH"
  DATASET_FOR_DP="$DATASET_PATH"
else
  DATASET_ABS="$DP_ROOT/$DATASET_PATH"
  DATASET_FOR_DP="$DATASET_PATH"
fi

run_cmd() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

require_glob() {
  local pattern="$1"
  local label="$2"
  compgen -G "$pattern" >/dev/null || {
    echo "Missing $label: $pattern" >&2
    exit 1
  }
}

echo "========== Three-model training chain =========="
echo "Run name: $RUN_NAME"
echo "Dry run: $DRY_RUN"
echo "Dataset: $DATASET_ABS"
echo "Target frequency: $TARGET_FREQUENCY"
echo "DP cache dir: $DP_CACHE_DIR"
echo "DP cache path: ${DP_CACHE_PATH:-<auto>}"
echo "ACT cache dir: $ACT_CACHE_DIR"
echo "Logs: $LOG_DIR"
echo "DP output: $DP_OUTPUT_DIR"
echo "ACT-EFF output: $ACT_EEF_CKPT_DIR"
echo "ACT-Joint output: $ACT_JOINT_CKPT_DIR"

require_path "$DATASET_ABS/replay_buffer.zarr" "dataset replay_buffer"
require_path "$DATASET_ABS/videos" "dataset videos"
require_glob "$DP_CACHE_DIR/arx5_*.zarr.zip" "DP cache"
require_glob "$ACT_CACHE_DIR/act_eef_cache_*.zarr" "ACT-EFF cache"
require_glob "$ACT_CACHE_DIR/act_joint_cache_*.zarr" "ACT-Joint cache"

mkdir -p "$LOG_DIR" "$DP_OUTPUT_DIR" "$ACT_EEF_CKPT_DIR" "$ACT_JOINT_CKPT_DIR"

if [[ -z "$DP_CACHE_PATH" ]]; then
  DP_CACHE_PATH="$(find "$DP_CACHE_DIR" -maxdepth 1 -type f -name 'arx5_*.zarr.zip' -size +1M | sort | head -n 1)"
fi
require_path "$DP_CACHE_PATH" "explicit DP cache"
echo "Resolved DP cache path: $DP_CACHE_PATH"

if [[ "$DRY_RUN" != "1" ]]; then
  set +u
  source "$PROJECT_ROOT/activate_arx5_env.sh"
  set -u
fi

if [[ "$SKIP_DP" != "1" ]]; then
  echo
  echo "========== DP training started =========="
  (
    cd "$DP_ROOT"
    run_cmd python train.py \
      --config-name=train_diffusion_unet_arx5_hybrid_workspace \
      task.dataset_path="$DATASET_FOR_DP" \
      task.dataset.cache_dir="$DP_CACHE_DIR" \
      task.dataset.cache_path="$DP_CACHE_PATH" \
      task.dataset.target_frequency="$TARGET_FREQUENCY" \
      training.device="$DP_DEVICE" \
      training.num_epochs="$DP_EPOCHS" \
      dataloader.batch_size="$DP_BATCH_SIZE" \
      val_dataloader.batch_size="$DP_BATCH_SIZE" \
      dataloader.num_workers="$DP_NUM_WORKERS" \
      val_dataloader.num_workers="$DP_NUM_WORKERS" \
      training.checkpoint_every="$DP_CHECKPOINT_EVERY" \
      training.val_every="$DP_VAL_EVERY" \
      training.sample_every="$DP_SAMPLE_EVERY" \
      logging.mode=offline \
      hydra.run.dir="$DP_OUTPUT_DIR"
  ) 2>&1 | tee "$LOG_DIR/dp_train.log"
  echo "========== DP training finished =========="
else
  echo
  echo "========== DP training skipped =========="
fi

train_act_mode() {
  local mode="$1"
  local ckpt_dir="$2"
  local log_path="$3"
  echo
  echo "========== ACT $mode training started =========="
  (
    cd "$PROJECT_ROOT"
    run_cmd python -m arx5_act.train_act \
      --dataset-path "$DATASET_ABS" \
      --ckpt-dir "$ckpt_dir" \
      --num-epochs "$ACT_EPOCHS" \
      --batch-size "$ACT_BATCH_SIZE" \
      --num-workers "$ACT_NUM_WORKERS" \
      --chunk-size "$ACT_CHUNK_SIZE" \
      --target-frequency "$ACT_TARGET_FREQUENCY" \
      --state-mode "$mode" \
      --cache-dir "$ACT_CACHE_DIR" \
      --val-ratio "$ACT_VAL_RATIO" \
      --val-every "$ACT_VAL_EVERY" \
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
      --device "$ACT_DEVICE"
  ) 2>&1 | tee "$log_path"
  echo "========== ACT $mode training finished =========="
}

if [[ "$SKIP_ACT_EEF" != "1" ]]; then
  train_act_mode eef "$ACT_EEF_CKPT_DIR" "$LOG_DIR/act_eef_train.log"
else
  echo
  echo "========== ACT eef training skipped =========="
fi

if [[ "$SKIP_ACT_JOINT" != "1" ]]; then
  train_act_mode joint "$ACT_JOINT_CKPT_DIR" "$LOG_DIR/act_joint_train.log"
else
  echo
  echo "========== ACT joint training skipped =========="
fi

echo
echo "========== All requested stages finished =========="
echo "DP log: $LOG_DIR/dp_train.log"
echo "ACT-EFF log: $LOG_DIR/act_eef_train.log"
echo "ACT-Joint log: $LOG_DIR/act_joint_train.log"
echo "DP output: $DP_OUTPUT_DIR"
echo "ACT-EFF ckpt: $ACT_EEF_CKPT_DIR"
echo "ACT-Joint ckpt: $ACT_JOINT_CKPT_DIR"
