#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
set -u

RUN_NAME="${RUN_NAME:-glue_clean_three_separate}"
DATASET_PATH="${DATASET_PATH:-data_local/glue_clean}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_WORKERS="${VAL_WORKERS:-2}"
VAL_EVERY="${VAL_EVERY:-10}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
PREV_COND_STEPS="${PREV_COND_STEPS:-4}"
PREV_CHUNK_DROPOUT="${PREV_CHUNK_DROPOUT:-0.25}"
WANDB_MODE_ARG="${WANDB_MODE_ARG:-offline}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="${ROOT_DIR}/logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

run_stage() {
  local stage_name="$1"
  shift
  echo ""
  echo "========== ${stage_name} =========="
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "on" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

common_env=(
  "DATASET_PATH=${DATASET_PATH}"
  "EPOCHS=${EPOCHS}"
  "BATCH_SIZE=${BATCH_SIZE}"
  "NUM_WORKERS=${NUM_WORKERS}"
  "VAL_WORKERS=${VAL_WORKERS}"
  "VAL_EVERY=${VAL_EVERY}"
  "MAX_VAL_STEPS=${MAX_VAL_STEPS}"
  "CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
  "WANDB_MODE_ARG=${WANDB_MODE_ARG}"
)

echo "========== DP three-stage training =========="
echo "Run name: ${RUN_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Dry run: ${DRY_RUN}"
echo "Logs: ${LOG_DIR}"

run_stage "DP-EEF" \
  env \
  "RUN_NAME=${RUN_NAME}/dp_eef" \
  "${common_env[@]}" \
  "${ROOT_DIR}/scripts/train_dp_eef.sh"

run_stage "DP-Joint" \
  env \
  "RUN_NAME=${RUN_NAME}/dp_joint" \
  "${common_env[@]}" \
  "${ROOT_DIR}/scripts/train_dp_joint.sh"

run_stage "DP-EEF-CFG" \
  env \
  "RUN_NAME=${RUN_NAME}/dp_eef_cfg_prev${PREV_COND_STEPS}" \
  "${common_env[@]}" \
  "PREV_COND_STEPS=${PREV_COND_STEPS}" \
  "PREV_CHUNK_DROPOUT=${PREV_CHUNK_DROPOUT}" \
  "${ROOT_DIR}/scripts/train_dp_eef_cfg.sh"

echo ""
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "on" ]]; then
  echo "DRY_RUN finished. No training was started."
else
  echo "All stages finished."
fi
