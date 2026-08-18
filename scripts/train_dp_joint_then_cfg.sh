#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
set -u

RUN_NAME="${RUN_NAME:-glue_motion_joint_cfg_200}"
DATASET_PATH="${DATASET_PATH:-data_local/glue_motion}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_WORKERS="${VAL_WORKERS:-2}"
VAL_EVERY="${VAL_EVERY:-25}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
TARGET_FREQUENCY="${TARGET_FREQUENCY:-20}"
PREV_COND_STEPS="${PREV_COND_STEPS:-4}"
PREV_CHUNK_DROPOUT="${PREV_CHUNK_DROPOUT:-0.3}"
WANDB_MODE_ARG="${WANDB_MODE_ARG:-offline}"
DRY_RUN="${DRY_RUN:-0}"

DP_ROOT="${ROOT_DIR}/diffusion_policy-main"
DATASET_ABS="${DP_ROOT}/${DATASET_PATH}"
CACHE_ROOT="${DP_ROOT}/data_local/cache/${RUN_NAME}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_NAME}"

mkdir -p "${LOG_DIR}" "${CACHE_ROOT}/dp_joint" "${CACHE_ROOT}/dp_eef_cfg_prev${PREV_COND_STEPS}"

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
  "TARGET_FREQUENCY=${TARGET_FREQUENCY}"
  "WANDB_MODE_ARG=${WANDB_MODE_ARG}"
)

echo "========== DP joint + CFG chain =========="
echo "Run name: ${RUN_NAME}"
echo "Dataset: ${DATASET_ABS}"
echo "Target frequency: ${TARGET_FREQUENCY}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Val every: ${VAL_EVERY}; max val steps: ${MAX_VAL_STEPS}"
echo "CFG prev_cond_steps: ${PREV_COND_STEPS}; prev_chunk_dropout: ${PREV_CHUNK_DROPOUT}"
echo "Joint output: ${DP_ROOT}/data/outputs/manual/${RUN_NAME}/dp_joint"
echo "CFG output: ${DP_ROOT}/data/outputs/manual/${RUN_NAME}/dp_eef_cfg_prev${PREV_COND_STEPS}"
echo "Cache root: ${CACHE_ROOT}"
echo "Logs: ${LOG_DIR}"

if [[ ! -d "${DATASET_ABS}/replay_buffer.zarr" ]]; then
  echo "Dataset replay buffer not found: ${DATASET_ABS}/replay_buffer.zarr" >&2
  exit 1
fi

run_stage "DP-Joint training" \
  env \
  "RUN_NAME=${RUN_NAME}/dp_joint" \
  "CACHE_DIR=${CACHE_ROOT}/dp_joint" \
  "${common_env[@]}" \
  "${ROOT_DIR}/scripts/train_dp_joint.sh"

run_stage "DP-EEF-CFG training" \
  env \
  "RUN_NAME=${RUN_NAME}/dp_eef_cfg_prev${PREV_COND_STEPS}" \
  "CACHE_DIR=${CACHE_ROOT}/dp_eef_cfg_prev${PREV_COND_STEPS}" \
  "${common_env[@]}" \
  "PREV_COND_STEPS=${PREV_COND_STEPS}" \
  "PREV_CHUNK_DROPOUT=${PREV_CHUNK_DROPOUT}" \
  "${ROOT_DIR}/scripts/train_dp_eef_cfg.sh"

echo ""
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "on" ]]; then
  echo "DRY_RUN finished. No training was started."
else
  echo "Training chain finished."
fi
