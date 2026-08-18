#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
set +u
source "${ROOT_DIR}/activate_arx5_env.sh"
set -u

RUN_NAME="${RUN_NAME:-glue_mini_cfg_variants}"
DATASET_PATH="${DATASET_PATH:-data_local/glue_mini}"
EEF_CKPT="${EEF_CKPT:-diffusion_policy-main/data/outputs/manual/glue_mini_dp_eef_200/checkpoints/latest.ckpt}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_WORKERS="${VAL_WORKERS:-2}"
VAL_EVERY="${VAL_EVERY:-25}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
TARGET_FREQUENCY="${TARGET_FREQUENCY:-20}"
WANDB_MODE_ARG="${WANDB_MODE_ARG:-offline}"
PREV_CHUNK_DROPOUT="${PREV_CHUNK_DROPOUT:-0.3}"
WARM_PREV_COND_STEPS="${WARM_PREV_COND_STEPS:-4}"
SCRATCH_PREV_COND_STEPS_A="${SCRATCH_PREV_COND_STEPS_A:-4}"
SCRATCH_PREV_COND_STEPS_B="${SCRATCH_PREV_COND_STEPS_B:-2}"
DRY_RUN="${DRY_RUN:-0}"

DP_ROOT="${ROOT_DIR}/diffusion_policy-main"
LOG_DIR="${ROOT_DIR}/logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
export PYTHONPATH="${ROOT_DIR}:${DP_ROOT}:${PYTHONPATH:-}"

run_cmd() {
  local stage="$1"
  shift
  echo ""
  echo "========== ${stage} =========="
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "on" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

train_cfg() {
  local stage="$1"
  local run_suffix="$2"
  local prev_steps="$3"
  local resume="$4"

  run_cmd "${stage}" \
    env \
    "RUN_NAME=${RUN_NAME}/${run_suffix}" \
    "DATASET_PATH=${DATASET_PATH}" \
    "EPOCHS=${EPOCHS}" \
    "BATCH_SIZE=${BATCH_SIZE}" \
    "NUM_WORKERS=${NUM_WORKERS}" \
    "VAL_WORKERS=${VAL_WORKERS}" \
    "VAL_EVERY=${VAL_EVERY}" \
    "MAX_VAL_STEPS=${MAX_VAL_STEPS}" \
    "CHECKPOINT_EVERY=${CHECKPOINT_EVERY}" \
    "PREV_COND_STEPS=${prev_steps}" \
    "PREV_CHUNK_DROPOUT=${PREV_CHUNK_DROPOUT}" \
    "TARGET_FREQUENCY=${TARGET_FREQUENCY}" \
    "WANDB_MODE_ARG=${WANDB_MODE_ARG}" \
    "RESUME=${resume}" \
    "${ROOT_DIR}/scripts/train_dp_eef_cfg.sh"
}

WARM_OUT="${DP_ROOT}/data/outputs/manual/${RUN_NAME}/cfg_warm_from_eef_prev${WARM_PREV_COND_STEPS}_dropout${PREV_CHUNK_DROPOUT}"

echo "========== Mini CFG variants training =========="
echo "Run name: ${RUN_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "EEF ckpt: ${EEF_CKPT}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Dropout: ${PREV_CHUNK_DROPOUT}"
echo "Warm-start prev_cond_steps: ${WARM_PREV_COND_STEPS}"
echo "Scratch prev_cond_steps: ${SCRATCH_PREV_COND_STEPS_A}, ${SCRATCH_PREV_COND_STEPS_B}"
echo "Logs: ${LOG_DIR}"
echo "Dry run: ${DRY_RUN}"

run_cmd "Create CFG warm-start checkpoint from EEF" \
  python -m arx5_dp_cfg.scripts.init_cfg_from_dp \
    --dp-ckpt "${EEF_CKPT}" \
    --output-dir "${WARM_OUT}" \
    --dataset-path "${DATASET_PATH}" \
    --prev-cond-steps "${WARM_PREV_COND_STEPS}" \
    --prev-chunk-dropout "${PREV_CHUNK_DROPOUT}" \
    --target-frequency "${TARGET_FREQUENCY}" \
    --num-epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --val-every "${VAL_EVERY}" \
    --max-val-steps "${MAX_VAL_STEPS}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --logging-mode "${WANDB_MODE_ARG}"

train_cfg \
  "Train CFG warm-start from EEF" \
  "cfg_warm_from_eef_prev${WARM_PREV_COND_STEPS}_dropout${PREV_CHUNK_DROPOUT}" \
  "${WARM_PREV_COND_STEPS}" \
  "True"

train_cfg \
  "Train CFG scratch prev${SCRATCH_PREV_COND_STEPS_A}" \
  "cfg_scratch_prev${SCRATCH_PREV_COND_STEPS_A}_dropout${PREV_CHUNK_DROPOUT}" \
  "${SCRATCH_PREV_COND_STEPS_A}" \
  "False"

train_cfg \
  "Train CFG scratch prev${SCRATCH_PREV_COND_STEPS_B}" \
  "cfg_scratch_prev${SCRATCH_PREV_COND_STEPS_B}_dropout${PREV_CHUNK_DROPOUT}" \
  "${SCRATCH_PREV_COND_STEPS_B}" \
  "False"

echo ""
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "on" ]]; then
  echo "DRY_RUN finished. No training was started."
else
  echo "All CFG variants finished."
fi
