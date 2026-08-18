#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source "${ROOT_DIR}/activate_arx5_env.sh"
set -u

RUN_NAME="${RUN_NAME:-glue_clean_dp_joint_200}"
DATASET_PATH="${DATASET_PATH:-data_local/glue_clean}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_WORKERS="${VAL_WORKERS:-2}"
VAL_EVERY="${VAL_EVERY:-10}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
WANDB_MODE_ARG="${WANDB_MODE_ARG:-offline}"
TARGET_FREQUENCY="${TARGET_FREQUENCY:-20}"
CACHE_DIR="${CACHE_DIR:-}"

DP_ROOT="${ROOT_DIR}/diffusion_policy-main"
OUT_DIR="${DP_ROOT}/data/outputs/manual/${RUN_NAME}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_NAME}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
export PYTHONPATH="${ROOT_DIR}:${DP_ROOT}:${PYTHONPATH:-}"
cd "${DP_ROOT}"

echo "========== DP-Joint training =========="
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUT_DIR}"
echo "Action: [robot_joint(6), robot_gripper(1)]"
echo "Target frequency: ${TARGET_FREQUENCY}"
if [[ -n "${CACHE_DIR}" ]]; then
  mkdir -p "${CACHE_DIR}"
  echo "Cache dir: ${CACHE_DIR}"
fi

dataset_overrides=(
  "task.dataset.target_frequency=${TARGET_FREQUENCY}"
)
if [[ -n "${CACHE_DIR}" ]]; then
  dataset_overrides+=("task.dataset.cache_dir=${CACHE_DIR}")
fi

dataloader_overrides=()
if [[ "${NUM_WORKERS}" == "0" ]]; then
  dataloader_overrides+=("dataloader.persistent_workers=False")
fi
if [[ "${VAL_WORKERS}" == "0" ]]; then
  dataloader_overrides+=("val_dataloader.persistent_workers=False")
fi

python train.py \
  --config-name=train_diffusion_unet_arx5_joint_hybrid_workspace \
  "task.dataset_path=${DATASET_PATH}" \
  "${dataset_overrides[@]}" \
  "training.num_epochs=${EPOCHS}" \
  "training.checkpoint_every=${CHECKPOINT_EVERY}" \
  "training.rollout_every=0" \
  "training.sample_every=0" \
  "training.val_every=${VAL_EVERY}" \
  "training.max_val_steps=${MAX_VAL_STEPS}" \
  "dataloader.batch_size=${BATCH_SIZE}" \
  "val_dataloader.batch_size=${BATCH_SIZE}" \
  "dataloader.num_workers=${NUM_WORKERS}" \
  "val_dataloader.num_workers=${VAL_WORKERS}" \
  "${dataloader_overrides[@]}" \
  "logging.mode=${WANDB_MODE_ARG}" \
  "hydra.run.dir=${OUT_DIR}" \
  2>&1 | tee "${LOG_DIR}/train.log"

echo "DP-Joint latest ckpt: ${OUT_DIR}/checkpoints/latest.ckpt"
