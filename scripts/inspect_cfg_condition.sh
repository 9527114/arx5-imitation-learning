#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/activate_arx5_env.sh"
set -u

CKPT_PATH="${CKPT_PATH:-diffusion_policy-main/data/outputs/2026.07.15/17.07.27_train_diffusion_unet_arx5_hybrid_arx5_image/checkpoints/latest.ckpt}"
DATASET_PATH="${DATASET_PATH:-diffusion_policy-main/data_local/data_pro/glue_motion_base}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
TARGET_FREQUENCY="${TARGET_FREQUENCY:-20}"
DEVICE="${DEVICE:-cuda:0}"

cd "${ROOT_DIR}"

python -m arx5_dp_cfg.scripts.inspect_cfg_condition_effect \
  --ckpt "${CKPT_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --target-frequency "${TARGET_FREQUENCY}" \
  --device "${DEVICE}" \
  --num-samples "${NUM_SAMPLES}"
