#!/usr/bin/env bash
set -eo pipefail

# =============================================================================
# Project paths
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DP_DIR="${ROOT_DIR}/diffusion_policy-main"

source "${ROOT_DIR}/activate_arx5_env.sh"

set -u

export PYTHONPATH="${DP_DIR}:${PYTHONPATH:-}"


# =============================================================================
# Helper functions
# =============================================================================

is_true() {
  case "${1,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_from_dp_dir() {
  local path="$1"

  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${DP_DIR}/${path}"
  fi
}


# =============================================================================
# Hardware / runtime configuration
# =============================================================================

DP_MODEL="${DP_MODEL:-X5}"
DP_INTERFACE="${DP_INTERFACE:-can1}"
DP_USB_DEVICE="${DP_USB_DEVICE:-0}"
DP_DEVICE="${DP_DEVICE:-cuda:0}"


# =============================================================================
# Safety / execution configuration
# =============================================================================
#
# Safe-by-default:
#
#   ./scripts/run_dp_pro.sh
#
# runs policy inference without enabling real-robot execution.
#
# Real robot:
#
#   DP_EXECUTE=1 ./scripts/run_dp_pro.sh
#
# If the tested experiment specifically requires action safety to be disabled:
#
#   DP_EXECUTE=1 \
#   DP_DISABLE_ACTION_SAFETY=1 \
#   ./scripts/run_dp_pro.sh
#

DP_EXECUTE="${DP_EXECUTE:-0}"
DP_DISABLE_ACTION_SAFETY="${DP_DISABLE_ACTION_SAFETY:-0}"


# =============================================================================
# Policy / scheduling configuration
# =============================================================================

DP_SUBMIT_EXTRA_STEPS="${DP_SUBMIT_EXTRA_STEPS:-0}"
DP_TIMESTAMP_MODE="${DP_TIMESTAMP_MODE:-obs}"

DP_REPLACE_FUTURE="${DP_REPLACE_FUTURE:-0}"
DP_REPLACE_BLEND_TIME="${DP_REPLACE_BLEND_TIME:-0.10}"
DP_REPLACE_MIN_LEAD_TIME="${DP_REPLACE_MIN_LEAD_TIME:-0.06}"

DP_PREVIEW_TIME="${DP_PREVIEW_TIME:-0.10}"
DP_VIDEO_DEVICES="${DP_VIDEO_DEVICES:-}"


# =============================================================================
# Checkpoint
# =============================================================================
#
# Relative CKPT_PATH values are interpreted relative to:
#
#   diffusion_policy-main/
#
# Example:
#
#   CKPT_PATH=data/outputs/manual/glue_mini_dp_eef_200/checkpoints/latest.ckpt
#

CKPT_PATH="${CKPT_PATH:-data/outputs/manual/glue_mini_dp_eef_200/checkpoints/latest.ckpt}"
RESOLVED_CKPT_PATH="$(resolve_from_dp_dir "${CKPT_PATH}")"

if [[ ! -f "${RESOLVED_CKPT_PATH}" ]]; then
  echo "[ARX5 DP] ERROR: checkpoint not found:"
  echo "  ${RESOLVED_CKPT_PATH}"
  exit 2
fi


# =============================================================================
# Trajectory log
# =============================================================================
#
# Default:
#
#   <repo_root>/
#     diffusion_policy-main/
#       data_local/
#         policy_logs/
#           dp_continuous_pro.jsonl
#
# You can override it with:
#
#   TRAJECTORY_LOG=/tmp/dp_test.jsonl ./scripts/run_dp_pro.sh
#
# Relative TRAJECTORY_LOG values are interpreted relative to
# diffusion_policy-main/.
#

TRAJECTORY_LOG="${TRAJECTORY_LOG:-data_local/policy_logs/dp_continuous_pro.jsonl}"
RESOLVED_TRAJECTORY_LOG="$(resolve_from_dp_dir "${TRAJECTORY_LOG}")"

mkdir -p "$(dirname "${RESOLVED_TRAJECTORY_LOG}")"


# =============================================================================
# Optional CLI flags
# =============================================================================

EXECUTE_ARGS=()

if is_true "${DP_EXECUTE}"; then
  EXECUTE_ARGS+=(--execute)
fi


ACTION_SAFETY_ARGS=()

if is_true "${DP_DISABLE_ACTION_SAFETY}"; then
  ACTION_SAFETY_ARGS+=(--disable-action-safety)
fi


VIDEO_DEVICE_ARGS=()

if [[ -n "${DP_VIDEO_DEVICES}" ]]; then
  VIDEO_DEVICE_ARGS+=(--video-devices "${DP_VIDEO_DEVICES}")
fi


if is_true "${DP_REPLACE_FUTURE}"; then
  REPLACE_FUTURE_FLAG="--continuous-replace-future"
else
  REPLACE_FUTURE_FLAG="--continuous-append-future"
fi


# =============================================================================
# Check checkpoint type
# =============================================================================
#
# Prevent accidentally loading an ARX5-DP-CFG checkpoint with the vanilla
# Diffusion Policy deployment entry point.
#

CKPT_TARGET="$(
python - "${RESOLVED_CKPT_PATH}" <<'PY'
import sys
import torch
from pathlib import Path

path = Path(sys.argv[1])

payload = torch.load(
    path,
    map_location="cpu",
)

cfg = payload.get("cfg", {})
print(cfg.get("_target_", ""))
PY
)"


if [[ "${CKPT_TARGET}" == arx5_dp_cfg.* ]]; then
  cat <<EOF

[ARX5 DP] ERROR

This checkpoint belongs to arx5_dp_cfg:

  checkpoint:
    ${RESOLVED_CKPT_PATH}

  _target_:
    ${CKPT_TARGET}

Use:

  scripts/run_dp_cfg_pro.sh

for CFG checkpoints.

Use:

  scripts/run_dp_pro.sh

only for vanilla Diffusion Policy checkpoints.

EOF

  exit 2
fi


# =============================================================================
# Startup summary
# =============================================================================

echo
echo "============================================================"
echo " ARX5 Diffusion Policy Deployment"
echo "============================================================"
echo
echo "Project root:"
echo "  ${ROOT_DIR}"
echo
echo "Checkpoint:"
echo "  ${RESOLVED_CKPT_PATH}"
echo
echo "Checkpoint target:"
echo "  ${CKPT_TARGET:-<unknown>}"
echo
echo "Robot:"
echo "  model      = ${DP_MODEL}"
echo "  interface  = ${DP_INTERFACE}"
echo
echo "Runtime:"
echo "  device     = ${DP_DEVICE}"
echo "  usb camera = ${DP_USB_DEVICE}"
echo
echo "Policy scheduling:"
echo "  timestamp mode         = ${DP_TIMESTAMP_MODE}"
echo "  preview time           = ${DP_PREVIEW_TIME}"
echo "  submit extra steps     = ${DP_SUBMIT_EXTRA_STEPS}"
echo "  replace future         = ${DP_REPLACE_FUTURE}"
echo "  replace blend time     = ${DP_REPLACE_BLEND_TIME}"
echo "  replace min lead time  = ${DP_REPLACE_MIN_LEAD_TIME}"
echo
echo "Trajectory log:"
echo "  ${RESOLVED_TRAJECTORY_LOG}"
echo

if is_true "${DP_EXECUTE}"; then
  echo "Execution:"
  echo "  REAL ROBOT EXECUTION ENABLED"
else
  echo "Execution:"
  echo "  DRY-RUN / POLICY-ONLY MODE"
  echo
  echo "  To enable real robot execution:"
  echo
  echo "    DP_EXECUTE=1 ./scripts/run_dp_pro.sh"
fi

echo

if is_true "${DP_DISABLE_ACTION_SAFETY}"; then
  echo "WARNING:"
  echo "  Action safety is DISABLED."
else
  echo "Action safety:"
  echo "  enabled"
fi

echo
echo "============================================================"
echo


# =============================================================================
# Run
# =============================================================================
#
# Run from diffusion_policy-main so any remaining relative paths inside
# downstream Python code have a deterministic working directory.
#

cd "${DP_DIR}"

python -m arx5_ckpt_loader.run_arx5_policy \
  --ckpt "${RESOLVED_CKPT_PATH}" \
  --model "${DP_MODEL}" \
  --interface "${DP_INTERFACE}" \
  --usb-device "${DP_USB_DEVICE}" \
  --device "${DP_DEVICE}" \
  "${EXECUTE_ARGS[@]}" \
  --execution-layer continuous \
  --arm-gain-mode pro \
  --arm-kp-scale 1 \
  --arm-kd-scale 1 \
  --preview-time "${DP_PREVIEW_TIME}" \
  --steps-per-inference 8 \
  --submit-extra-steps "${DP_SUBMIT_EXTRA_STEPS}" \
  --timestamp-mode "${DP_TIMESTAMP_MODE}" \
  --command-latency 0 \
  --action-exec-latency 0 \
  --boundary-blend-steps 0 \
  --no-prepend-current-action \
  "${ACTION_SAFETY_ARGS[@]}" \
  --continuous-frequency 200 \
  --continuous-max-pos-speed 0.65 \
  --continuous-max-rot-speed 1.05 \
  "${VIDEO_DEVICE_ARGS[@]}" \
  "${REPLACE_FUTURE_FLAG}" \
  --continuous-replace-blend-time "${DP_REPLACE_BLEND_TIME}" \
  --continuous-replace-min-lead-time "${DP_REPLACE_MIN_LEAD_TIME}" \
  --reset-target home \
  --reset-gripper-target sdk \
  --reset-attempts 0 \
  --trajectory-log "${RESOLVED_TRAJECTORY_LOG}"
