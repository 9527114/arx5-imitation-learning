#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${ROOT_DIR}/activate_arx5_env.sh"

set -u

export PYTHONPATH="${ROOT_DIR}/diffusion_policy-main:${PYTHONPATH:-}"

# =============================================================================
# Deployment configuration
# =============================================================================

DP_SUBMIT_EXTRA_STEPS="${DP_SUBMIT_EXTRA_STEPS:-0}"
DP_TIMESTAMP_MODE="${DP_TIMESTAMP_MODE:-obs}"

DP_REPLACE_FUTURE="${DP_REPLACE_FUTURE:-0}"
DP_REPLACE_BLEND_TIME="${DP_REPLACE_BLEND_TIME:-0.10}"
DP_REPLACE_MIN_LEAD_TIME="${DP_REPLACE_MIN_LEAD_TIME:-0.06}"

DP_PREVIEW_TIME="${DP_PREVIEW_TIME:-0.10}"
DP_VIDEO_DEVICES="${DP_VIDEO_DEVICES:-}"

CKPT_PATH="${CKPT_PATH:-data/outputs/manual/glue_mini_dp_eef_200/checkpoints/latest.ckpt}"

# =============================================================================
# Trajectory log
# =============================================================================
#
# Default location:
#
#   <repo_root>/
#     diffusion_policy-main/
#       data_local/
#         policy_logs/
#           dp_continuous_pro.jsonl
#
# Using ROOT_DIR here makes the log location independent of the shell's
# current working directory.
#
# You can override it when launching:
#
#   TRAJECTORY_LOG=/tmp/dp_test.jsonl ./scripts/run_dp_pro.sh
#

TRAJECTORY_LOG="${
  TRAJECTORY_LOG:-
  ${ROOT_DIR}/diffusion_policy-main/data_local/policy_logs/dp_continuous_pro.jsonl
}"

# Make sure the log directory exists.
mkdir -p "$(dirname "${TRAJECTORY_LOG}")"

# =============================================================================
# Future trajectory mode
# =============================================================================

if [[ "${DP_REPLACE_FUTURE}" == "1" || "${DP_REPLACE_FUTURE}" == "true" ]]; then
  REPLACE_FUTURE_FLAG="--continuous-replace-future"
else
  REPLACE_FUTURE_FLAG="--continuous-append-future"
fi

# =============================================================================
# Video devices
# =============================================================================

VIDEO_DEVICE_ARGS=()

if [[ -n "${DP_VIDEO_DEVICES}" ]]; then
  VIDEO_DEVICE_ARGS+=(--video-devices "${DP_VIDEO_DEVICES}")
fi

# =============================================================================
# Check checkpoint type
# =============================================================================

CKPT_TARGET="$(python - <<PY
import torch
from pathlib import Path

p = Path("${CKPT_PATH}")

if not p.is_absolute():
    p = Path("${ROOT_DIR}") / "diffusion_policy-main" / p

payload = torch.load(p, map_location="cpu")

print(payload["cfg"].get("_target_", ""))
PY
)"

if [[ "${CKPT_TARGET}" == arx5_dp_cfg.* ]]; then
  cat <<EOF
This checkpoint belongs to arx5_dp_cfg:
  ${CKPT_PATH}
  _target_: ${CKPT_TARGET}

Use scripts/run_dp_cfg_pro.sh for CFG checkpoints.
Use scripts/run_dp_pro.sh only for pure diffusion_policy checkpoints.
EOF

  exit 2
fi

# =============================================================================
# Information
# =============================================================================

echo "[ARX5 DP] checkpoint:"
echo "  ${CKPT_PATH}"

echo "[ARX5 DP] trajectory log:"
echo "  ${TRAJECTORY_LOG}"

echo "[ARX5 DP] future trajectory mode:"
echo "  ${REPLACE_FUTURE_FLAG}"

# =============================================================================
# Run Diffusion Policy
# =============================================================================

python -m arx5_ckpt_loader.run_arx5_policy \
  --ckpt "${CKPT_PATH}" \
  --model X5 \
  --interface can1 \
  --usb-device 0 \
  --device cuda:0 \
  --execute \
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
  --disable-action-safety \
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
  --trajectory-log "${TRAJECTORY_LOG}"
