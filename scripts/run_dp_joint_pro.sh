#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/activate_arx5_env.sh"
set -u
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/diffusion_policy-main:${PYTHONPATH:-}"

CKPT_PATH="${CKPT_PATH:-data/outputs/manual/glue_mini_dp_joint_200/checkpoints/latest.ckpt}"
DP_JOINT_VIDEO_DEVICES="${DP_JOINT_VIDEO_DEVICES:-}"
DP_JOINT_EXECUTE="${DP_JOINT_EXECUTE:-1}"
DP_JOINT_TIMESTAMP_MODE="${DP_JOINT_TIMESTAMP_MODE:-obs}"
DP_JOINT_STEPS_PER_INFERENCE="${DP_JOINT_STEPS_PER_INFERENCE:-7}"
DP_JOINT_SUBMIT_EXTRA_STEPS="${DP_JOINT_SUBMIT_EXTRA_STEPS:-0}"
DP_JOINT_REPLAN_LOOKAHEAD="${DP_JOINT_REPLAN_LOOKAHEAD:-0.12}"
DP_JOINT_PREVIEW_TIME="${DP_JOINT_PREVIEW_TIME:-0.10}"
DP_JOINT_CONTINUOUS_FREQUENCY="${DP_JOINT_CONTINUOUS_FREQUENCY:-200}"
DP_JOINT_REPLACE_BLEND_TIME="${DP_JOINT_REPLACE_BLEND_TIME:-0.12}"
DP_JOINT_REPLACE_MIN_LEAD_TIME="${DP_JOINT_REPLACE_MIN_LEAD_TIME:-0.06}"
DP_JOINT_REPLACE_FUTURE="${DP_JOINT_REPLACE_FUTURE:-0}"
DP_JOINT_MAX_STEP="${DP_JOINT_MAX_STEP:-0.12}"
DP_JOINT_MAX_GRIPPER_STEP="${DP_JOINT_MAX_GRIPPER_STEP:-0.08}"
DP_JOINT_CLOSE_GATE_DELAY="${DP_JOINT_CLOSE_GATE_DELAY:-0.30}"
DP_JOINT_CLOSE_GATE_THRESHOLD="${DP_JOINT_CLOSE_GATE_THRESHOLD:-0.003}"
DP_JOINT_CLOSE_GATE_RELEASE_WIDTH="${DP_JOINT_CLOSE_GATE_RELEASE_WIDTH:-0.04}"
DP_JOINT_RESET_MODE="${DP_JOINT_RESET_MODE:-hold_sdk_home}"
DP_JOINT_RESET_DURATION="${DP_JOINT_RESET_DURATION:-2.0}"
DP_JOINT_RESET_HOLD_TIME="${DP_JOINT_RESET_HOLD_TIME:-0.2}"

EXECUTE_FLAG=""
if [[ "${DP_JOINT_EXECUTE}" == "1" || "${DP_JOINT_EXECUTE}" == "true" ]]; then
  EXECUTE_FLAG="--execute"
fi

MAX_STEP_ARGS=()
if [[ -n "${DP_JOINT_MAX_STEP}" ]]; then
  MAX_STEP_ARGS+=(--max-action-joint-step "${DP_JOINT_MAX_STEP}")
fi
if [[ -n "${DP_JOINT_MAX_GRIPPER_STEP}" ]]; then
  MAX_STEP_ARGS+=(--max-action-gripper-step "${DP_JOINT_MAX_GRIPPER_STEP}")
fi

VIDEO_DEVICE_ARGS=()
if [[ -n "${DP_JOINT_VIDEO_DEVICES}" ]]; then
  VIDEO_DEVICE_ARGS+=(--video-devices "${DP_JOINT_VIDEO_DEVICES}")
fi

if [[ "${DP_JOINT_REPLACE_FUTURE}" == "1" || "${DP_JOINT_REPLACE_FUTURE}" == "true" ]]; then
  REPLACE_FUTURE_FLAG="--continuous-replace-future"
else
  REPLACE_FUTURE_FLAG="--continuous-append-future"
fi

python -m arx5_ckpt_loader.run_arx5_joint_policy \
  --ckpt "${CKPT_PATH}" \
  --model X5 \
  --interface can1 \
  --usb-device 0 \
  "${VIDEO_DEVICE_ARGS[@]}" \
  --device cuda:0 \
  --command-mode traj \
  --arm-gain-mode pro \
  --arm-kp-scale 1 \
  --arm-kd-scale 1 \
  --preview-time "${DP_JOINT_PREVIEW_TIME}" \
  --continuous-frequency "${DP_JOINT_CONTINUOUS_FREQUENCY}" \
  --continuous-replace-blend-time "${DP_JOINT_REPLACE_BLEND_TIME}" \
  --continuous-replace-min-lead-time "${DP_JOINT_REPLACE_MIN_LEAD_TIME}" \
  "${REPLACE_FUTURE_FLAG}" \
  --steps-per-inference "${DP_JOINT_STEPS_PER_INFERENCE}" \
  --submit-extra-steps "${DP_JOINT_SUBMIT_EXTRA_STEPS}" \
  --replan-lookahead "${DP_JOINT_REPLAN_LOOKAHEAD}" \
  --timestamp-mode "${DP_JOINT_TIMESTAMP_MODE}" \
  --command-latency 0.01 \
  --action-exec-latency 0.01 \
  --prepend-current-action \
  --close-gate-delay "${DP_JOINT_CLOSE_GATE_DELAY}" \
  --close-gate-threshold "${DP_JOINT_CLOSE_GATE_THRESHOLD}" \
  --close-gate-release-width "${DP_JOINT_CLOSE_GATE_RELEASE_WIDTH}" \
  --reset-attempts 0 \
  --reset-mode "${DP_JOINT_RESET_MODE}" \
  --reset-duration "${DP_JOINT_RESET_DURATION}" \
  --reset-hold-time "${DP_JOINT_RESET_HOLD_TIME}" \
  --gripper-safe-torque 0.75 \
  --gripper-safe-margin 0.002 \
  "${MAX_STEP_ARGS[@]}" \
  ${EXECUTE_FLAG}
