#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/activate_arx5_env.sh"
set -u

CKPT_PATH="${CKPT_PATH:-data/outputs/2026.07.15/18.01.36_train_diffusion_unet_arx5_hybrid_arx5_image/checkpoints/latest.ckpt}"
CFG_VIDEO_DEVICES="${CFG_VIDEO_DEVICES:-}"
CFG_W="${CFG_W:-0.3}"
ALLOW_UNCOND="${ALLOW_UNCOND:-0}"
EAG_POS_THRESHOLD="${EAG_POS_THRESHOLD:-0.02}"
EAG_ROT_THRESHOLD="${EAG_ROT_THRESHOLD:-0.05}"
EAG_TARGET="${EAG_TARGET:-executor_current}"
REPLAN_LOOKAHEAD="${REPLAN_LOOKAHEAD:-0.12}"
ASYNC_SWITCH_LEAD_TIME="${ASYNC_SWITCH_LEAD_TIME:-0.08}"
ASYNC_TARGET_REPLAN_INTERVAL="${ASYNC_TARGET_REPLAN_INTERVAL:-0.24}"
ASYNC_MIN_REPLAN_INTERVAL="${ASYNC_MIN_REPLAN_INTERVAL:-0.16}"
PREVIEW_TIME="${PREVIEW_TIME:-0.05}"
CFG_TIMESTAMP_MODE="${CFG_TIMESTAMP_MODE:-now}"
CFG_STEPS_PER_INFERENCE="${CFG_STEPS_PER_INFERENCE:-7}"
CFG_SUBMIT_EXTRA_STEPS="${CFG_SUBMIT_EXTRA_STEPS:-10}"
CFG_BOUNDARY_BLEND_STEPS="${CFG_BOUNDARY_BLEND_STEPS:-8}"
STARTUP_SAFETY_TIME="${STARTUP_SAFETY_TIME:-1.2}"
STARTUP_MAX_POS_STEP="${STARTUP_MAX_POS_STEP:-0.006}"
STARTUP_MAX_ROT_STEP="${STARTUP_MAX_ROT_STEP:-0.02}"
STARTUP_MAX_GRIPPER_STEP="${STARTUP_MAX_GRIPPER_STEP:-0.004}"
REPLACE_BLEND_TIME="${REPLACE_BLEND_TIME:-0.14}"
REPLACE_MIN_LEAD_TIME="${REPLACE_MIN_LEAD_TIME:-0.08}"
GRIPPER_MARGIN="${GRIPPER_MARGIN:-0.0}"
ACTION_POS_SMOOTHING_ALPHA="${ACTION_POS_SMOOTHING_ALPHA:-1.0}"
ACTION_ROT_SMOOTHING_ALPHA="${ACTION_ROT_SMOOTHING_ALPHA:-1.0}"
ACTION_GRIPPER_SMOOTHING_ALPHA="${ACTION_GRIPPER_SMOOTHING_ALPHA:-1.0}"
CFG_PREPEND_CURRENT="${CFG_PREPEND_CURRENT:-1}"
CFG_DISABLE_ACTION_SAFETY="${CFG_DISABLE_ACTION_SAFETY:-1}"
CFG_PREV_LATENCY="${CFG_PREV_LATENCY:-0.20}"
CFG_PREV_LATENCY_MARGIN="${CFG_PREV_LATENCY_MARGIN:-0.03}"
CFG_PREV_MAX_LATENCY="${CFG_PREV_MAX_LATENCY:-0.25}"
CFG_PREV_COND_STEPS="${CFG_PREV_COND_STEPS:-4}"
CFG_PREV_MAX_START_IDX="${CFG_PREV_MAX_START_IDX:-4}"
CFG_SEED_CURRENT_PREV="${CFG_SEED_CURRENT_PREV:-1}"
CFG_REQUIRE_FULL_PREV="${CFG_REQUIRE_FULL_PREV:-1}"
CFG_REPLACE_ONLY_WHEN_ACTIVE="${CFG_REPLACE_ONLY_WHEN_ACTIVE:-0}"
CFG_DROP_PREFIX="${CFG_DROP_PREFIX:-on}"
CFG_DROP_PREFIX_ONLY_WHEN_GUIDED="${CFG_DROP_PREFIX_ONLY_WHEN_GUIDED:-1}"
CFG_PREFIX_KEEP_STEPS="${CFG_PREFIX_KEEP_STEPS:-1}"
EAG_MODE="${EAG_MODE:-on}"
if [[ "${EAG_MODE}" == "off" || "${EAG_MODE}" == "0" || "${EAG_MODE}" == "false" ]]; then
  EAG_FLAG="--no-eag"
else
  EAG_FLAG="--eag"
fi
if [[ "${CFG_DROP_PREFIX}" == "off" || "${CFG_DROP_PREFIX}" == "0" || "${CFG_DROP_PREFIX}" == "false" ]]; then
  CFG_DROP_PREFIX_FLAG="--no-cfg-drop-conditioned-prefix"
else
  CFG_DROP_PREFIX_FLAG="--cfg-drop-conditioned-prefix"
fi
if [[ "${CFG_DROP_PREFIX_ONLY_WHEN_GUIDED}" == "0" || "${CFG_DROP_PREFIX_ONLY_WHEN_GUIDED}" == "false" || "${CFG_DROP_PREFIX_ONLY_WHEN_GUIDED}" == "off" ]]; then
  CFG_DROP_PREFIX_GUIDED_FLAG="--cfg-drop-prefix-even-unguided"
else
  CFG_DROP_PREFIX_GUIDED_FLAG="--cfg-drop-prefix-only-when-guided"
fi
if [[ "${CFG_REQUIRE_FULL_PREV}" == "0" || "${CFG_REQUIRE_FULL_PREV}" == "false" || "${CFG_REQUIRE_FULL_PREV}" == "off" ]]; then
  CFG_REQUIRE_FULL_PREV_FLAG="--cfg-allow-partial-prev"
else
  CFG_REQUIRE_FULL_PREV_FLAG="--cfg-require-full-prev"
fi
if [[ "${CFG_SEED_CURRENT_PREV}" == "0" || "${CFG_SEED_CURRENT_PREV}" == "false" || "${CFG_SEED_CURRENT_PREV}" == "off" ]]; then
  CFG_SEED_CURRENT_PREV_FLAG="--no-cfg-seed-current-prev"
else
  CFG_SEED_CURRENT_PREV_FLAG="--cfg-seed-current-prev"
fi
if [[ "${CFG_REPLACE_ONLY_WHEN_ACTIVE}" == "0" || "${CFG_REPLACE_ONLY_WHEN_ACTIVE}" == "false" || "${CFG_REPLACE_ONLY_WHEN_ACTIVE}" == "off" ]]; then
  CFG_REPLACE_FLAG="--cfg-always-replace-future"
else
  CFG_REPLACE_FLAG="--cfg-replace-only-when-active"
fi

VIDEO_DEVICE_ARGS=()
if [[ -n "${CFG_VIDEO_DEVICES}" ]]; then
  VIDEO_DEVICE_ARGS+=(--video-devices "${CFG_VIDEO_DEVICES}")
fi

if [[ "${CFG_PREPEND_CURRENT}" == "1" || "${CFG_PREPEND_CURRENT}" == "true" ]]; then
  PREPEND_FLAG="--prepend-current-action"
else
  PREPEND_FLAG="--no-prepend-current-action"
fi

SAFETY_ARGS=()
if [[ "${CFG_DISABLE_ACTION_SAFETY}" == "1" || "${CFG_DISABLE_ACTION_SAFETY}" == "true" ]]; then
  SAFETY_ARGS+=(--disable-action-safety)
fi

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/diffusion_policy-main:${PYTHONPATH:-}"
cd "${ROOT_DIR}/diffusion_policy-main"

if [[ "${CFG_W}" == "0" || "${CFG_W}" == "0.0" ]] && [[ "${ALLOW_UNCOND}" != "1" ]]; then
  cat <<'EOF'
Refusing to run CFG_W=0 by default.

CFG_W=0 is not the original DP policy. It runs the CFG model's unconditional
branch, which was only trained through prev_chunk_dropout and can be much less
stable on the robot.

Use scripts/run_dp_pro.sh for the original DP baseline.
If you intentionally want the unconditional CFG ablation, rerun with:
  ALLOW_UNCOND=1 CFG_W=0 ...
EOF
  exit 2
fi

echo "CFG deployment timing: preview_time=${PREVIEW_TIME}s timestamp_mode=${CFG_TIMESTAMP_MODE} replan_lookahead=${REPLAN_LOOKAHEAD}s cfg_prev_cond_steps=${CFG_PREV_COND_STEPS} cfg_prev_latency=${CFG_PREV_LATENCY}s margin=${CFG_PREV_LATENCY_MARGIN}s require_full_prev=${CFG_REQUIRE_FULL_PREV}"

SUBMIT_EXTRA_ARGS=()
if [[ -n "${CFG_SUBMIT_EXTRA_STEPS}" ]]; then
  SUBMIT_EXTRA_ARGS+=(--submit-extra-steps "${CFG_SUBMIT_EXTRA_STEPS}")
fi

python -m arx5_dp_cfg.run_arx5_cfg_policy \
  --ckpt "${CKPT_PATH}" \
  --model X5 \
  --interface can1 \
  --usb-device 0 \
  "${VIDEO_DEVICE_ARGS[@]}" \
  --device cuda:0 \
  --execute \
  --execution-layer continuous \
  --replan-lookahead "${REPLAN_LOOKAHEAD}" \
  --async-switch-lead-time "${ASYNC_SWITCH_LEAD_TIME}" \
  --async-target-replan-interval "${ASYNC_TARGET_REPLAN_INTERVAL}" \
  --async-min-replan-interval "${ASYNC_MIN_REPLAN_INTERVAL}" \
  --arm-gain-mode pro \
  --arm-kp-scale 1 \
  --arm-kd-scale 1 \
  --preview-time "${PREVIEW_TIME}" \
  --startup-safety-time "${STARTUP_SAFETY_TIME}" \
  --startup-max-pos-step "${STARTUP_MAX_POS_STEP}" \
  --startup-max-rot-step "${STARTUP_MAX_ROT_STEP}" \
  --startup-max-gripper-step "${STARTUP_MAX_GRIPPER_STEP}" \
  --steps-per-inference "${CFG_STEPS_PER_INFERENCE}" \
  "${SUBMIT_EXTRA_ARGS[@]}" \
  --command-latency 0.01 \
  --action-exec-latency 0.01 \
  --timestamp-mode "${CFG_TIMESTAMP_MODE}" \
  --boundary-blend-steps "${CFG_BOUNDARY_BLEND_STEPS}" \
  "${PREPEND_FLAG}" \
  "${SAFETY_ARGS[@]}" \
  --action-pos-smoothing-alpha "${ACTION_POS_SMOOTHING_ALPHA}" \
  --action-rot-smoothing-alpha "${ACTION_ROT_SMOOTHING_ALPHA}" \
  --action-gripper-smoothing-alpha "${ACTION_GRIPPER_SMOOTHING_ALPHA}" \
  --continuous-frequency 200 \
  --continuous-max-pos-speed 0.65 \
  --continuous-max-rot-speed 1.05 \
  --continuous-replace-blend-time "${REPLACE_BLEND_TIME}" \
  --continuous-replace-min-lead-time "${REPLACE_MIN_LEAD_TIME}" \
  "${CFG_REPLACE_FLAG}" \
  --cfg-prev-action \
  --cfg-prev-cond-steps "${CFG_PREV_COND_STEPS}" \
  --cfg-prev-latency "${CFG_PREV_LATENCY}" \
  --cfg-prev-latency-margin "${CFG_PREV_LATENCY_MARGIN}" \
  --cfg-prev-latency-ema-alpha 0.8 \
  --cfg-prev-max-latency "${CFG_PREV_MAX_LATENCY}" \
  --cfg-prev-max-start-idx "${CFG_PREV_MAX_START_IDX}" \
  "${CFG_SEED_CURRENT_PREV_FLAG}" \
  "${CFG_REQUIRE_FULL_PREV_FLAG}" \
  --cfg-guidance-weight "${CFG_W}" \
  "${EAG_FLAG}" \
  --eag-target "${EAG_TARGET}" \
  --eag-pos-threshold "${EAG_POS_THRESHOLD}" \
  --eag-rot-threshold "${EAG_ROT_THRESHOLD}" \
  "${CFG_DROP_PREFIX_FLAG}" \
  "${CFG_DROP_PREFIX_GUIDED_FLAG}" \
  --cfg-prefix-keep-steps "${CFG_PREFIX_KEEP_STEPS}" \
  --debug-cfg-prev-action \
  --reset-target home \
  --reset-gripper-target sdk \
  --reset-attempts 0 \
  --gripper-margin "${GRIPPER_MARGIN}" \
  --trajectory-log data_local/policy_logs/dp_cfg_continuous_pro.jsonl
