#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/activate_arx5_env.sh"
set -u
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

EXECUTE_FLAG="${1:-}"

python -m arx5_act.run_act_policy \
  --ckpt-dir act_outputs/glue_motion_edge_v2_three_200/joint \
  --ckpt-name policy_best.ckpt \
  --model X5 \
  --interface can1 \
  --usb-device 0 \
  --device cuda \
  --command-mode cmd \
  --arm-gain-mode pro \
  --arm-kp-scale 1 \
  --arm-kd-scale 1 \
  --preview-time 0.1 \
  --steps-per-inference 8 \
  --command-latency 0.01 \
  --action-exec-latency 0.01 \
  --temporal-agg \
  --temporal-agg-k 0.01 \
  --temporal-agg-order oldest \
  --query-frequency 1 \
  --max-action-joint-step 0.04 \
  --max-action-gripper-step 0.003 \
  --reset-target home \
  --reset-gripper-target sdk \
  --reset-gain-mode default \
  --reset-arm-kp-scale 1 \
  --reset-arm-kd-scale 1 \
  --reset-attempts 0 \
  --gripper-safe-torque 0.75 \
  --gripper-safe-margin 0.002 \
  ${EXECUTE_FLAG}
