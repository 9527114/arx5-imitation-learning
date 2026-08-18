#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ENV_SCRIPT="$PROJECT_ROOT/activate_arx5_env.sh"
DEV="${CAN_DEV:-/dev/ttyACM0}"
IFACE="${CAN_IFACE:-can1}"
SLCAN_SPEED="${SLCAN_SPEED:-s8}"

if [[ ! -e "$DEV" ]]; then
  echo "CAN device not found: $DEV" >&2
  echo "Available candidates:" >&2
  find /dev -maxdepth 1 \( -name 'ttyACM*' -o -name 'arxcan*' -o -name 'ttyUSB*' \) -print >&2
  exit 1
fi

if ip link show "$IFACE" >/dev/null 2>&1; then
  echo "$IFACE already exists; setting it UP"
  sudo ip link set "$IFACE" up
else
  echo "Starting $IFACE from $DEV"
  sudo slcand -o -c "-$SLCAN_SPEED" "$DEV" "$IFACE"
  sleep 0.3

  if ! ip link show "$IFACE" >/dev/null 2>&1; then
    echo "Failed to create $IFACE from $DEV" >&2
    echo "Check whether another slcand process is holding $DEV." >&2
    exit 1
  fi

  sudo ip link set "$IFACE" up
fi

ip link show "$IFACE"

if [[ -f "$ENV_SCRIPT" ]]; then
  echo ""
  echo "To set ARX5 Python/SDK environment in this terminal, run:"
  echo "  source $ENV_SCRIPT"
fi
