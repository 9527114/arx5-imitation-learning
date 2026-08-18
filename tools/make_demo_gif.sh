#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  cat <<'EOF'
Usage:
  tools/make_demo_gif.sh INPUT.mp4 START_TIME DURATION OUTPUT.gif

Example:
  tools/make_demo_gif.sh raw_demo.mp4 00:00:05 8 assets/demos/arx5_dp_grasp_main.gif

This script does not modify the input video.
EOF
  exit 2
fi

INPUT="$1"
START="$2"
DURATION="$3"
OUTPUT="$4"

mkdir -p "$(dirname "$OUTPUT")"

PALETTE="$(mktemp --suffix=.png)"
trap 'rm -f "$PALETTE"' EXIT

ffmpeg -y -ss "$START" -t "$DURATION" -i "$INPUT" \
  -vf "fps=12,scale=720:-1:flags=lanczos,palettegen" \
  "$PALETTE"

ffmpeg -y -ss "$START" -t "$DURATION" -i "$INPUT" -i "$PALETTE" \
  -lavfi "fps=12,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse" \
  "$OUTPUT"

echo "Saved GIF: $OUTPUT"
