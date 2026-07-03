#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hjs/Projects/table_arrangement/tidy_dataset"
EXP_ROOT="$ROOT/data/sgbot/exp0"
RUNNER="$ROOT/sgbot_comparison/run_sgbot_compare.py"
LOG_DIR="$ROOT/sgbot_comparison/logs"
LOG_FILE="$LOG_DIR/exp0_without_obstacle.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[start] exp0 without obstacle $(date --iso-8601=seconds)"
mapfile -t SCENES < <(find "$EXP_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
echo "[info] scenes=${#SCENES[@]}"

for scene_dir in "${SCENES[@]}"; do
  echo "[scene-start] $(basename "$scene_dir") $(date --iso-8601=seconds)"
  xvfb-run -a conda run -n sgbot --no-capture-output python "$RUNNER" --no-obstacle "$scene_dir"
  echo "[scene-done] $(basename "$scene_dir") $(date --iso-8601=seconds)"
done

echo "[done] exp0 without obstacle $(date --iso-8601=seconds)"
