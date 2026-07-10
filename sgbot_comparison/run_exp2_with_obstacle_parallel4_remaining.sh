#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/hjs/Projects/table_arrangement/tidy_dataset"
EXP_ROOT="$ROOT/data/sgbot/exp2"
RUNNER="$ROOT/sgbot_comparison/run_sgbot_compare.py"
LOG_ROOT="$ROOT/sgbot_comparison/logs"
PARALLEL=4
SKIP_FIRST=5

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_ROOT/exp2_with_obstacle_parallel4_remaining_$STAMP"
RUN_LOG="$LOG_ROOT/exp2_with_obstacle_parallel4_remaining_$STAMP.log"

mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_LOG") 2>&1

find "$EXP_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n +"$((SKIP_FIRST + 1))" > "$RUN_DIR/scenes.txt"
TOTAL="$(wc -l < "$RUN_DIR/scenes.txt")"

echo "[start] exp2 with obstacle parallel=$PARALLEL skip_first=$SKIP_FIRST total=$TOTAL $(date --iso-8601=seconds)"
echo "[info] run_dir=$RUN_DIR"
echo "[info] run_log=$RUN_LOG"

run_one() {
  local idx="$1"
  local scene_dir="$2"
  local name
  local scene_log
  local status_file

  name="$(basename "$scene_dir")"
  scene_log="$RUN_DIR/${idx}_${name}.log"
  status_file="$RUN_DIR/${idx}.status"

  echo "[scene-start] [$idx/$TOTAL] $name $(date --iso-8601=seconds)"
  if xvfb-run -a conda run -n sgbot --no-capture-output python "$RUNNER" "$scene_dir" --no-vlm > "$scene_log" 2>&1; then
    echo -e "$(date --iso-8601=seconds)\tOK\t$idx\t$name\t$scene_log" > "$status_file"
    echo "[scene-done]  [$idx/$TOTAL] $name $(date --iso-8601=seconds)"
  else
    local code="$?"
    echo -e "$(date --iso-8601=seconds)\tFAIL($code)\t$idx\t$name\t$scene_log" > "$status_file"
    echo "[scene-fail]  [$idx/$TOTAL] $name code=$code log=$scene_log $(date --iso-8601=seconds)"
  fi
}

idx=0
running=0
while IFS= read -r scene_dir; do
  idx="$((idx + 1))"
  run_one "$idx" "$scene_dir" &
  running="$((running + 1))"
  if [ "$running" -ge "$PARALLEL" ]; then
    wait -n
    running="$((running - 1))"
  fi
done < "$RUN_DIR/scenes.txt"

while [ "$running" -gt 0 ]; do
  wait -n
  running="$((running - 1))"
done

cat "$RUN_DIR"/*.status | sort -t $'\t' -k3,3n > "$RUN_DIR/status.tsv"
FAILED="$(rg -c $'\tFAIL' "$RUN_DIR/status.tsv" || echo 0)"

echo "[done] exp2 with obstacle total=$TOTAL failed=$FAILED $(date --iso-8601=seconds)"
echo "[done] status=$RUN_DIR/status.tsv"

test "$FAILED" -eq 0
