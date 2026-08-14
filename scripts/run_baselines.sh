#!/usr/bin/env bash
# Runs the 6 BASELINE encoders (everything except "ours" -- that's
# assumed to already be running manually elsewhere, e.g. your current
# cuda:2 finetuning run) split across two GPUs:
#   - FREE_GPU:  starts immediately (default 3)
#   - BUSY_GPU:  waits until it's actually free (memory usage drops
#                below FREE_THRESHOLD_MIB) before starting its queue --
#                default 2, since that's where "ours" is running right
#                now. Polls nvidia-smi every POLL_SECONDS.
#
# Same underlying scripts/train_model.py call, same flags/convention as
# scripts/run_all_models.sh (PRETRAIN_EPOCHS=1, VAL_SAMPLE_SIZE=3000 by
# default, both overridable via env var) -- this is just a variant of
# that script for "one GPU is busy with a manual run right now, queue
# the rest and start the busy one automatically once it's done."
#
# Usage:
#   ./scripts/run_baselines.sh
#   FREE_GPU=3 BUSY_GPU=2 ./scripts/run_baselines.sh
#   POLL_SECONDS=30 FREE_THRESHOLD_MIB=2000 ./scripts/run_baselines.sh
set -euo pipefail

SYNSQL_ROOT="${SYNSQL_ROOT:-../SynSQL-2.5M}"
DATABASES_ROOT="${DATABASES_ROOT:-$SYNSQL_ROOT/databases}"
QUESTIONS_JSON="${QUESTIONS_JSON:-$SYNSQL_ROOT/questions_with_tables.json}"
TABLES_JSON="${TABLES_JSON:-$SYNSQL_ROOT/tables.json}"
SPLIT_JSON="${SPLIT_JSON:-configs/splits/query_split.json}"
CORPUS_JSON="${CORPUS_JSON:-configs/splits/corpus.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-eval/report_runs}"
LOG_DIR="${LOG_DIR:-eval/report_runs/logs}"
SEED="${SEED:-42}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
VAL_SAMPLE_SIZE="${VAL_SAMPLE_SIZE:-3000}"

FREE_GPU="${FREE_GPU:-3}"
BUSY_GPU="${BUSY_GPU:-2}"
POLL_SECONDS="${POLL_SECONDS:-60}"
FREE_THRESHOLD_MIB="${FREE_THRESHOLD_MIB:-2000}"  # under this = "free enough" to start

# Everything except "ours" -- that model is assumed already running
# manually. Rerun scripts/run_all_models.sh (or pass --encoder ours to
# train_model.py yourself) separately if you need to relaunch it.
BASELINES=(bert tabbie strubert tapas turl hytrel)

mkdir -p "$LOG_DIR"

TABLES_FLAG=()
if [ -n "$TABLES_JSON" ]; then
  TABLES_FLAG=(--tables_json "$TABLES_JSON")
fi

run_encoder() {
  local encoder="$1"
  local gpu="$2"
  local log_file="$LOG_DIR/${encoder}.log"
  echo "[gpu${gpu}] starting ${encoder} -- log: ${log_file}"
  python -m scripts.train_model \
    --encoder "$encoder" \
    --databases_root "$DATABASES_ROOT" \
    --questions_json "$QUESTIONS_JSON" \
    --split_json "$SPLIT_JSON" \
    --corpus_json "$CORPUS_JSON" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --device "cuda:${gpu}" \
    --seed "$SEED" \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --val_sample_size "$VAL_SAMPLE_SIZE" \
    "${TABLES_FLAG[@]}" \
    > "$log_file" 2>&1
  local status=$?
  if [ $status -eq 0 ]; then
    echo "[gpu${gpu}] finished ${encoder}"
  else
    echo "[gpu${gpu}] FAILED ${encoder} (exit ${status}) -- see ${log_file}"
  fi
  return $status
}

wait_for_gpu_free() {
  local gpu="$1"
  echo "[gpu${gpu}] waiting for GPU ${gpu} to free up (polling every ${POLL_SECONDS}s, threshold ${FREE_THRESHOLD_MIB} MiB used) ..."
  while true; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d '[:space:]')
    if [ -z "$used" ]; then
      echo "[gpu${gpu}] nvidia-smi query failed -- retrying in ${POLL_SECONDS}s"
    elif [ "$used" -lt "$FREE_THRESHOLD_MIB" ]; then
      echo "[gpu${gpu}] GPU ${gpu} is free (memory used: ${used} MiB) -- starting queue"
      return
    else
      echo "[gpu${gpu}] still busy (memory used: ${used} MiB) -- checking again in ${POLL_SECONDS}s"
    fi
    sleep "$POLL_SECONDS"
  done
}

run_queue() {
  local gpu="$1"
  local wait_first="$2"
  shift 2
  if [ "$wait_first" = "true" ]; then
    wait_for_gpu_free "$gpu"
  fi
  local failures=0
  for encoder in "$@"; do
    if ! run_encoder "$encoder" "$gpu"; then
      failures=$((failures + 1))
    fi
  done
  exit $failures
}

# Split baselines alternately between the two GPUs (3 each).
queue_free=()
queue_busy=()
for i in "${!BASELINES[@]}"; do
  if (( i % 2 == 0 )); then
    queue_free+=("${BASELINES[$i]}")
  else
    queue_busy+=("${BASELINES[$i]}")
  fi
done

echo "GPU ${FREE_GPU} (starts now) queue: ${queue_free[*]}"
echo "GPU ${BUSY_GPU} (waits until free) queue: ${queue_busy[*]}"
echo

run_queue "$FREE_GPU" "false" "${queue_free[@]}" &
pid_free=$!
run_queue "$BUSY_GPU" "true" "${queue_busy[@]}" &
pid_busy=$!

exit_code=0
wait $pid_free || exit_code=1
wait $pid_busy || exit_code=1

echo
echo "=== Summary (best validation MAP / test MAP per encoder) ==="
python3 - "$CHECKPOINT_DIR" "${BASELINES[@]}" << 'PYEOF'
import json
import os
import sys

checkpoint_dir = sys.argv[1]
encoders = sys.argv[2:]

combined = []
print(f"{'encoder':<10} {'best_val_map':>13} {'test_map':>10}")
for encoder in encoders:
    results_path = os.path.join(checkpoint_dir, encoder, "results.json")
    if not os.path.exists(results_path):
        print(f"{encoder:<10} {'<no results.json found>':>13}")
        continue
    with open(results_path, "r", encoding="utf-8") as f:
        r = json.load(f)
    combined.append(r)
    val_map = r.get("best_val_map")
    test_map = r.get("test_map")
    val_str = f"{val_map:.4f}" if val_map is not None else "n/a"
    test_str = f"{test_map:.4f}" if test_map is not None else "n/a"
    print(f"{encoder:<10} {val_str:>13} {test_str:>10}")

combined_path = os.path.join(checkpoint_dir, "combined_results_baselines.json")
with open(combined_path, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2)
print(f"\nwrote combined results for {len(combined)}/{len(encoders)} encoder(s) to {combined_path}")
PYEOF

if [ $exit_code -ne 0 ]; then
  echo
  echo "One or more encoders failed -- check the per-encoder logs in $LOG_DIR"
fi
exit $exit_code
