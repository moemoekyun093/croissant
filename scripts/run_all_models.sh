#!/usr/bin/env bash
# Runs scripts/train_model.py (ELECTRA pretrain + query-table finetune,
# same paradigm/hyperparameters, see that script's docstring) for our
# model AND every baseline encoder, split across GPUs 2 and 3 only --
# GPUs 0 and 1 are busy with other work, this script never touches them.
#
# Each GPU processes its assigned encoders ONE AT A TIME (sequential --
# two encoders sharing a GPU simultaneously would just fight over
# memory/compute), while GPU 2's queue and GPU 3's queue run in PARALLEL
# with each other as background jobs.
#
# Usage (defaults already point at ../SynSQL-2.5M, same layout
# scripts/run_pilot.sh expects -- override any of these env vars if
# your copy lives somewhere else or is laid out differently):
#   ./scripts/run_all_models.sh
#
# PRETRAIN_EPOCHS defaults to 1 (not pretrain.yaml's 15) -- capped low
# here deliberately for a full 7-model sweep's total wall-clock; override
# via env var if you want every encoder pretrained longer. Finetune
# epochs are NOT capped here -- they still come from finetune.yaml's own
# default (currently 10) -- pass FINETUNE_EPOCHS-equivalent flags
# yourself by editing this script if you want that capped too.
#
# VAL_SAMPLE_SIZE defaults to 3000 -- without it, a real split's val set
# (can be hundreds of thousands of queries) gets ranked against the full
# corpus every single finetune epoch, which is prohibitively slow.
#
# Every flag below is overridable via env var, same convention as
# scripts/run_pilot.sh. Split/corpus files must already exist -- run
# scripts/build_query_splits.py once first if configs/splits/*.json
# aren't there yet.
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
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"  # ignored when SKIP_PRETRAIN=true
VAL_SAMPLE_SIZE="${VAL_SAMPLE_SIZE:-3000}"  # see scripts/train_model.py's --val_sample_size --
                                             # a real split's val set can be hundreds of
                                             # thousands of queries; ranking all of them against
                                             # the full corpus every epoch is impractically slow.
SKIP_PRETRAIN="${SKIP_PRETRAIN:-true}"  # see train_model.py's --skip_pretrain -- finetune-only,
                                         # treat as a fast comparison run, not the full agreed
                                         # "ELECTRA pretrain + finetune" methodology for reporting.
                                         # Set to false to run real pretraining for every encoder.
GPUS=(2 3)  # 0 and 1 are busy -- do not add them here.

# Every encoder this codebase can train: "ours" + every registered
# baseline (bert/tabbie/strubert/tapas/turl/hytrel), same list
# scripts/train_model.py's ENCODER_CHOICES resolves to.
ENCODERS=(ours bert tabbie strubert tapas turl hytrel)

mkdir -p "$LOG_DIR"

echo "=== GPU 2 / GPU 3 memory before starting (sanity check -- re-verify these are actually free) ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv -i 2,3 || echo "nvidia-smi unavailable/not on this machine -- skipping check"
echo

TABLES_FLAG=()
if [ -n "$TABLES_JSON" ]; then
  TABLES_FLAG=(--tables_json "$TABLES_JSON")
fi

SKIP_PRETRAIN_FLAG=()
if [ "$SKIP_PRETRAIN" = "true" ]; then
  SKIP_PRETRAIN_FLAG=(--skip_pretrain)
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
    "${SKIP_PRETRAIN_FLAG[@]}" \
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

run_queue() {
  # Runs its assigned encoders one after another, on one GPU.
  local gpu="$1"
  shift
  local failures=0
  for encoder in "$@"; do
    if ! run_encoder "$encoder" "$gpu"; then
      failures=$((failures + 1))
    fi
  done
  exit $failures
}

# Split ENCODERS alternately between the two GPUs so each queue is
# roughly evenly sized (7 encoders -> 4 on one GPU, 3 on the other).
queue_gpu0=()
queue_gpu1=()
for i in "${!ENCODERS[@]}"; do
  if (( i % 2 == 0 )); then
    queue_gpu0+=("${ENCODERS[$i]}")
  else
    queue_gpu1+=("${ENCODERS[$i]}")
  fi
done

echo "GPU ${GPUS[0]} queue: ${queue_gpu0[*]}"
echo "GPU ${GPUS[1]} queue: ${queue_gpu1[*]}"
echo

run_queue "${GPUS[0]}" "${queue_gpu0[@]}" &
pid0=$!
run_queue "${GPUS[1]}" "${queue_gpu1[@]}" &
pid1=$!

exit_code=0
wait $pid0 || exit_code=1
wait $pid1 || exit_code=1

echo
echo "=== Summary (best validation MAP / test MAP per encoder) ==="
# Each encoder's train_model.py run writes its own structured
# eval/report_runs/<encoder>/results.json (see that script) -- read
# those directly instead of grepping log text, and also combine them
# into ONE persisted, reportable file (not just terminal output) so
# there's a single artifact to hand off for the write-up.
python3 - "$CHECKPOINT_DIR" "${ENCODERS[@]}" << 'PYEOF'
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

combined_path = os.path.join(checkpoint_dir, "combined_results.json")
with open(combined_path, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2)
print(f"\nwrote combined results for {len(combined)}/{len(encoders)} encoder(s) to {combined_path}")
PYEOF

if [ $exit_code -ne 0 ]; then
  echo
  echo "One or more encoders failed -- check the per-encoder logs in $LOG_DIR"
fi
exit $exit_code
