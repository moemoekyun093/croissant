#!/usr/bin/env bash
# End-to-end pilot run: sanity check -> build query split/corpus ->
# ELECTRA pretraining pilot -> query-table finetuning pilot (early
# stopping on val MAP). Stops at the first failure (set -e) so you
# don't burn GPU time on finetuning against a broken pretrain run.
#
# Edit the paths below (or export them before running), then:
#   chmod +x scripts/run_pilot.sh
#   ./scripts/run_pilot.sh
#
# Recommended to run this inside tmux/screen (see notes at the bottom of
# this file) -- it's long enough that an SSH drop shouldn't kill it.

set -euo pipefail

# ---- edit these ----
SYNSQL_ROOT="${SYNSQL_ROOT:-/path/to/synsql}"
TABLES_JSON="${TABLES_JSON:-$SYNSQL_ROOT/tables.json}"
DATABASES_ROOT="${DATABASES_ROOT:-$SYNSQL_ROOT/databases}"
QUESTIONS_JSON="${QUESTIONS_JSON:-$SYNSQL_ROOT/questions_with_tables.json}"
DEVICE="${DEVICE:-cuda:2}"
SEED="${SEED:-42}"

# pilot scope -- small on purpose, bump these once this run looks healthy
N_DBS="${N_DBS:-20}"
N_TABLES="${N_TABLES:-200}"
N_EXAMPLES="${N_EXAMPLES:-500}"          # query examples, before train/val/test split
CORPUS_SIZE="${CORPUS_SIZE:-2000}"       # retrieval candidate pool size (always includes every query's true positive)
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-2}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-15}" # early stopping will usually end this well before the cap
PATIENCE="${PATIENCE:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# ---------------------

PRETRAIN_DIR="eval/report_runs/pretrain"
FINETUNE_DIR="eval/report_runs/finetune"
SPLIT_JSON="configs/splits/query_split.json"
CORPUS_JSON="configs/splits/corpus.json"

echo "=== [1/4] sanity checks (fast, synthetic data, no GPU-heavy work) ==="
python -m scripts.sanity_checks

echo
echo "=== [2/4] build query train/val/test split + fixed corpus ==="
python -m scripts.build_query_splits \
    --tables_json "$TABLES_JSON" \
    --databases_root "$DATABASES_ROOT" \
    --questions_json "$QUESTIONS_JSON" \
    --n_examples "$N_EXAMPLES" \
    --corpus_size "$CORPUS_SIZE" \
    --seed "$SEED" \
    --split_output "$SPLIT_JSON" \
    --corpus_output "$CORPUS_JSON"

echo
echo "=== [3/4] ELECTRA pretraining pilot ==="
echo "n_dbs=$N_DBS n_tables=$N_TABLES epochs=$PRETRAIN_EPOCHS batch_size=$BATCH_SIZE device=$DEVICE seed=$SEED"
python -m scripts.pretrain_electra \
    --tables_json "$TABLES_JSON" \
    --databases_root "$DATABASES_ROOT" \
    --n_dbs "$N_DBS" \
    --n_tables "$N_TABLES" \
    --num_epochs "$PRETRAIN_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --seed "$SEED"

# pick the checkpoint from the highest epoch number just produced
# (-V = natural/version sort, so epoch10 correctly sorts after epoch9 --
# a plain lexical sort would put epoch10 before epoch2)
LATEST_CKPT=$(ls -1 "$PRETRAIN_DIR"/checkpoint_epoch*.pt 2>/dev/null \
    | sort -V \
    | tail -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: no checkpoint found in $PRETRAIN_DIR -- pretraining must not have completed an epoch." >&2
    exit 1
fi
echo "using checkpoint: $LATEST_CKPT"

echo
echo "=== [4/4] query-table finetuning pilot (early stopping on val MAP) ==="
echo "epochs<=$FINETUNE_EPOCHS patience=$PATIENCE batch_size=$BATCH_SIZE device=$DEVICE seed=$SEED"
python -m scripts.finetune_query_table \
    --tables_json "$TABLES_JSON" \
    --databases_root "$DATABASES_ROOT" \
    --questions_json "$QUESTIONS_JSON" \
    --split_json "$SPLIT_JSON" \
    --corpus_json "$CORPUS_JSON" \
    --pretrained_checkpoint "$LATEST_CKPT" \
    --num_epochs "$FINETUNE_EPOCHS" \
    --patience "$PATIENCE" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --seed "$SEED"

echo
echo "=== pilot run complete ==="
echo "pretrain logs:  $PRETRAIN_DIR/train.log"
echo "finetune logs:  $FINETUNE_DIR/train.log"
echo "best finetune checkpoint: $FINETUNE_DIR/best_model.pt"
