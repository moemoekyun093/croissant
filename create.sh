#!/usr/bin/env bash
#
# Scaffolds the new src/ package structure (+ configs/, eval/report_runs/).
# Safe to re-run: only creates directories/files that don't already exist,
# never overwrites anything, and never touches jar/ or scripts/.

set -euo pipefail

ROOT_DIR="$(pwd)"

echo "Scaffolding project structure under: ${ROOT_DIR}"

# ----------------------------------------------------------------------
# Helper: create a file with a stub docstring, only if it doesn't exist
# ----------------------------------------------------------------------
create_stub() {
    local filepath="$1"
    local docstring="$2"

    if [ -f "$filepath" ]; then
        echo "  skip (exists): $filepath"
        return
    fi

    mkdir -p "$(dirname "$filepath")"
    cat > "$filepath" <<EOF
"""
${docstring}
"""
EOF
    echo "  created: $filepath"
}

create_empty_init() {
    local filepath="$1"
    if [ -f "$filepath" ]; then
        echo "  skip (exists): $filepath"
        return
    fi
    mkdir -p "$(dirname "$filepath")"
    touch "$filepath"
    echo "  created: $filepath"
}

# ----------------------------------------------------------------------
# configs/
# ----------------------------------------------------------------------
echo ""
echo "-- configs/"
mkdir -p configs
create_stub "configs/model.yaml"    "Hyperparameters for the custom transformer architecture (dims, layers, heads)."
create_stub "configs/training.yaml" "Training hyperparameters: learning rate, batch size, epochs, optimizer settings."
create_stub "configs/indexing.yaml" "Corpus paths, index paths, and model names used for baseline indexing."

# ----------------------------------------------------------------------
# src/data
# ----------------------------------------------------------------------
echo ""
echo "-- src/data"
create_empty_init "src/data/__init__.py"
create_stub "src/data/table.py"  "Table and Column dataclasses -- the shared contract used by encoding, models, and scoring."
create_stub "src/data/corpus.py" "Corpus and query loading utilities (load_table_names, get_queries), adapted from colbert_indexed.py."

# ----------------------------------------------------------------------
# src/encoding
# ----------------------------------------------------------------------
echo ""
echo "-- src/encoding"
create_empty_init "src/encoding/__init__.py"
create_stub "src/encoding/cell_encoder.py" "Off-the-shelf BERT-style encoder: table -> per-cell embeddings."

# ----------------------------------------------------------------------
# src/models
# ----------------------------------------------------------------------
echo ""
echo "-- src/models"
create_empty_init "src/models/__init__.py"
create_stub "src/models/column_aggregator.py" "Novel transformer architecture: cell embeddings -> column-level (or table-level) embeddings."

# ----------------------------------------------------------------------
# src/scoring
# ----------------------------------------------------------------------
echo ""
echo "-- src/scoring"
create_empty_init "src/scoring/__init__.py"
create_stub "src/scoring/maxsim.py" "MaxSim scoring over sets of embeddings, agnostic to column- or table-level granularity."

# ----------------------------------------------------------------------
# src/training
# ----------------------------------------------------------------------
echo ""
echo "-- src/training"
create_empty_init "src/training/__init__.py"
create_stub "src/training/trainer.py" "Generic training loop: epoch iteration, checkpointing, logging hooks."
create_stub "src/training/losses.py"  "Contrastive / ranking loss functions for training the column aggregator."
create_stub "src/training/optim.py"   "Optimizer and learning-rate scheduler setup."

# ----------------------------------------------------------------------
# src/retrieval
# ----------------------------------------------------------------------
echo ""
echo "-- src/retrieval"
create_empty_init "src/retrieval/__init__.py"
create_stub "src/retrieval/base.py"          "BaseRetriever interface: encode_documents, encode_queries, retrieve."
create_stub "src/retrieval/colbert_plaid.py" "Corpus-scale ColBERT + PLAID baseline retriever, migrated from colbert_indexed.py."
create_stub "src/retrieval/cell_column.py"   "Retriever wiring together encoding + models + scoring for the cell/column pipeline."

create_empty_init "src/retrieval/indexing/__init__.py"
create_stub "src/retrieval/indexing/lucene_index.py"  "Lucene index building, extracted from build_retrieval_indexes.py."
create_stub "src/retrieval/indexing/colbert_index.py" "ColBERT/PLAID index building + corpus loading, extracted from build_retrieval_indexes.py."

# ----------------------------------------------------------------------
# eval/
# ----------------------------------------------------------------------
echo ""
echo "-- eval/"
mkdir -p eval/report_runs
echo "  created (or already present): eval/report_runs"

echo ""
echo "Done. jar/ and scripts/ were left untouched."