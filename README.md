# Web Table Retrieval Pipeline — Overview

## Repository layout

- `src/` — library code (`models`, `encoding`, `data`, `training`, `scoring`, `eval`, `utils`).
- `scripts/` — runnable entry points (`train_model`, `pretrain_electra`, `finetune_query_table`, `prepare_*`, `eval_*`, `run_*.sh`); run as `python -m scripts.<name>` from the repo root.
- `scripts/data_prep/` — corpus building, normalization, materialization, query splits.
- `scripts/diagnostics/` — data inspection and sanity-check tools.
- `scripts/debug/` — crash repros and profiling.
- `tests/` — tests; run as `python -m tests.<name>`.
- `configs/`, `deploy/`, `docs/` — YAML configs, GPU-monitor service files and docs.

### Training library (`src/training/`, `src/models/`)

- `pretrain_trainer.py` — `PretrainTrainer` (ELECTRA-style replaced-cell discriminator).
- `finetune_trainer.py` — `FinetuneTrainer` (query→table InfoNCE, corpus scoring, MAP/MRR evaluation).
- `scheduler.py` — `build_scheduler` (linear warmup/decay).
- `checkpoints.py` — `latest_checkpoint`, `safe_load_cache`.
- `trainer.py` — thin re-export of the above, so `from src.training.trainer import ...` still works.
- `src/models/factory.py` — `build_table_model(args)` builds Croissant or a baseline from `--encoder`.
- `src/utils/` — shared `clean_text`, `get_sqlite_path`, `atomic_json`, `load_json`.

### Results

`python -m scripts.eval_test ...` writes per-model test MAP/MRR, val MAP, epoch and run arguments to
`results/eval_test_<timestamp>.json` (override with `--results_json`). Training logs and checkpoints are
**not** stored in the repo: they go to `checkpoint_dir` in the configs (`eval/report_runs/...`) or the
`/tmp/ayane/<run>/...` paths used in the runbook, so copy them somewhere permanent after a run.

## Data-prep scripts (`scripts/data_prep/`)

| # | File | Purpose |
|---|------|---------|
| 1 | `scripts/data_prep/build_row_major_corpus.py` | Parses raw web table dumps (`.json.gz`, WDC-style `relation` format) into a flat, row-major JSONL corpus (`[TABLE]/[SCHEMA]/[ROWS]` text format), then builds a Lucene/BM25 index over it. |
| 2 | `scripts/data_prep/build_bird_augmented_corpus.py` | Uses each BIRD table (name + columns + sampled SQLite row values) as a query against the row-major Lucene index, retrieves the top-k most similar web tables per BIRD table, deduplicates into a selected set, and pads with random web tables up to a target size. Writes the filtered/augmented corpus. **Does not include the BIRD tables themselves** — output is web tables only (`source: "webtable"`). |
| 3 | `scripts/data_prep/build_bird_jsonl.py` | Converts the actual BIRD tables (name, columns, sampled row values) into documents in the same corpus schema, tagged `source: "bird"`, with IDs offset from a fixed `START_ID` to avoid collisions with web-table IDs. This is the step that gets BIRD tables *into* the corpus as retrievable documents. |
| 4 | `scripts/data_prep/build_retrieval_indexes.py` | Takes a final merged corpus and builds two indexes over it: a Lucene/BM25 sparse index, and a ColBERT (PLAID) dense multi-vector index via PyLate, for hybrid or independent top-k retrieval. |

## Directories

**Indexes**
- `/mnt/nas/webtables/row_major_corpus` & `/mnt/nas/webtables/row_major_lucene` — row-major web table corpus + its Lucene index, used to find distractor tables close to BIRD tables when building the augmented corpus.
- `/mnt/nas/webtables/colbert_index2` — ColBERT (PLAID) index, used for independent top-k dense retrieval.
- `/mnt/nas/ayane/tables/lucene_index` — Lucene index used for running sparse-retrieval baselines on BIRD queries.

**Datasets**
- `/mnt/nas/ayane/tables/big_corpus.jsonl` — final ~1M-table corpus (web tables + appended BIRD tables) used for BIRD retrieval experiments.

## Pipeline Commands (in order)

\```bash
# 1. Build the row-major web table corpus + Lucene index
python -m scripts.data_prep.build_row_major_corpus \
    --input_dir /mnt/nas/webtables/extracted \
    --corpus_dir /mnt/nas/webtables/row_major_corpus \
    --index_dir /mnt/nas/webtables/row_major_lucene

# 2. Retrieve web tables similar to BIRD tables + random-fill to target size
python -m scripts.data_prep.build_bird_augmented_corpus \
    --bird_json ./jar/data/bird/dev_tables.json \
    --bird_db_root /mnt/nas/ayane/tables/dev_database \
    --index_dir /mnt/nas/webtables/row_major_lucene/ \
    --corpus_jsonl /mnt/nas/webtables/row_major_corpus/corpus.jsonl \
    --output_jsonl /mnt/nas/ayane/tables/big_corpus.jsonl \
    --top_per_table 10000 \
    --target_size 1000000

# 3. Convert BIRD tables into corpus-schema documents and append them
python -m scripts.data_prep.build_bird_jsonl \
    --bird_json ./jar/data/bird/dev_tables.json \
    --bird_db_root /mnt/nas/ayane/tables/dev_database \
    --output_jsonl /mnt/nas/ayane/tables/bird_tables.jsonl

cat /mnt/nas/ayane/tables/bird_tables.jsonl >> /mnt/nas/ayane/tables/big_corpus.jsonl

# 4. Build final Lucene + ColBERT indexes over the merged corpus
python -m scripts.data_prep.build_retrieval_indexes \
    --corpus_jsonl /mnt/nas/ayane/tables/big_corpus.jsonl \
    --lucene_dir /mnt/nas/ayane/tables/lucene_index \
    --colbert_dir /mnt/nas/webtables/colbert_index2 \
    --threads 16 \
    --batch_size 256
\```

## Open Questions

1. **Query decontextualization** — is there a better way to decontextualize BIRD queries than padding with raw row values? LLM-prompting attempts have not worked well so far.