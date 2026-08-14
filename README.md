# Web Table Retrieval Pipeline — Overview

## Scripts

| # | File | Purpose |
|---|------|---------|
| 1 | `build_row_major_corpus.py` | Parses raw web table dumps (`.json.gz`, WDC-style `relation` format) into a flat, row-major JSONL corpus (`[TABLE]/[SCHEMA]/[ROWS]` text format), then builds a Lucene/BM25 index over it. |
| 2 | `build_bird_augmented_corpus.py` | Uses each BIRD table (name + columns + sampled SQLite row values) as a query against the row-major Lucene index, retrieves the top-k most similar web tables per BIRD table, deduplicates into a selected set, and pads with random web tables up to a target size. Writes the filtered/augmented corpus. **Does not include the BIRD tables themselves** — output is web tables only (`source: "webtable"`). |
| 3 | `build_bird_jsonl.py` | Converts the actual BIRD tables (name, columns, sampled row values) into documents in the same corpus schema, tagged `source: "bird"`, with IDs offset from a fixed `START_ID` to avoid collisions with web-table IDs. This is the step that gets BIRD tables *into* the corpus as retrievable documents. |
| 4 | `build_retrieval_indexes.py` | Takes a final merged corpus and builds two indexes over it: a Lucene/BM25 sparse index, and a ColBERT (PLAID) dense multi-vector index via PyLate, for hybrid or independent top-k retrieval. |

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
python build_row_major_corpus.py \
    --input_dir /mnt/nas/webtables/extracted \
    --corpus_dir /mnt/nas/webtables/row_major_corpus \
    --index_dir /mnt/nas/webtables/row_major_lucene

# 2. Retrieve web tables similar to BIRD tables + random-fill to target size
python build_bird_augmented_corpus.py \
    --bird_json ./jar/data/bird/dev_tables.json \
    --bird_db_root /mnt/nas/ayane/tables/dev_database \
    --index_dir /mnt/nas/webtables/row_major_lucene/ \
    --corpus_jsonl /mnt/nas/webtables/row_major_corpus/corpus.jsonl \
    --output_jsonl /mnt/nas/ayane/tables/big_corpus.jsonl \
    --top_per_table 10000 \
    --target_size 1000000

# 3. Convert BIRD tables into corpus-schema documents and append them
python build_bird_jsonl.py \
    --bird_json ./jar/data/bird/dev_tables.json \
    --bird_db_root /mnt/nas/ayane/tables/dev_database \
    --output_jsonl /mnt/nas/ayane/tables/bird_tables.jsonl

cat /mnt/nas/ayane/tables/bird_tables.jsonl >> /mnt/nas/ayane/tables/big_corpus.jsonl

# 4. Build final Lucene + ColBERT indexes over the merged corpus
python build_retrieval_indexes.py \
    --corpus_jsonl /mnt/nas/ayane/tables/big_corpus.jsonl \
    --lucene_dir /mnt/nas/ayane/tables/lucene_index \
    --colbert_dir /mnt/nas/webtables/colbert_index2 \
    --threads 16 \
    --batch_size 256
\```

## Open Questions

1. **Query decontextualization** — is there a better way to decontextualize BIRD queries than padding with raw row values? LLM-prompting attempts have not worked well so far.