"""
Standalone pre-materialization step for the fixed retrieval corpus built
by scripts/build_query_splits.py.

The corpus (configs/splits/corpus.json by default) only lists which
(db_id, table_name) pairs belong to it -- SynSQLTableDataset.load_corpus()
turns that into real Table objects (actual header/cell values) by
querying each table's SQLite database live. Since the corpus is FIXED
(never re-split, identical across every model/run), doing that live read
inside every train_model.py/finetune_query_table.py invocation is pure
waste after the first time -- run this script ONCE instead, ahead of
time, and every future training run will find the cached
corpus.materialized.json already there and load it in seconds instead of
minutes (168,241 tables took ~14 minutes of live SQLite reads on a real
SynSQL-2.5M-scale run; the cached JSON load is near-instant).

This is the exact same caching load_corpus() already does automatically
on first use -- this script just lets you pay that one-time cost
up front, deliberately, before kicking off a long training run (or the
full scripts/run_all_models.sh sweep, where every one of the 7 encoders
would otherwise redundantly trigger it independently).

Usage:
    python -m scripts.materialize_corpus \\
        --databases_root ../SynSQL-2.5M/databases \\
        --tables_json ../SynSQL-2.5M/tables.json \\
        --corpus_json configs/splits/corpus.json

    # custom cache location (default: corpus_json with a
    # ".materialized.json" suffix, e.g. configs/splits/corpus.materialized.json):
    python -m scripts.materialize_corpus \\
        --databases_root ../SynSQL-2.5M/databases \\
        --corpus_json configs/splits/corpus.json \\
        --materialized_cache_path /fast/local/disk/corpus.materialized.json

Safe to re-run: if the cache file already exists, this just confirms
that and exits immediately (matches load_corpus()'s own behavior) --
delete the cache file first if the underlying databases changed and you
need a fresh read.
"""

import argparse
import os

from src.data.synsql_dataset import SynSQLTableDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", default=None)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--corpus_json", default="configs/splits/corpus.json")
    parser.add_argument(
        "--materialized_cache_path", default=None,
        help="defaults to --corpus_json with a '.materialized.json' suffix, "
             "same convention as SynSQLTableDataset.load_corpus()",
    )
    parser.add_argument("--max_rows", type=int, default=50)
    args = parser.parse_args()

    cache_path = args.materialized_cache_path
    if cache_path is None:
        base, _ = os.path.splitext(args.corpus_json)
        cache_path = f"{base}.materialized.json"

    if os.path.exists(cache_path):
        print(f"{cache_path!r} already exists -- nothing to do.")
        print("(delete it first if the underlying databases changed and you need a fresh read)")
        raise SystemExit(0)

    print(f"indexing tables from {args.tables_json} / {args.databases_root} ...")
    table_dataset = SynSQLTableDataset(
        tables_json=args.tables_json,
        databases_root=args.databases_root,
        max_rows=args.max_rows,
    )

    # Does the live SQLite read (with progress logging) AND writes the
    # cache file -- see SynSQLTableDataset.load_corpus's docstring.
    tables = table_dataset.load_corpus(args.corpus_json, materialized_cache_path=cache_path)

    print(f"\nDone -- materialized {len(tables)} table(s) to {cache_path!r}.")
    print("Future train_model.py / finetune_query_table.py runs pointed at the same")
    print("--corpus_json will now load this cache instead of re-querying SQLite.")
