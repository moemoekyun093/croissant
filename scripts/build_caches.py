"""
Builds every disk cache this pipeline can use, once, up front -- so a
real train_model.py run (or any repro script) starts warm instead of
paying these costs on its own first launch.

Caches built, in order (each step's docstring in the underlying code has
the full rationale -- this script is just a thin driver):

  1. schema cache        (SynSQLTableDataset)  -- db_id -> table_names,
                          read from live sqlite_master once per db_id.
  2. resolved examples   (SynSQLQueryDataset)  -- the fully filtered/
     cache                                        validated (question,
                          db_id, table_names) list, so questions_with_
                          tables.json never needs re-parsing.
  3. materialized corpus (SynSQLTableDataset.load_corpus) -- real row
                          data for every corpus table, so no live SQL
                          reads are needed to re-materialize the corpus.
  4. table-embedding     (BaselineCellwiseAdapter, --encoder bert/tapas
     cache                  only) -- BERT's actual output vectors for
                          every corpus table, so the backbone never has
                          to run again for a table already seen. Skipped
                          entirely for --encoder ours/tabbie/strubert/
                          turl/hytrel (see adapter.py's
                          _FULLY_FROZEN_ENCODERS -- only bert/tapas have
                          a fully-frozen, cacheable-forever encoder).

Steps 1-2 happen automatically just by constructing the datasets (no
extra code needed -- this script's value for those two is purely "run
this once, up front, and see the print statements confirming it
worked" instead of discovering it mid-training-run). Steps 3-4 are
explicit calls this script makes on your behalf.

Usage:
    python -m scripts.build_caches \\
        --databases_root ../SynSQL-2.5M/databases \\
        --questions_json ../SynSQL-2.5M/questions_with_tables.json \\
        --tables_json ../SynSQL-2.5M/tables.json \\
        --split_json configs/splits/query_split.json \\
        --corpus_json configs/splits/corpus.json \\
        --encoder bert \\
        --device cpu

    # skip the (slow, real-BERT-forward-pass) table-embedding cache and
    # only build the fast JSON caches (schema, resolved examples, corpus):
    python -m scripts.build_caches ... --skip_table_cache

Safe to re-run any time -- every step here is itself cache-aware (a
cache file that already exists is loaded, not rebuilt), so running this
again after it already succeeded once is fast and a no-op past step 1.
"""

import argparse
import os

from src.data.synsql_dataset import SynSQLQueryDataset, SynSQLTableDataset
from src.encoding.baseline_encoders.adapter import _FULLY_FROZEN_ENCODERS, build_baseline_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", default=None)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--questions_json", required=True)
    parser.add_argument("--split_json", default="configs/splits/query_split.json")
    parser.add_argument("--corpus_json", default="configs/splits/corpus.json")
    parser.add_argument("--max_rows", type=int, default=50)
    parser.add_argument(
        "--encoder", default=None, choices=["bert", "tapas", "tabbie", "strubert", "turl", "hytrel", "ours"],
        help="only needed if you also want the table-embedding cache built (step 4) -- omit to build just the JSON caches (steps 1-3)",
    )
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--table_batch_size", type=int, default=32)
    parser.add_argument("--table_cache_path", default=None)
    parser.add_argument("--skip_table_cache", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("=== step 1-2: schema + resolved-examples caches (SynSQLTableDataset / SynSQLQueryDataset) ===")
    table_dataset = SynSQLTableDataset(
        databases_root=args.databases_root,
        tables_json=args.tables_json,
        max_rows=args.max_rows,
    )
    query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    print(f"-> {len(query_dataset)} resolved example(s), {len(table_dataset._table_names_cache)} database schema(s) cached\n")

    print("=== step 3: materialized corpus cache (SynSQLTableDataset.load_corpus) ===")
    corpus_tables = table_dataset.load_corpus(args.corpus_json)
    print(f"-> {len(corpus_tables)} corpus table(s) materialized\n")

    if args.skip_table_cache or args.encoder is None:
        print("skipping step 4 (table-embedding cache) -- pass --encoder bert (or tapas) to build it.")
        raise SystemExit(0)

    if args.encoder not in _FULLY_FROZEN_ENCODERS:
        print(
            f"=== step 4: skipped -- --encoder {args.encoder!r} is not in "
            f"_FULLY_FROZEN_ENCODERS ({sorted(_FULLY_FROZEN_ENCODERS)}). "
            f"Its table embeddings depend on trainable weights that change "
            f"every training step, so they can't be cached once and reused "
            f"across a whole run -- see adapter.py's cacheable docstring. ==="
        )
        raise SystemExit(0)

    print(f"=== step 4: table-embedding cache ({args.encoder}, {len(corpus_tables)} table(s), device={args.device}) ===")
    table_cache_path = args.table_cache_path or os.path.join("eval/report_runs", args.encoder, "table_cache.pt")

    model = build_baseline_model(args.encoder, embed_dim=args.embed_dim, device=args.device)
    if os.path.exists(table_cache_path):
        print(f"found existing table cache at {table_cache_path!r} -- loading and topping up any missing tables ...")
        model.load_table_cache(table_cache_path)
        print(f"loaded {len(model._table_cache)} entries")

    model.eval()

    n_before = len(model._table_cache)

    # Same size-sort _corpus_scores uses (largest tables last), purely so
    # progress printing is informative -- doesn't affect correctness.
    order = sorted(range(len(corpus_tables)), key=lambda i: (corpus_tables[i].num_columns, corpus_tables[i].num_rows))
    n_chunks = (len(order) + args.table_batch_size - 1) // args.table_batch_size

    import torch

    with torch.no_grad():
        for chunk_idx, c_start in enumerate(range(0, len(order), args.table_batch_size)):
            idx_chunk = order[c_start : c_start + args.table_batch_size]
            c_chunk = [corpus_tables[i] for i in idx_chunk]
            n_already_cached = sum(1 for t in c_chunk if t.table_id in model._table_cache)
            if n_already_cached == len(c_chunk):
                continue  # every table in this chunk is already cached -- skip the backbone call entirely
            print(f"[{chunk_idx + 1}/{n_chunks}] encoding {len(c_chunk) - n_already_cached}/{len(c_chunk)} new table(s) ...")
            model.forward_batch_cellwise(c_chunk)

    n_new = len(model._table_cache) - n_before
    if n_new == 0 and os.path.exists(table_cache_path):
        print(f"-> every corpus table was already cached ({n_before} entries) -- {table_cache_path} left untouched.")
    else:
        os.makedirs(os.path.dirname(table_cache_path) or ".", exist_ok=True)
        model.save_table_cache(table_cache_path)
        print(f"-> encoded {n_new} new table(s), saved {len(model._table_cache)} total entries to {table_cache_path}")
