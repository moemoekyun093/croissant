"""
Minimal, fast repro for the CUDA "index out of bounds" crash seen during
finetuning's first validation pass on --encoder bert.

The crash happens entirely inside BaselineCellwiseAdapter.forward_batch_cellwise
being called on corpus tables (src/training/trainer.py's _corpus_scores) --
nothing about pretraining, queries, or the training loop is involved. So
instead of running the full train_model.py pipeline (pretrain + finetune +
train-batch construction, several minutes before validation ever runs),
this script loads ONLY the corpus and replays the exact same chunking
_corpus_scores uses, directly, against a freshly-built bert model -- no
queries, no training loop, no pretraining. Should hit the same crash (if
it's real and data-dependent) in seconds, not minutes.

With this session's defensive checks now in bert_baseline.py/
tapas_encoder.py's forward_batch (raise a clear ValueError naming the
offending table BEFORE the backbone call, instead of letting an
out-of-range id/position reach the GPU), this should now either:
  (a) print a clear ValueError identifying exactly which corpus table
      and what's wrong with it, or
  (b) run cleanly through the whole corpus with no crash at all --
      meaning the original bug either wasn't in bert_baseline.py's
      forward_batch, or has already been fixed by those checks forcing
      correct behavior upstream (unlikely -- the checks raise, they
      don't silently fix anything -- so (b) really means "no bad table
      found", pointing elsewhere).

Usage:
    python -m scripts.repro_bert_batch_crash \\
        --databases_root ../SynSQL-2.5M/databases \\
        --tables_json ../SynSQL-2.5M/tables.json \\
        --corpus_json configs/splits/corpus.json \\
        --device cuda:3

    # if the corpus is huge and you just want to hit the crash fast,
    # cap how many tables get tried (still in the corpus's own
    # size-sorted chunk order, same as real validation):
    python -m scripts.repro_bert_batch_crash ... --max_tables 2000
"""

import argparse

from src.data.synsql_dataset import SynSQLTableDataset
from src.encoding.baseline_encoders.adapter import build_baseline_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", default=None)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--corpus_json", default="configs/splits/corpus.json")
    parser.add_argument("--encoder", default="bert", choices=["bert", "tapas"])
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--table_batch_size", type=int, default=32, help="same default as trainer.py's _corpus_scores")
    parser.add_argument("--max_tables", type=int, default=None, help="stop after this many corpus tables (in size-sorted order) -- omit to run the whole corpus")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    print(f"indexing tables from {args.tables_json} / {args.databases_root} ...")
    table_dataset = SynSQLTableDataset(
        tables_json=args.tables_json,
        databases_root=args.databases_root,
    )

    print(f"loading corpus from {args.corpus_json} (materialized cache if present) ...")
    corpus_tables = table_dataset.load_corpus(args.corpus_json)
    print(f"corpus: {len(corpus_tables)} table(s)")

    # SAME size-sort _corpus_scores uses, so chunk composition here
    # matches real validation exactly (this matters if the bug is
    # triggered by a specific table landing in a specific chunk
    # alongside specific other tables, e.g. a padding/maxlen interaction).
    order = sorted(range(len(corpus_tables)), key=lambda i: (corpus_tables[i].num_columns, corpus_tables[i].num_rows))
    if args.max_tables is not None:
        order = order[: args.max_tables]
        print(f"capped to first {len(order)} table(s) in size-sorted order")

    print(f"building {args.encoder} model on {args.device} ...")
    model = build_baseline_model(args.encoder, embed_dim=args.embed_dim, device=args.device)

    n_chunks = (len(order) + args.table_batch_size - 1) // args.table_batch_size
    for chunk_idx, c_start in enumerate(range(0, len(order), args.table_batch_size)):
        idx_chunk = order[c_start : c_start + args.table_batch_size]
        c_chunk = [corpus_tables[i] for i in idx_chunk]
        shapes = [(t.num_rows, t.num_columns) for t in c_chunk]
        print(
            f"[{chunk_idx + 1}/{n_chunks}] encoding {len(c_chunk)} table(s) "
            f"(rows,cols range: {min(shapes)}..{max(shapes)}) ..."
        )
        try:
            model.forward_batch_cellwise(c_chunk)
        except Exception:
            print("\n=== CRASHED on this chunk -- table_id / shape for each table in it: ===")
            for t in c_chunk:
                print(f"  {t.table_id!r}: {t.num_rows} rows x {t.num_columns} cols")
            raise

    print("\nNo crash -- every corpus table encoded successfully.")
