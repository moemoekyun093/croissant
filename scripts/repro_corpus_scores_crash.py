"""
Full, faithful repro of src/training/trainer.py::FinetuneTrainer._corpus_scores
-- the exact code path that crashes during finetuning's first validation
pass -- without running pretraining, training batches, or the rest of
train_model.py first. Gets to the crash in the time it takes to load the
corpus + build two small models, not minutes into a real run.

This supersedes scripts/repro_bert_batch_crash.py (confirmed: corpus
encoding alone is clean) and scripts/repro_query_encoder_crash.py
(confirmed: needs a run to check) -- this script runs BOTH halves
together, in the exact same order/shapes _corpus_scores itself uses,
so there's no question of whether a piecemeal repro was truly
equivalent to the real thing.

Usage:
    python -m scripts.repro_corpus_scores_crash \\
        --databases_root ../SynSQL-2.5M/databases \\
        --questions_json ../SynSQL-2.5M/questions_with_tables.json \\
        --tables_json ../SynSQL-2.5M/tables.json \\
        --split_json configs/splits/query_split.json \\
        --corpus_json configs/splits/corpus.json \\
        --encoder bert \\
        --val_sample_size 3000 \\
        --val_corpus_sample_size 2000 \\
        --val_n_hard_negatives 2 \\
        --scoring_mode row_match \\
        --device cuda:3
"""

import argparse
import random

from src.data.synsql_dataset import SynSQLQueryDataset, SynSQLTableDataset
from src.encoding.baseline_encoders.adapter import build_baseline_model
from src.scoring.multi_score import MultiScorer
from src.training.losses import cross_score_queries_tables
from src.training.query_encoder import QueryEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", default=None)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--questions_json", required=True)
    parser.add_argument("--split_json", default="configs/splits/query_split.json")
    parser.add_argument("--corpus_json", default="configs/splits/corpus.json")
    parser.add_argument("--encoder", default="bert", choices=["bert", "tapas", "tabbie", "strubert", "turl", "hytrel"])
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--scoring_mode", default="row_match")
    parser.add_argument("--table_batch_size", type=int, default=32)
    parser.add_argument("--val_sample_size", type=int, default=None)
    parser.add_argument("--val_corpus_sample_size", type=int, default=None)
    parser.add_argument("--val_n_hard_negatives", type=int, default=2)
    parser.add_argument("--query_max_length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    print(f"indexing tables from {args.tables_json} / {args.databases_root} ...")
    table_dataset = SynSQLTableDataset(tables_json=args.tables_json, databases_root=args.databases_root)

    query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    print(f"loaded {len(query_dataset)} query -> table example(s)")

    resolved = query_dataset.resolve_split(args.split_json)
    val_indices = resolved["val"]
    if args.val_sample_size is not None and args.val_sample_size < len(val_indices):
        val_indices = random.Random(args.seed).sample(val_indices, args.val_sample_size)
    print(f"val: {len(val_indices)} quer(ies)")

    examples = []
    for i in val_indices:
        ex = query_dataset.examples[i]
        examples.append((ex.question, ex.db_id, ex.table_names))

    print(f"loading corpus from {args.corpus_json} (materialized cache if present) ...")
    corpus_tables = table_dataset.load_corpus(args.corpus_json)
    print(f"corpus: {len(corpus_tables)} table(s)")

    if args.val_corpus_sample_size is not None and args.val_corpus_sample_size < len(corpus_tables):
        positive_ids = {f"{db_id}#sep#{t}" for _q, db_id, table_names in examples for t in table_names}
        forced = [t for t in corpus_tables if t.table_id in positive_ids]
        forced_ids = {t.table_id for t in forced}
        if args.val_n_hard_negatives > 0:
            db_to_tables = {}
            for t in corpus_tables:
                db_to_tables.setdefault(t.table_id.split("#sep#", 1)[0], []).append(t)
            hard_neg_rng = random.Random(args.seed)
            for db_id in {db_id for _q, db_id, _tn in examples}:
                cands = [t for t in db_to_tables.get(db_id, []) if t.table_id not in forced_ids]
                hard_neg_rng.shuffle(cands)
                for t in cands[: args.val_n_hard_negatives]:
                    if t.table_id not in forced_ids:
                        forced.append(t)
                        forced_ids.add(t.table_id)
        remaining = [t for t in corpus_tables if t.table_id not in forced_ids]
        n_fill = max(0, args.val_corpus_sample_size - len(forced))
        filler = random.Random(args.seed).sample(remaining, min(n_fill, len(remaining)))
        corpus_tables = forced + filler
        print(f"subsampled corpus for val: {len(corpus_tables)} table(s)")

    print(f"building {args.encoder} model + QueryEncoder + MultiScorer on {args.device} ...")
    model = build_baseline_model(args.encoder, embed_dim=args.embed_dim, device=args.device)
    query_encoder = QueryEncoder(model_name="bert-base-uncased", output_dim=args.embed_dim, max_length=args.query_max_length, trainable=False)
    if args.device is not None:
        query_encoder = query_encoder.to(args.device)
    scorer = MultiScorer()
    if args.device is not None:
        scorer = scorer.to(args.device)

    model.eval()
    query_encoder.eval()
    scorer.eval()

    corpus_ids = [t.table_id for t in corpus_tables]
    id_to_idx = {tid: i for i, tid in enumerate(corpus_ids)}
    n_queries = len(examples)
    n_corpus = len(corpus_tables)
    print(f"n_queries={n_queries}, n_corpus={n_corpus}")

    order = sorted(range(n_corpus), key=lambda i: (corpus_tables[i].num_columns, corpus_tables[i].num_rows))

    import torch

    with torch.no_grad():
        print("encoding ALL val queries in ONE batched QueryEncoder call ...")
        questions = [q for q, _, _ in examples]
        try:
            Q, query_mask = query_encoder(questions)
        except Exception:
            print("\n=== CRASHED encoding queries -- narrow down with scripts.repro_query_encoder_crash --one_at_a_time ===")
            raise
        Q = Q * query_mask.unsqueeze(-1)
        print(f"Q shape: {tuple(Q.shape)} -- query encoding OK")

        n_chunks = (n_corpus + args.table_batch_size - 1) // args.table_batch_size
        for chunk_idx, c_start in enumerate(range(0, n_corpus, args.table_batch_size)):
            idx_chunk = order[c_start : c_start + args.table_batch_size]
            c_chunk = [corpus_tables[i] for i in idx_chunk]
            print(f"[{chunk_idx + 1}/{n_chunks}] encoding {len(c_chunk)} table(s) + scoring against all {n_queries} quer(ies) ...")
            try:
                X, col_mask, row_mask, cell_mask = model.forward_batch_cellwise(c_chunk)
                cross = cross_score_queries_tables(scorer, args.scoring_mode, Q, X, row_mask, col_mask)
                idx_tensor = torch.tensor(idx_chunk, dtype=torch.long)
                scores = torch.zeros(n_queries, n_corpus)
                scores[0:n_queries].index_copy_(1, idx_tensor, cross.to(scores.device))
            except Exception:
                print(f"\n=== CRASHED on chunk {chunk_idx + 1}/{n_chunks} -- table_id / shape for each table in it: ===")
                for t in c_chunk:
                    print(f"  {t.table_id!r}: {t.num_rows} rows x {t.num_columns} cols")
                raise

    print("\nNo crash -- full _corpus_scores logic replayed successfully against the entire corpus.")
