"""
Minimal, fast repro targeting the OTHER half of _corpus_scores that
scripts/repro_bert_batch_crash.py didn't test: query encoding.

scripts/repro_bert_batch_crash.py confirmed the corpus/table-encoding
side of the pipeline (BaselineCellwiseAdapter.forward_batch_cellwise,
including bert_baseline.py's forward_batch and the cell_mask fix in
adapter.py) runs cleanly across the ENTIRE real corpus with no crash.
src/scoring/multi_score.py was also checked by hand -- every op there is
einsum/broadcast/masked-reduction, no advanced/fancy indexing at all
(which is specifically what the CUDA IndexKernel.cu "index out of
bounds" assert comes from), so it's very unlikely to be the source
either.

That leaves the one thing neither of those covers:
src/training/trainer.py's _corpus_scores calls
`Q, query_mask = self.query_encoder(questions)` ONCE, encoding EVERY
validation query in a single batched QueryEncoder forward pass, before
ever touching the corpus. This script replays exactly that call in
isolation -- loads the real query split, resolves val (or --split
train/test), and runs QueryEncoder over those questions directly. No
table encoding, no scoring, no training loop.

Usage:
    python -m scripts.repro_query_encoder_crash \\
        --databases_root ../SynSQL-2.5M/databases \\
        --questions_json ../SynSQL-2.5M/questions_with_tables.json \\
        --tables_json ../SynSQL-2.5M/tables.json \\
        --split_json configs/splits/query_split.json \\
        --val_sample_size 3000 \\
        --device cuda:3
"""

import argparse
import random

from src.data.synsql_dataset import SynSQLQueryDataset, SynSQLTableDataset
from src.training.query_encoder import QueryEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", default=None)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--questions_json", required=True)
    parser.add_argument("--split_json", default="configs/splits/query_split.json")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--val_sample_size", type=int, default=None, help="subsample like train_model.py's --val_sample_size, fixed via --seed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query_model_name", default=None)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--query_max_length", type=int, default=32)
    parser.add_argument("--query_trainable", action="store_true", default=False)
    parser.add_argument("--device", default=None)
    parser.add_argument("--one_at_a_time", action="store_true", help="encode each question SEPARATELY instead of one big batch, to isolate which specific question (if any) is the problem")
    args = parser.parse_args()

    print(f"indexing tables from {args.tables_json} / {args.databases_root} ...")
    table_dataset = SynSQLTableDataset(
        tables_json=args.tables_json,
        databases_root=args.databases_root,
    )

    query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    print(f"loaded {len(query_dataset)} query -> table example(s)")

    resolved = query_dataset.resolve_split(args.split_json)
    indices = resolved[args.split]
    print(f"{args.split}: {len(indices)} quer(ies)")

    if args.val_sample_size is not None and args.val_sample_size < len(indices):
        indices = random.Random(args.seed).sample(indices, args.val_sample_size)
        print(f"subsampled to {len(indices)} quer(ies) (--val_sample_size {args.val_sample_size})")

    questions = [query_dataset.examples[i].question for i in indices]

    kwargs = {}
    if args.query_model_name is not None:
        kwargs["model_name"] = args.query_model_name
    query_encoder = QueryEncoder(
        model_name=args.query_model_name or "bert-base-uncased",
        output_dim=args.embed_dim,
        max_length=args.query_max_length,
        trainable=args.query_trainable,
    )
    if args.device is not None:
        query_encoder = query_encoder.to(args.device)

    if args.one_at_a_time:
        print(f"encoding {len(questions)} question(s) ONE AT A TIME ...")
        for i, q in enumerate(questions):
            try:
                query_encoder([q])
            except Exception:
                print(f"\n=== CRASHED on question index {i} ===")
                print(f"question: {q!r}")
                print(f"length (chars): {len(q)}")
                raise
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(questions)} OK")
        print("\nNo crash -- every question encoded successfully, one at a time.")
    else:
        print(f"encoding {len(questions)} question(s) in ONE batched call (matches _corpus_scores' own query_batch_size=None default) ...")
        try:
            query_encoder(questions)
        except Exception:
            print("\n=== CRASHED -- re-run with --one_at_a_time to find the specific offending question ===")
            raise
        print("\nNo crash -- all questions encoded successfully in one batch.")
