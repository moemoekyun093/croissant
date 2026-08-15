"""
Micro-benchmark isolating where time actually goes in a bert finetuning
step, at small batch sizes -- query encoding vs. raw BERT table encoding
vs. the full adapter forward pass (caching + cell_mask scatter on top of
the raw encode) vs. scoring.

Motivation: trainer.py's _score_batch does query encode -> table encode ->
scoring every step, but a slow step could come from any of those three,
or from adapter overhead ON TOP of the raw BERT call itself (caching
lookups, cell_mask construction) rather than the BERT forward pass
proper. This isolates each piece directly instead of guessing from
overall step time.

Runs each stage `--repeats` times per batch size (after `--warmup`
untimed repeats) and reports mean/median ms per stage, both total and
per-item, for each --batch_sizes value -- small batches specifically,
since that's where fixed overhead (Python loops, tokenizer calls,
adapter bookkeeping) matters most relative to actual compute.

Usage:
    python -m scripts.profile_bert_stages \\
        --databases_root ../SynSQL-2.5M/databases \\
        --questions_json ../SynSQL-2.5M/questions_with_tables.json \\
        --tables_json ../SynSQL-2.5M/tables.json \\
        --split_json configs/splits/query_split.json \\
        --corpus_json configs/splits/corpus.json \\
        --encoder bert \\
        --batch_sizes 1,4,8,16,32,64 \\
        --repeats 10 \\
        --device cuda:3
"""

import argparse
import random
import statistics
import time

import torch

from src.data.synsql_dataset import SynSQLQueryDataset, SynSQLTableDataset
from src.encoding.baseline_encoders.adapter import build_baseline_model
from src.scoring.multi_score import MultiScorer
from src.training.losses import cross_score_queries_tables
from src.training.query_encoder import QueryEncoder


def _timed(fn, repeats: int, warmup: int, device: str):
    for _ in range(warmup):
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times, out


def _fmt(times_s, n_items):
    ms = [t * 1000 for t in times_s]
    mean_ms = statistics.mean(ms)
    median_ms = statistics.median(ms)
    per_item = mean_ms / max(1, n_items)
    return f"mean {mean_ms:7.1f}ms  median {median_ms:7.1f}ms  ({per_item:6.2f}ms/item)"


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
    parser.add_argument("--batch_sizes", default="1,4,8,16,32,64")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--query_max_length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(f"indexing tables from {args.tables_json} / {args.databases_root} ...")
    table_dataset = SynSQLTableDataset(tables_json=args.tables_json, databases_root=args.databases_root)
    query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    resolved = query_dataset.resolve_split(args.split_json)

    max_bs = max(batch_sizes)
    train_indices = random.Random(args.seed).sample(resolved["train"], min(max_bs, len(resolved["train"])))
    questions_pool = [query_dataset.examples[i].question for i in train_indices]

    print(f"loading corpus from {args.corpus_json} ...")
    corpus_tables = table_dataset.load_corpus(args.corpus_json)
    tables_pool = random.Random(args.seed).sample(corpus_tables, min(max_bs, len(corpus_tables)))
    print(f"pool sizes -- questions: {len(questions_pool)}, tables: {len(tables_pool)}")

    print(f"building {args.encoder} model + QueryEncoder + MultiScorer on {args.device} ...")
    model = build_baseline_model(args.encoder, embed_dim=args.embed_dim, device=args.device)
    query_encoder = QueryEncoder(model_name="bert-base-uncased", output_dim=args.embed_dim, max_length=args.query_max_length, trainable=False).to(args.device)
    scorer = MultiScorer().to(args.device)
    model.eval()
    query_encoder.eval()
    scorer.eval()

    baseline_encoder = getattr(model, "baseline_encoder", None)
    has_raw_forward_batch = baseline_encoder is not None and hasattr(baseline_encoder, "forward_batch")

    header = (
        f"{'batch':>6} | {'query encode (QueryEncoder)':<38} | "
        f"{'raw backbone encode (forward_batch)':<38} | "
        f"{'full adapter encode (forward_batch_cellwise)':<46} | "
        f"{'scoring':<32}"
    )
    print("\n" + header)
    print("-" * len(header))

    with torch.no_grad():
        for bs in batch_sizes:
            questions = (questions_pool * ((bs // max(1, len(questions_pool))) + 1))[:bs]
            tables = (tables_pool * ((bs // max(1, len(tables_pool))) + 1))[:bs]
            # Table is column-major (list[Column]); baseline_encoder.forward_batch
            # wants (headers, rows, caption) tuples, same conversion
            # adapter.py's forward_batch_cellwise itself uses.
            tables_as_tuples = [(*model._table_to_headers_rows(t), None) for t in tables]

            q_times, (Q, query_mask) = _timed(lambda: query_encoder(questions), args.repeats, args.warmup, args.device)
            Q = Q * query_mask.unsqueeze(-1)

            if has_raw_forward_batch:
                raw_times, _ = _timed(lambda: baseline_encoder.forward_batch(tables_as_tuples), args.repeats, args.warmup, args.device)
                raw_str = _fmt(raw_times, bs)
            else:
                raw_str = "n/a (no forward_batch on this encoder)"

            full_times, (X, col_mask, row_mask, cell_mask) = _timed(lambda: model.forward_batch_cellwise(tables), args.repeats, args.warmup, args.device)

            score_times, _ = _timed(lambda: cross_score_queries_tables(scorer, args.scoring_mode, Q, X, row_mask, col_mask), args.repeats, args.warmup, args.device)

            print(
                f"{bs:>6} | {_fmt(q_times, bs):<38} | {raw_str:<38} | "
                f"{_fmt(full_times, bs):<46} | {_fmt(score_times, bs):<32}"
            )

    print(
        "\nnote: 'full adapter encode' includes cache lookups + cell_mask "
        "scatter on top of the same raw backbone call timed separately -- "
        "a big gap between the two at small batch sizes points at adapter "
        "bookkeeping overhead (Python loops, tokenizer calls) rather than "
        "the BERT forward pass itself as the bottleneck. Cache is cold on "
        "the first repeat within warmup and stays warm for the rest, so "
        "these numbers mostly reflect cache-HIT cost after warmup -- "
        "rerun with --warmup 0 to see cold-cache cost instead."
    )
