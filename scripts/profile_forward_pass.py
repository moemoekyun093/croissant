"""
Measures how much of a forward pass is spent on cell encoding (BERT +
numeric embedder) versus the custom table-level layers (ColumnAggregator
/ CrossColumnAttention / ChannelMix / RowCollapse) -- run this before
deciding whether to stack more table-level layers, so the decision is
based on a real measurement instead of an inference from indirect
timing evidence.

Usage:
    python -m scripts.profile_forward_pass \
        --corpus_jsonl /mnt/nas/ayane/tables/big_corpus.jsonl \
        --checkpoint eval/report_runs/run_full/checkpoint_epoch2.pt \
        --embed_dim 64 --device cuda:2 --batch_size 64 --n_runs 3
"""

import argparse

import torch

from src.data.corpus_loader import iter_tables_from_jsonl
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import TableEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_jsonl", required=True)
    parser.add_argument("--checkpoint", default=None, help="optional -- timing doesn't depend on weights")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_runs", type=int, default=3, help="repeat runs -- first one includes CUDA warmup overhead, ignore it")
    args = parser.parse_args()

    print(f"loading {args.batch_size} real tables from corpus...")
    tables = []
    for t in iter_tables_from_jsonl(args.corpus_jsonl):
        tables.append(t)
        if len(tables) >= args.batch_size:
            break
    print(f"loaded {len(tables)} tables")

    cell_encoder = CellEncoder(text_model_name="bert-base-uncased", output_dim=args.embed_dim)
    model = TableEncoder(cell_encoder, embed_dim=args.embed_dim, num_layers=args.num_layers).to(args.device)

    if args.checkpoint:
        print(f"loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print(f"\nrunning {args.n_runs} profiled forward passes (ignore the first -- CUDA warmup):\n")
    with torch.no_grad():
        for i in range(args.n_runs):
            print(f"run {i+1}/{args.n_runs}:")
            model.forward_batch(tables, profile=True)
            print()