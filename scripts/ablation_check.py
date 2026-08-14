"""
Header vs. content ablation: loads a trained checkpoint and measures
held-out loss under three conditions --

    full:          normal forward pass (headers + cell content)
    headers_only:  cell content zeroed -- isolates what header text alone
                   contributes
    content_only:  headers zeroed -- isolates what cell content alone
                   contributes

Comparing the three tells you whether the expensive cell-content
pipeline (BERT, numeric embedder, per-channel attention) is adding real
value beyond header matching, or whether the model has learned to lean
almost entirely on one signal.

Usage:
    python -m scripts.ablation_check \
        --corpus_jsonl /path/to/corpus.jsonl \
        --checkpoint eval/report_runs/pilot/checkpoint_epoch14.pt \
        --n_tables 10000 --embed_dim 64

Note: this rebuilds a fresh random val split (seeded) rather than
reusing pilot_train.py's exact held-out set, since the val table IDs
weren't persisted to disk during training. Close enough to answer this
question, but worth fixing (save val table IDs alongside checkpoints)
if you need the literal same held-out set for later comparisons.
"""

import argparse
import random

import torch

from src.data.augmentation import augment_table
from src.data.corpus_loader import iter_tables_from_jsonl
from src.data.table import Table
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import TableEncoder
from src.training.losses import info_nce_loss


def load_tables(corpus_jsonl: str, n_tables: int) -> list[Table]:
    tables = []
    for t in iter_tables_from_jsonl(corpus_jsonl):
        tables.append(t)
        if len(tables) >= n_tables:
            break
    return tables


def make_batches(tables: list[Table], batch_size: int, max_columns: int = 20) -> list[list[Table]]:
    capped = []
    for t in tables:
        if len(t.columns) > max_columns:
            t = Table(table_id=t.table_id, table_name=t.table_name, columns=t.columns[:max_columns])
        capped.append(t)
    capped.sort(key=lambda t: (t.num_columns, t.num_rows))
    batches = [capped[i:i+batch_size] for i in range(0, len(capped), batch_size)]
    return [b for b in batches if len(b) >= 2]


@torch.no_grad()
def eval_ablation(model, val_batches, ablation, temperature=0.07):
    model.eval()
    losses = []
    for batch in val_batches:
        augmented = [augment_table(t, 0.7, 0.7) for t in batch]
        B = len(batch)
        all_tables = batch + augmented
        X, mask = model.forward_batch(all_tables, ablation=ablation)
        loss = info_nce_loss(X, mask, B, temperature=temperature)
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_tables", type=int, default=10_000)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1, help="must match the checkpoint being loaded")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"loading up to {args.n_tables} tables ...")
    tables = load_tables(args.corpus_jsonl, args.n_tables)
    random.shuffle(tables)
    n_val = max(1, int(len(tables) * args.val_frac))
    val_tables = tables[:n_val]
    print(f"held-out val set: {len(val_tables)} tables")

    val_batches = make_batches(val_tables, args.batch_size)
    print(f"{len(val_batches)} val batches")

    cell_encoder = CellEncoder(text_model_name="bert-base-uncased", output_dim=args.embed_dim)
    model = TableEncoder(cell_encoder, embed_dim=args.embed_dim, num_layers=args.num_layers)

    print(f"loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)

    results = {}
    for ablation, label in [
        (None, "full (headers + content)"),
        ("headers_only", "headers_only (content zeroed)"),
        ("content_only", "content_only (headers zeroed)"),
    ]:
        loss = eval_ablation(model, val_batches, ablation)
        results[label] = loss
        print(f"  {label}: val loss = {loss:.4f}")

    print("\n== summary ==")
    for label, loss in results.items():
        print(f"  {label}: {loss:.4f}")

    full = results["full (headers + content)"]
    headers_only = results["headers_only (content zeroed)"]
    content_only = results["content_only (headers zeroed)"]

    print(f"\nheaders_only vs full: {'similar' if abs(headers_only - full) < 0.1 else 'different'} "
          f"(diff={headers_only - full:+.4f})")
    print(f"content_only vs full: {'similar' if abs(content_only - full) < 0.1 else 'different'} "
          f"(diff={content_only - full:+.4f})")