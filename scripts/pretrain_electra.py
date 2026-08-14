"""
ELECTRA-style pretraining: cell corruption + per-cell discriminator.

Usage:
    python -m scripts.pretrain_electra \
        --tables_json /path/to/synsql/tables.json \
        --databases_root /path/to/synsql/databases \
        --embed_dim 64 \
        --batch_size 64 \
        --device cuda:2

Data source is SynSQLTableDataset (src/data/synsql_dataset.py) -- real
cell values pulled live from SQLite per table, not a precomputed JSONL
corpus. See that module's docstring for the assumed tables.json /
databases/ layout.
"""

import argparse
import random

from src.data.synsql_dataset import SynSQLTableDataset
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import DiscriminatorHead, TableEncoder
from src.training.trainer import PretrainTrainer


def make_batches(tables, batch_size, max_columns=20):
    """Same bucket-by-size + truncate-wide-tables strategy as
    scripts/pilot_train.py::make_batches -- keeps padding cost down."""
    from src.data.table import Table

    capped = []
    for t in tables:
        if len(t.columns) > max_columns:
            t = Table(table_id=t.table_id, table_name=t.table_name, columns=t.columns[:max_columns])
        capped.append(t)

    capped.sort(key=lambda t: (t.num_columns, t.num_rows))
    batches = [capped[i : i + batch_size] for i in range(0, len(capped), batch_size)]
    batches = [b for b in batches if len(b) >= 2]
    random.shuffle(batches)
    return batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", required=True)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--max_rows", type=int, default=50)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--embed_dim", type=int, default=64, help="must be even (see CellEncoder)")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--corrupt_frac", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--checkpoint_dir", default="eval/report_runs/pretrain")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--resume_from", default=None)
    args = parser.parse_args()

    print(f"indexing tables from {args.tables_json} / {args.databases_root} ...")
    table_dataset = SynSQLTableDataset(
        tables_json=args.tables_json,
        databases_root=args.databases_root,
        max_rows=args.max_rows,
    )
    print(f"found {len(table_dataset)} tables")

    tables = list(table_dataset.iter_tables())
    random.shuffle(tables)
    n_val = max(1, int(len(tables) * args.val_frac))
    val_tables, train_tables = tables[:n_val], tables[n_val:]
    print(f"split: {len(train_tables)} train / {len(val_tables)} held-out val")

    batches = make_batches(train_tables, args.batch_size)
    val_batches = make_batches(val_tables, args.batch_size)
    print(f"{len(batches)} train batches/epoch, {len(val_batches)} val batches")

    cell_encoder = CellEncoder(text_model_name="bert-base-uncased", output_dim=args.embed_dim)
    model = TableEncoder(cell_encoder, embed_dim=args.embed_dim, num_layers=args.num_layers)
    discriminator = DiscriminatorHead(embed_dim=args.embed_dim)

    trainer = PretrainTrainer(
        model,
        discriminator,
        lr=args.lr,
        corrupt_frac=args.corrupt_frac,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    print(f"starting ELECTRA pretraining on {args.device} ...")
    trainer.fit(
        batches,
        num_epochs=args.num_epochs,
        steps_per_epoch=len(batches),
        log_every=args.log_every,
        val_batches=val_batches,
        resume_from=args.resume_from,
    )

    print("\nPretraining complete.")
