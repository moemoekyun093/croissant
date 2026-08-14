"""
Finetuning: real query -> positive-table contrastive training, starting
from an ELECTRA-pretrained encoder checkpoint.

Usage:
    python -m scripts.finetune_query_table \
        --tables_json /path/to/synsql/tables.json \
        --databases_root /path/to/synsql/databases \
        --questions_json /path/to/synsql/questions_with_tables.json \
        --pretrained_checkpoint eval/report_runs/pretrain/checkpoint_epoch14.pt \
        --embed_dim 64 \
        --batch_size 32 \
        --device cuda:2

Loads the pretrained TableEncoder via load_pretrained_encoder(), which
discards any discriminator-head weights from that checkpoint -- the
discriminator was only ever needed for the pretraining task.
"""

import argparse
import random

from src.data.synsql_dataset import SynSQLQueryDataset, SynSQLTableDataset
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import TableEncoder, load_pretrained_encoder
from src.training.query_encoder import QueryEncoder
from src.training.trainer import FinetuneTrainer


def resolve_batches(query_dataset: SynSQLQueryDataset, batch_size: int, rng: random.Random):
    """Yields batches of (question, positive_table) pairs -- exactly one
    positive table per query, randomly chosen when a query has more than
    one valid positive (see FinetuneTrainer._score_batch's docstring for
    why exactly one is required: the cross-score matrix assumes a single
    diagonal positive per query)."""
    for idx_batch in query_dataset.iter_batches(batch_size):
        batch = []
        for idx in idx_batch:
            question, tables = query_dataset[idx]
            table = rng.choice(tables)
            batch.append((question, table))
        if len(batch) >= 2:
            yield batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", required=True)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--questions_json", required=True)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--max_rows", type=int, default=50)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--embed_dim", type=int, default=64, help="must match the pretrained checkpoint")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--scoring_mode", default="row_match")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--checkpoint_dir", default="eval/report_runs/finetune")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"indexing tables from {args.tables_json} / {args.databases_root} ...")
    table_dataset = SynSQLTableDataset(
        tables_json=args.tables_json,
        databases_root=args.databases_root,
        max_rows=args.max_rows,
    )

    query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    print(f"loaded {len(query_dataset)} query -> table example(s)")

    n_val = max(1, int(len(query_dataset) * args.val_frac))
    all_indices = list(range(len(query_dataset)))
    rng.shuffle(all_indices)
    val_indices, train_indices = set(all_indices[:n_val]), set(all_indices[n_val:])

    class _Subset:
        def __init__(self, base, indices):
            self.base, self.indices = base, list(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            return self.base[self.indices[i]]

        def iter_batches(self, batch_size, shuffle=True):
            order = list(range(len(self.indices)))
            if shuffle:
                rng.shuffle(order)
            for i in range(0, len(order), batch_size):
                yield order[i : i + batch_size]

    train_subset = _Subset(query_dataset, train_indices)
    val_subset = _Subset(query_dataset, val_indices)

    train_batches = list(resolve_batches(train_subset, args.batch_size, rng))
    val_batches = list(resolve_batches(val_subset, args.batch_size, rng))
    print(f"{len(train_batches)} train batches/epoch, {len(val_batches)} val batches")

    cell_encoder = CellEncoder(text_model_name="bert-base-uncased", output_dim=args.embed_dim)
    model = TableEncoder(cell_encoder, embed_dim=args.embed_dim, num_layers=args.num_layers)
    load_pretrained_encoder(model, args.pretrained_checkpoint, device=args.device)

    query_encoder = QueryEncoder(model_name="bert-base-uncased", output_dim=args.embed_dim)

    trainer = FinetuneTrainer(
        model,
        query_encoder,
        lr=args.lr,
        scoring_mode=args.scoring_mode,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    print(f"starting finetuning on {args.device} (scoring_mode={args.scoring_mode}) ...")
    trainer.fit(
        train_batches,
        num_epochs=args.num_epochs,
        steps_per_epoch=len(train_batches),
        log_every=args.log_every,
        val_batches=val_batches,
    )

    print("\nFinetuning complete.")
