"""
Finetuning: real query -> positive-table contrastive training, starting
from an ELECTRA-pretrained encoder checkpoint.

Defaults for every flag below come from configs/model.yaml and
configs/finetune.yaml (via src/training/config.py::apply_yaml_defaults)
-- running with no flags at all uses whatever's currently in those
files. Pass a --flag explicitly to override just that one value for a
single run.

Usage:
    python -m scripts.finetune_query_table \
        --tables_json /path/to/synsql/tables.json \
        --databases_root /path/to/synsql/databases \
        --questions_json /path/to/synsql/questions_with_tables.json \
        --pretrained_checkpoint eval/report_runs/pretrain/checkpoint_epoch14.pt

    # small pilot run -- caps how many query->table examples get used
    python -m scripts.finetune_query_table \
        --tables_json /path/to/synsql/tables.json \
        --databases_root /path/to/synsql/databases \
        --questions_json /path/to/synsql/questions_with_tables.json \
        --pretrained_checkpoint eval/report_runs/pretrain/checkpoint_epoch14.pt \
        --n_examples 500

Loads the pretrained TableEncoder via load_pretrained_encoder(), which
discards any discriminator-head weights from that checkpoint -- the
discriminator was only ever needed for the pretraining task.
"""

import argparse
import random

from src.data.synsql_dataset import SynSQLQueryDataset, SynSQLTableDataset
from src.data.table import Table
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import TableEncoder, load_pretrained_encoder
from src.training.config import apply_yaml_defaults
from src.training.query_encoder import QueryEncoder
from src.training.trainer import FinetuneTrainer


def cap_columns(table: Table, max_columns: int) -> Table:
    """Same outlier-wide-table truncation as the pretraining path's
    make_batches -- a positive table with many more columns than usual
    would otherwise single-handedly set the padding cost for whatever
    batch it lands in."""
    if len(table.columns) <= max_columns:
        return table
    return Table(table_id=table.table_id, table_name=table.table_name, columns=table.columns[:max_columns])


def resolve_batches(
    query_dataset: SynSQLQueryDataset, batch_size: int, max_columns: int, rng: random.Random
):
    """Yields batches of (question, positive_table) pairs -- exactly one
    positive table per query, randomly chosen when a query has more than
    one valid positive (see FinetuneTrainer._score_batch's docstring for
    why exactly one is required: the cross-score matrix assumes a single
    diagonal positive per query)."""
    for idx_batch in query_dataset.iter_batches(batch_size):
        batch = []
        for idx in idx_batch:
            question, tables = query_dataset[idx]
            table = cap_columns(rng.choice(tables), max_columns)
            batch.append((question, table))
        if len(batch) >= 2:
            yield batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", required=True)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--questions_json", required=True)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--max_columns", type=int)
    parser.add_argument("--val_frac", type=float)
    parser.add_argument("--embed_dim", type=int, help="must match the pretrained checkpoint")
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--text_model_name")
    parser.add_argument("--text_max_length", type=int)
    parser.add_argument("--text_max_batch_size", type=int)
    parser.add_argument(
        "--text_trainable", action=argparse.BooleanOptionalAction,
        help="unfreeze CellEncoder's BERT -- off by default, see training_config.md",
    )
    parser.add_argument("--nonlinearity", choices=["sigmoid", "tanh", "relu"])
    parser.add_argument(
        "--channel_mix_hidden_dim", type=int,
        help="defaults to embed_dim * 2 when omitted",
    )
    parser.add_argument("--query_model_name")
    parser.add_argument(
        "--query_trainable", action=argparse.BooleanOptionalAction,
        help="train the query tower's BERT -- on by default, unlike CellEncoder's frozen BERT",
    )
    parser.add_argument("--query_max_length", type=int)
    parser.add_argument("--exclude_special_tokens", action=argparse.BooleanOptionalAction)
    parser.add_argument("--scoring_mode", choices=["global", "row_match", "column_match", "col_deepset", "row_deepset", "mixture"])
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--grad_clip_norm", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_examples", type=int, default=None,
        help="pilot run: randomly cap the number of query->table examples "
             "used (applied after loading questions_json, before the "
             "train/val split)",
    )

    apply_yaml_defaults(parser, "configs/model.yaml", "configs/finetune.yaml")
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

    all_indices = list(range(len(query_dataset)))
    rng.shuffle(all_indices)
    if args.n_examples is not None and args.n_examples < len(all_indices):
        all_indices = all_indices[: args.n_examples]
        print(f"pilot run: sampling {len(all_indices)} example(s)")

    n_val = max(1, int(len(all_indices) * args.val_frac))
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

    train_batches = list(resolve_batches(train_subset, args.batch_size, args.max_columns, rng))
    val_batches = list(resolve_batches(val_subset, args.batch_size, args.max_columns, rng))
    print(f"{len(train_batches)} train batches/epoch, {len(val_batches)} val batches")

    cell_encoder = CellEncoder(
        text_model_name=args.text_model_name,
        output_dim=args.embed_dim,
        text_max_length=args.text_max_length,
        text_trainable=args.text_trainable,
        text_max_batch_size=args.text_max_batch_size,
    )
    model = TableEncoder(
        cell_encoder,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        nonlinearity=args.nonlinearity,
        channel_mix_hidden_dim=args.channel_mix_hidden_dim,
    )
    load_pretrained_encoder(model, args.pretrained_checkpoint, device=args.device)

    query_encoder = QueryEncoder(
        model_name=args.query_model_name,
        output_dim=args.embed_dim,
        max_length=args.query_max_length,
        trainable=args.query_trainable,
        exclude_special_tokens=args.exclude_special_tokens,
    )

    trainer = FinetuneTrainer(
        model,
        query_encoder,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip_norm=args.grad_clip_norm,
        temperature=args.temperature,
        scoring_mode=args.scoring_mode,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed,
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
