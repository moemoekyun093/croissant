"""
ELECTRA-style pretraining: cell corruption + per-cell discriminator.

Defaults for every flag below come from configs/model.yaml and
configs/pretrain.yaml (via src/training/config.py::apply_yaml_defaults)
-- running with no flags at all uses whatever's currently in those
files. Pass a --flag explicitly to override just that one value for a
single run.

Usage:
    python -m scripts.pretrain_electra \
        --tables_json /path/to/synsql/tables.json \
        --databases_root /path/to/synsql/databases

    # override one value without touching the yaml:
    python -m scripts.pretrain_electra \
        --tables_json /path/to/synsql/tables.json \
        --databases_root /path/to/synsql/databases \
        --corrupt_frac 0.25

    # small pilot run -- caps how much of SynSQL gets touched at all,
    # not just how much gets trained on: --n_dbs limits WHICH databases
    # are even queried (cheapest lever), --n_tables further caps the
    # total table count after that. Recommended before a full run.
    python -m scripts.pretrain_electra \
        --tables_json /path/to/synsql/tables.json \
        --databases_root /path/to/synsql/databases \
        --n_dbs 20 --n_tables 200

Data source is SynSQLTableDataset (src/data/synsql_dataset.py) -- real
cell values pulled live from SQLite per table, not a precomputed JSONL
corpus. See that module's docstring for the assumed tables.json /
databases/ layout.
"""

import argparse
import os
import random

from src.data.synsql_dataset import SynSQLTableDataset
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import DiscriminatorHead, TableEncoder
from src.training.config import apply_yaml_defaults
from src.training.trainer import PretrainTrainer


def make_batches(tables, batch_size, rng, max_columns=20):
    """Same bucket-by-size + truncate-wide-tables strategy as
    scripts/real_data_check.py::bucket_tables -- keeps padding cost down.

    Also drops any table with zero rows or zero columns before batching
    -- a handful of real SynSQL tables are genuinely empty in their
    source SQLite database (SynSQLTableDataset.get_table() faithfully
    returns them as a Table with 0-length columns), and there's nothing
    for ANY encoder to learn from an empty table. Most baselines happen
    to tolerate this silently, but src/encoding/baseline_encoders/
    common.py::validate_table (used by e.g. turl.py) explicitly raises
    on it, which crashes the whole run the first time a batch happens to
    contain one. Filtering here is the shared choke point every
    encoder's pretrain batches pass through, so it's a fix for all of
    them at once, not a turl-specific patch."""
    from src.data.table import Table

    capped = []
    skipped = 0
    for t in tables:
        if t.num_columns == 0 or t.num_rows == 0:
            skipped += 1
            continue
        if len(t.columns) > max_columns:
            t = Table(table_id=t.table_id, table_name=t.table_name, columns=t.columns[:max_columns])
        capped.append(t)

    if skipped:
        print(f"[make_batches] skipped {skipped} empty table(s) (0 rows or 0 columns)")

    capped.sort(key=lambda t: (t.num_columns, t.num_rows))
    batches = [capped[i : i + batch_size] for i in range(0, len(capped), batch_size)]
    batches = [b for b in batches if len(b) >= 2]
    rng.shuffle(batches)
    return batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", required=True)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--max_columns", type=int)
    parser.add_argument("--val_frac", type=float)
    parser.add_argument("--embed_dim", type=int, help="must be even (see CellEncoder)")
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
    parser.add_argument(
        "--discriminator_hidden_dim", type=int,
        help="defaults to embed_dim when omitted",
    )
    parser.add_argument("--corrupt_frac", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--grad_clip_norm", type=float)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument(
        "--n_dbs", type=int, default=None,
        help="pilot run: randomly sample only N databases, instead of every "
             "db_id in tables_json/databases_root -- avoids enumerating "
             "tables for the whole corpus. Recommended for a small run.",
    )
    parser.add_argument(
        "--n_tables", type=int, default=None,
        help="pilot run: further cap the total number of tables loaded "
             "(applied after --n_dbs sampling; selection is shuffled first, "
             "not just the first N off each database in schema order)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="seeds db/table sampling and batch shuffling, so --n_dbs/"
             "--n_tables pilot runs are reproducible across runs",
    )
    parser.add_argument(
        "--text_cache_path", default=None,
        help="path to CellEncoder's cell/header BERT-embedding cache (see "
             "CellEncoder.save_text_cache/load_text_cache). If a file already "
             "exists here, it's loaded before training so no cell/header string "
             "seen in a previous run has to go through BERT again. Always saved "
             "here at the end (accumulating this run's newly-seen strings), so a "
             "later scripts.finetune_query_table run can load the same file and "
             "start warm instead of cold. Defaults to <checkpoint_dir>/text_cache.pt.",
    )

    apply_yaml_defaults(parser, "configs/model.yaml", "configs/pretrain.yaml")
    args = parser.parse_args()

    if args.text_cache_path is None:
        args.text_cache_path = os.path.join(args.checkpoint_dir, "text_cache.pt")

    rng = random.Random(args.seed)

    print(f"indexing tables from {args.tables_json} / {args.databases_root} ...")
    table_dataset = SynSQLTableDataset(
        tables_json=args.tables_json,
        databases_root=args.databases_root,
        max_rows=args.max_rows,
    )

    db_ids = table_dataset.db_ids()
    print(f"{len(db_ids)} database(s) known")
    if args.n_dbs is not None and args.n_dbs < len(db_ids):
        rng.shuffle(db_ids)
        db_ids = db_ids[: args.n_dbs]
        print(f"pilot run: sampling {len(db_ids)} database(s)")

    # gather (db_id, table_name) pairs from only the sampled databases --
    # NOT table_dataset.iter_tables(), which would touch every db_id in
    # the dataset regardless of n_dbs/n_tables. Shuffle the pairs BEFORE
    # applying n_tables, so the cap is a random sample across the sampled
    # databases rather than just the first tables off each db in
    # sqlite_master's own (insertion) order.
    table_keys = [(db_id, t) for db_id in db_ids for t in table_dataset.tables_in_db(db_id)]
    rng.shuffle(table_keys)
    if args.n_tables is not None and args.n_tables < len(table_keys):
        table_keys = table_keys[: args.n_tables]
        print(f"pilot run: sampling {len(table_keys)} table(s)")

    tables = [table_dataset.get_table(db_id, t) for db_id, t in table_keys]
    print(f"loaded {len(tables)} table(s)")

    n_val = max(1, int(len(tables) * args.val_frac))
    val_tables, train_tables = tables[:n_val], tables[n_val:]
    print(f"split: {len(train_tables)} train / {len(val_tables)} held-out val")

    batches = make_batches(train_tables, args.batch_size, rng, max_columns=args.max_columns)
    val_batches = make_batches(val_tables, args.batch_size, rng, max_columns=args.max_columns)
    print(f"{len(batches)} train batches/epoch, {len(val_batches)} val batches")

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
    if os.path.exists(args.text_cache_path):
        print(f"loading cell/header text cache from {args.text_cache_path} ...")
        model.load_text_cache(args.text_cache_path)
        print(f"text cache warm-started with {cell_encoder.text_embedder.cache_size()} entries")

    discriminator = DiscriminatorHead(embed_dim=args.embed_dim, hidden_dim=args.discriminator_hidden_dim)

    trainer = PretrainTrainer(
        model,
        discriminator,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip_norm=args.grad_clip_norm,
        corrupt_frac=args.corrupt_frac,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed,
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

    os.makedirs(os.path.dirname(args.text_cache_path) or ".", exist_ok=True)
    model.save_text_cache(args.text_cache_path)
    print(
        f"saved text cache ({cell_encoder.text_embedder.cache_size()} entries) "
        f"to {args.text_cache_path}"
    )

    print("\nPretraining complete.")
