"""
Pilot training run: a bounded number of real tables (10,000 by default),
one pass, to get a first read on whether training actually works before
committing to a long run over the full corpus.

Usage:
    python -m scripts.pilot_train \
        --corpus_jsonl /mnt/nas/ayane/tables/big_corpus.jsonl \
        --n_tables 10000 \
        --batch_size 64 \
        --device cuda:2

--device defaults to cuda:2 -- pick whichever GPU is actually free via
nvidia-smi first; don't assume cuda:0 is available on a shared machine.
"""

import argparse
import random
import time

from src.data.corpus_loader import iter_tables_from_jsonl
from src.data.table import Table
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import TableEncoder
from src.training.trainer import Trainer


def load_tables(corpus_jsonl: str, n_tables: int) -> list[Table]:
    tables = []
    for t in iter_tables_from_jsonl(corpus_jsonl):
        tables.append(t)
        if len(tables) >= n_tables:
            break
    return tables


def make_batches(
    tables: list[Table],
    batch_size: int,
    max_columns: int = 20,
) -> list[list[Table]]:
    """
    Buckets tables by size before batching, so a rare very wide/tall
    table doesn't force the whole batch's padding (and therefore its
    compute/memory cost) up to match it -- ColumnAggregator's cost
    scales as B * max_n * max_m^2 * k, dominated entirely by the largest
    table in each batch. Sorting first means similarly-sized tables end
    up together, so most batches stay cheap and only a few pay the cost
    of the genuine outliers.

    Also truncates any table's columns to max_columns -- a small number
    of extreme outliers (e.g. >30 columns, seen in this corpus) would
    otherwise single-handedly set the padding cost for whatever batch
    they land in, even after bucketing.
    """

    capped = []
    for t in tables:
        if len(t.columns) > max_columns:
            t = Table(
                table_id=t.table_id,
                table_name=t.table_name,
                columns=t.columns[:max_columns],
            )
        capped.append(t)

    # sort by (columns, rows) so similarly-shaped tables land in the
    # same batch -- shuffling would undo exactly what bucketing buys us
    capped.sort(key=lambda t: (t.num_columns, t.num_rows))

    batches = [
        capped[i : i + batch_size]
        for i in range(0, len(capped), batch_size)
    ]
    batches = [b for b in batches if len(b) >= 2]

    # shuffle BATCH ORDER (not contents) so training doesn't see easy/
    # hard batches in a fixed size-sorted sequence every epoch
    random.shuffle(batches)

    return batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_jsonl", required=True)
    parser.add_argument("--n_tables", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="number of stacked TableLayer blocks (ColumnAggregator -> "
        "CrossColumnAttention -> ChannelMix) -- cheap to increase, since "
        "profiling shows cell encoding (BERT) dominates >99%% of wall time",
    )
    parser.add_argument(
        "--text_max_batch_size",
        type=int,
        default=2048,
        help="max cells per internal BERT forward call inside TextEmbedder -- "
        "raise this if using a much larger --batch_size and want fewer, bigger "
        "BERT calls per training step",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--checkpoint_dir", default="eval/report_runs/pilot")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--resume_from",
        default=None,
        help="path to a checkpoint .pt file to resume training from",
    )
    args = parser.parse_args()

    print(f"loading up to {args.n_tables} tables from {args.corpus_jsonl} ...")
    t0 = time.time()
    tables = load_tables(args.corpus_jsonl, args.n_tables)
    print(f"loaded {len(tables)} tables in {time.time() - t0:.1f}s")

    random.shuffle(tables)
    n_val = max(1, int(len(tables) * args.val_frac))
    val_tables, train_tables = tables[:n_val], tables[n_val:]
    print(f"split: {len(train_tables)} train / {len(val_tables)} held-out val")

    batches = make_batches(train_tables, args.batch_size)
    val_batches = make_batches(val_tables, args.batch_size)
    print(f"{len(batches)} train batches/epoch, {len(val_batches)} val batches (batch_size={args.batch_size})")

    cell_encoder = CellEncoder(
        text_model_name="bert-base-uncased",
        output_dim=args.embed_dim,
        text_max_batch_size=args.text_max_batch_size,
    )
    model = TableEncoder(cell_encoder, embed_dim=args.embed_dim, num_layers=args.num_layers)

    trainer = Trainer(
        model,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    print(f"starting pilot run on {args.device} ...")
    trainer.fit(
        batches,
        num_epochs=args.num_epochs,
        steps_per_epoch=len(batches),
        log_every=args.log_every,
        val_batches=val_batches,
        resume_from=args.resume_from,
    )

    print("\nPilot run complete.")