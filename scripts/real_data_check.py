"""
Runs the pipeline against a small real slice of your actual corpus,
before committing to a full run over the full ~1M tables.

Checks:
  1. Corpus health -- parse success rate, ragged-row skip rate
  2. Size distribution -- rows/columns per table (flags anything that
     could be slow/memory-heavy given the O(m^2)/O(n^2) attention steps)
  3. Crash resilience -- does every table survive a forward pass
  4. Timing -- encoding throughput, extrapolated to a full-corpus epoch
  5. A few real ELECTRA pretraining steps -- watches for NaN loss /
     crashes with real (not synthetic) data

Usage:
    python -m scripts.real_data_check --corpus_jsonl /path/to/corpus.jsonl --n_tables 200
"""

import argparse
import time

import torch

from src.data.corpus_loader import iter_tables_from_jsonl
from src.data.table import Table
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import DiscriminatorHead, TableEncoder
from src.training.trainer import PretrainTrainer


def load_slice(corpus_jsonl: str, n_tables: int) -> tuple[list[Table], int]:
    """Loads the first n_tables usable Tables from the corpus, plus a
    count of how many raw lines were skipped to get there (degenerate /
    zero-column records, per table_from_corpus_record's own filtering)."""

    tables = []
    lines_seen = 0

    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    for table in iter_tables_from_jsonl(corpus_jsonl):
        lines_seen += 1
        tables.append(table)
        if len(tables) >= n_tables:
            break

    skipped = lines_seen - len(tables)  # should be 0, iter already filters
    print(f"[corpus health] loaded {len(tables)} usable tables")
    print(f"[corpus health] total lines in file: {total_lines}")

    return tables, skipped


def report_size_distribution(tables: list[Table]) -> None:
    rows = [t.num_rows for t in tables]
    cols = [t.num_columns for t in tables]

    def stats(name, values):
        print(
            f"  {name}: min={min(values)} max={max(values)} "
            f"mean={sum(values) / len(values):.1f}"
        )

    print("[size distribution]")
    stats("rows", rows)
    stats("columns", cols)

    big_col_tables = [t for t in tables if t.num_columns > 30]
    big_row_tables = [t for t in tables if t.num_rows > 100]
    if big_col_tables:
        print(
            f"  [WARN] {len(big_col_tables)} table(s) with >30 columns -- "
            f"CrossColumnAttention is O(n^2) per channel, check these don't "
            f"blow up memory/time"
        )
    if big_row_tables:
        print(
            f"  [WARN] {len(big_row_tables)} table(s) with >100 rows -- "
            f"ColumnAggregator's XX^T is O(m^2) per column, same concern"
        )


def check_crash_resilience(model: TableEncoder, tables: list[Table]) -> list[Table]:
    """Returns the tables that survived a forward pass without error."""

    ok_tables = []
    failures = []

    model.eval()
    with torch.no_grad():
        for i, table in enumerate(tables):
            print(
                f"  [{i+1}/{len(tables)}] table_id={table.table_id} "
                f"n_cols={table.num_columns} n_rows={table.num_rows} ...",
                end=" ",
                flush=True,
            )
            t0 = time.time()
            try:
                X, col_mask, row_mask, cell_mask = model.forward_batch_cellwise([table])
                assert X.shape == (1, table.num_columns, table.num_rows, model.embed_dim)
                ok_tables.append(table)
                print(f"ok ({time.time()-t0:.2f}s)", flush=True)
            except Exception as e:
                failures.append((table.table_id, table.table_name, str(e)))
                print(f"FAILED ({time.time()-t0:.2f}s): {e}", flush=True)

    print(f"[crash resilience] {len(ok_tables)}/{len(tables)} tables encoded successfully")
    if failures:
        print(f"[crash resilience] {len(failures)} failure(s):")
        for table_id, table_name, error in failures[:10]:
            print(f"    table_id={table_id} name={table_name!r}: {error}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")

    return ok_tables


def check_timing(model: TableEncoder, tables: list[Table], full_corpus_size: int) -> None:
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i, table in enumerate(tables):
            t0 = time.time()
            model.forward_batch_cellwise([table])
            elapsed = time.time() - t0
            if elapsed > 2.0 or i % 20 == 0:
                print(
                    f"  [timing {i+1}/{len(tables)}] table_id={table.table_id} "
                    f"n_cols={table.num_columns} n_rows={table.num_rows} "
                    f"took {elapsed:.2f}s",
                    flush=True,
                )
    elapsed = time.time() - start

    per_table = elapsed / len(tables)
    print(f"[timing] {len(tables)} tables encoded in {elapsed:.1f}s ({per_table*1000:.1f} ms/table)")

    est_full_epoch = per_table * full_corpus_size
    print(
        f"[timing] estimated time for one epoch over {full_corpus_size:,} tables: "
        f"{est_full_epoch/60:.1f} min ({est_full_epoch/3600:.1f} hr) "
        f"-- encoding only, no augmentation/backward pass included"
    )


def bucket_tables(
    tables: list[Table], batch_size: int, max_columns: int = 20
) -> list[list[Table]]:
    """Same fix as scripts/pretrain_electra.py: cap extreme-outlier column counts and
    sort by size before batching, so a rare wide/tall table doesn't force
    padding cost onto an entire batch of otherwise small tables."""

    capped = []
    for t in tables:
        if len(t.columns) > max_columns:
            t = Table(table_id=t.table_id, table_name=t.table_name, columns=t.columns[:max_columns])
        capped.append(t)

    capped.sort(key=lambda t: (t.num_columns, t.num_rows))

    return [capped[i : i + batch_size] for i in range(0, len(capped), batch_size)]


def check_real_pretrain_steps(
    model_builder,
    discriminator_builder,
    tables: list[Table],
    batch_size: int = 8,
    n_steps: int | None = None,
    corrupt_frac: float = 0.15,
) -> None:
    """
    Runs a few real ELECTRA pretraining steps (cell corruption ->
    DiscriminatorHead -> BCE loss) via PretrainTrainer, watching for NaN
    loss or crashes on real (not synthetic) data.

    n_steps: if None, runs over ALL bucketed batches (recommended for an
    honest throughput estimate -- bucketing sorts small-to-large, so
    only looking at the first few steps sees only the smallest tables
    and badly overstates real throughput).
    """

    model = model_builder()
    discriminator = discriminator_builder()
    trainer = PretrainTrainer(model, discriminator, lr=1e-4, warmup_ratio=0.0, corrupt_frac=corrupt_frac)
    trainer.optimizer = torch.optim.AdamW(trainer._trainable_params(), lr=1e-4)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda step: 1.0
    )

    batches = bucket_tables(tables, batch_size)
    if n_steps is not None:
        batches = batches[:n_steps]

    print(f"[real pretrain steps] running {len(batches)} batches, batch_size={batch_size}")

    total_elapsed = 0.0
    total_table_forwards = 0

    for step, batch in enumerate(batches):
        if len(batch) < 2:
            print(f"  step {step}: batch too small, skipping")
            continue

        print(f"  step {step}: batch max_columns={max(t.num_columns for t in batch)} "
              f"max_rows={max(t.num_rows for t in batch)}", flush=True)

        t0 = time.time()
        try:
            loss_value = trainer.train_step(batch)
        except Exception as e:
            print(f"  step {step}: CRASHED -- {e}")
            raise
        elapsed = time.time() - t0

        total_elapsed += elapsed
        # unlike the old augmentation-contrastive path, ELECTRA forwards
        # ONE corrupted copy of each table per step, not an
        # original+augmented pair -- so no 2x factor here.
        total_table_forwards += len(batch)

        status = "NaN!" if loss_value != loss_value else f"{loss_value:.4f}"
        print(f"  step {step}: loss {status} ({elapsed:.1f}s)")

    if total_elapsed > 0:
        tables_per_sec = total_table_forwards / total_elapsed
        print(
            f"\n[real pretrain steps] aggregate: {total_table_forwards} table-forwards "
            f"in {total_elapsed:.1f}s ({tables_per_sec:.1f} tables/s)"
        )

        full_epoch_forwards = 1_000_000
        est_min = full_epoch_forwards / tables_per_sec / 60
        print(
            f"[real pretrain steps] estimated full-corpus epoch (batched, "
            f"train+forward, no backward-pass overhead separated out): "
            f"{est_min:.1f} min ({est_min/60:.1f} hr)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_jsonl", required=True)
    parser.add_argument("--n_tables", type=int, default=200)
    parser.add_argument("--full_corpus_size", type=int, default=1_000_000)
    parser.add_argument("--embed_dim", type=int, default=32)
    args = parser.parse_args()

    def build_model() -> TableEncoder:
        cell_encoder = CellEncoder(
            text_model_name="bert-base-uncased", output_dim=args.embed_dim
        )
        return TableEncoder(cell_encoder, embed_dim=args.embed_dim)

    def build_discriminator() -> DiscriminatorHead:
        return DiscriminatorHead(embed_dim=args.embed_dim)

    tables, skipped = load_slice(args.corpus_jsonl, args.n_tables)
    report_size_distribution(tables)

    model = build_model()
    ok_tables = check_crash_resilience(model, tables)

    if ok_tables:
        check_timing(model, ok_tables, args.full_corpus_size)
        check_real_pretrain_steps(build_model, build_discriminator, ok_tables, n_steps=None)

    print("\nReal-data slice check complete.")