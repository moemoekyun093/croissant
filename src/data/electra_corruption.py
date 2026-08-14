"""
ELECTRA-style cell corruption for self-supervised table pretraining.

Cheap-swap corruption -- NO generator network: a corrupted cell's
replacement value is just another REAL value drawn from the same column
(same header text) elsewhere in the current batch, never a synthesized
or model-produced value. This is the same idea TABBIE's own
corrupt-cell-detection pretraining task uses (see
src/encoding/baseline_encoders/tabbie.py's README entry), and it avoids
having to train a second (generator) network alongside the discriminator
-- the discriminator's job is exactly "is this cell's value the one that
actually belongs in this column, or a plausible-looking value stolen
from somewhere else", not "did a generator model fool me".

"Same column" is defined by exact match on cleaned header text -- the
only signal available across different tables' schemas. A cell's
replacement pool is every OTHER real (non-null) value seen anywhere in
the batch under that same header string, EXCLUDING that cell's own
original value where an alternative exists (so a corrupted cell is
genuinely different from its original, not a same-value no-op swap).

Corruption never changes a table's shape (same columns, same rows) --
only cell VALUES are swapped -- so label grids line up exactly with
CellEncoder's own row/column padding without any extra bookkeeping.
"""

from __future__ import annotations

import random
from collections import defaultdict

from src.data.table import Column, Table


def _column_value_pools(tables: list[Table]) -> dict[str, list[str]]:
    """
    header text -> every real (non-empty) cell value seen anywhere in
    this batch under that header, across all tables. Built once per
    batch (not per cell) so sampling a replacement is O(1) instead of
    O(batch size) per corrupted cell.
    """
    pools: dict[str, list[str]] = defaultdict(list)
    for table in tables:
        for column in table.columns:
            for cell in column.cells:
                if cell.strip() != "":
                    pools[column.header].append(cell)
    return pools


def corrupt_tables(
    tables: list[Table],
    corrupt_frac: float = 0.15,
    rng: random.Random | None = None,
) -> tuple[list[Table], list[list[list[int]]]]:
    """
    tables:        a batch of B Tables (NOT modified in place -- new
                    Table/Column objects are returned).
    corrupt_frac:   fraction of a table's real (non-null) cells to
                    corrupt, independently per table.
    rng:            optional seeded random.Random for reproducibility;
                    defaults to the module-level `random` functions.

    returns:
        corrupted_tables: list of B Tables, same shape as the input,
                           with a random subset of cells swapped for
                           another real value from the same column
                           (elsewhere in the batch).
        label_grids:       list of B [n_cols x n_rows] int grids,
                           label_grids[t][c][r] = 1 if that cell was
                           replaced, else 0. Never marks an
                           already-empty cell as corrupted (nothing
                           meaningful to detect there -- cell_mask
                           already excludes null cells from
                           electra_discriminator_loss, and there's no
                           "real" value to compare against).
    """
    rand = rng or random
    pools = _column_value_pools(tables)

    corrupted_tables: list[Table] = []
    label_grids: list[list[list[int]]] = []

    for table in tables:
        new_columns: list[Column] = []
        table_labels: list[list[int]] = []

        for column in table.columns:
            pool = pools.get(column.header, [])
            new_cells: list[str] = list(column.cells)
            col_labels = [0] * len(column.cells)

            real_indices = [i for i, c in enumerate(column.cells) if c.strip() != ""]
            n_corrupt = int(round(len(real_indices) * corrupt_frac))

            if n_corrupt > 0 and pool:
                corrupt_indices = rand.sample(real_indices, min(n_corrupt, len(real_indices)))
                for i in corrupt_indices:
                    original = new_cells[i]
                    # prefer a genuinely different value; if the pool
                    # happens to be all-identical to this cell's own
                    # value (e.g. a constant column), fall back to
                    # allowing a same-value draw rather than skipping
                    # corruption for that cell entirely.
                    alternatives = [v for v in pool if v != original]
                    replacement = rand.choice(alternatives) if alternatives else rand.choice(pool)
                    new_cells[i] = replacement
                    col_labels[i] = 1

            new_columns.append(Column(header=column.header, cells=new_cells))
            table_labels.append(col_labels)

        corrupted_tables.append(
            Table(table_id=table.table_id, table_name=table.table_name, columns=new_columns)
        )
        label_grids.append(table_labels)

    return corrupted_tables, label_grids


def pad_labels(
    label_grids: list[list[list[int]]],
    device="cpu",
):
    """
    (torch imported lazily here -- corrupt_tables() itself has no torch
    dependency, so plain Table/Column corruption logic stays testable
    without torch installed.)

    Scatters the ragged per-table label grids from corrupt_tables() into
    a padded [B, max_n, max_m] float tensor -- max_n/max_m computed the
    SAME way CellEncoder.encode_tables_batched computes them (max
    columns, max rows across the batch), so this lines up with the
    model's own col_mask/row_mask/cell_mask without any extra
    coordination needed, as long as it's called on the SAME (possibly
    corrupted) tables the model just encoded.

    Padded (and originally-null) positions are 0 -- combine with the
    model's own cell_mask when computing the discriminator loss so
    padding/null cells never contribute.
    """
    import torch

    B = len(label_grids)
    n_list = [len(grid) for grid in label_grids]
    m_list = [len(grid[0]) if grid else 0 for grid in label_grids]
    max_n = max(n_list) if n_list else 1
    max_m = max(m_list) if m_list else 1

    labels = torch.zeros(B, max_n, max_m, device=device)
    for t_idx, grid in enumerate(label_grids):
        for c_idx, col_labels in enumerate(grid):
            for r_idx, label in enumerate(col_labels):
                if label:
                    labels[t_idx, c_idx, r_idx] = 1.0

    return labels
