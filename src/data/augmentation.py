"""
Table-level augmentations for self-supervised contrastive training.

Following Starmie's ablation (column-subset dropping outperformed
shuffle-based augmentations even for their non-permutation-invariant
encoder), and consistent with our architecture's *built-in* permutation
invariance (which makes any shuffle-based augmentation a zero-gradient
no-op for us specifically): every augmentation here is subset-dropping,
never reordering.
"""

import random

from src.data.table import Column, Table


def drop_rows(table: Table, keep_frac: float = 0.7) -> Table:
    """
    Randomly keep keep_frac of the rows, applied identically across every
    column so row alignment across columns is preserved.
    """

    m = table.num_rows
    n_keep = max(1, int(round(m * keep_frac)))
    keep_indices = sorted(random.sample(range(m), n_keep))

    new_columns = [
        Column(header=col.header, cells=[col.cells[i] for i in keep_indices])
        for col in table.columns
    ]

    return Table(
        table_id=table.table_id,
        table_name=table.table_name,
        columns=new_columns,
    )


def drop_columns(table: Table, keep_frac: float = 0.7) -> Table:
    """Randomly keep keep_frac of the columns."""

    n = table.num_columns
    n_keep = max(1, int(round(n * keep_frac)))
    keep_indices = sorted(random.sample(range(n), n_keep))

    new_columns = [table.columns[i] for i in keep_indices]

    return Table(
        table_id=table.table_id,
        table_name=table.table_name,
        columns=new_columns,
    )


def augment_table(
    table: Table,
    row_keep_frac: float = 0.7,
    col_keep_frac: float = 0.7,
) -> Table:
    """
    Composed augmentation used to produce the positive view of a table
    for contrastive training: drop a random subset of rows, then a
    random subset of the remaining columns.
    """

    augmented = drop_rows(table, row_keep_frac)
    augmented = drop_columns(augmented, col_keep_frac)
    return augmented