"""
Core data structures shared across encoding and models: a table is a
list of columns, each column a header plus raw cell text.

This is the only shape CellEncoder, ColumnAggregator, and friends need
to agree on -- everything downstream is plain torch tensors.
"""

from dataclasses import dataclass, field


@dataclass
class Column:
    """A single column: a header plus its raw cell text, top to bottom."""

    header: str
    cells: list[str] = field(default_factory=list)
    # True if this column is a declared SQL FOREIGN KEY (set by
    # SynSQLTableDataset.get_table() from the live schema; defaults to
    # False for any other data source, e.g. corpus_loader.py tables,
    # where this isn't known/applicable). Deliberately NOT set for
    # PRIMARY KEY columns -- see src/data/electra_corruption.py::
    # build_non_fk_mask for why the two are treated differently in the
    # ELECTRA discriminator loss.
    is_foreign_key: bool = False

    def __len__(self) -> int:
        return len(self.cells)


@dataclass
class Table:
    """A table as a list of columns, plus identifying metadata."""

    table_id: str
    table_name: str
    columns: list[Column] = field(default_factory=list)

    @property
    def num_columns(self) -> int:
        return len(self.columns)

    @property
    def num_rows(self) -> int:
        """Number of rows, taken as the shortest column (columns should
        normally be equal length, but this guards against ragged input)."""
        if not self.columns:
            return 0
        return min(len(col) for col in self.columns)

    @classmethod
    def from_columns(
        cls,
        table_id: str,
        table_name: str,
        columns: list[dict],
    ) -> "Table":
        """
        Build a Table from a plain list of {"header": ..., "values": [...]}
        dicts -- the same shape produced by extract_columns() in the
        corpus-building scripts.
        """

        cols = [
            Column(header=c["header"], cells=list(c["values"]))
            for c in columns
        ]

        return cls(table_id=table_id, table_name=table_name, columns=cols)