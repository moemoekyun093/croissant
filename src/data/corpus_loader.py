"""
Reconstructs raw Table objects (with per-cell values) from the existing
corpus.jsonl records -- which only persist a flattened `contents` string,
not the original per-column values.

This works because create_document()'s format is fully deterministic:

    "[TABLE]\\n{page_title}\\n\\n[SCHEMA]\\n{col1} | {col2} | ...\\n\\n[ROWS]\\n{row1}\\n{row2}\\n..."

with every row's cells joined by " | ". Splitting on the fixed
"\\n\\n[SCHEMA]\\n" and "\\n\\n[ROWS]\\n" markers recovers the header and
rows blocks exactly; splitting each row line on " | " recovers per-cell
values.

Known limitation: if a cell's own text happens to contain the literal
substring " | ", that row will split into more pieces than there are
columns and gets skipped (logged, not silently dropped) rather than
guessed at -- this is rare, but a real source of quiet data loss worth
checking corpus-wide if row counts look consistently lower than expected.
"""

import json
from typing import Iterator

from src.data.table import Column, Table

_SCHEMA_MARKER = "\n\n[SCHEMA]\n"
_ROWS_MARKER = "\n\n[ROWS]\n"
_TABLE_PREFIX = "[TABLE]\n"
_CELL_SEP = " | "


def table_from_corpus_record(record: dict) -> Table:
    """
    record: one parsed JSON line from corpus.jsonl, e.g.
        {"id": "69", "table_name": ..., "column_names": [...], "contents": ...}

    returns: a Table with real per-column cell values, or an empty-column
    Table if the record has no schema/rows (matches create_document()'s
    own fallback for tables with zero usable columns).
    """

    table_id = str(record["id"])
    table_name = record.get("table_name", "")
    column_names = record.get("column_names", [])
    contents = record.get("contents", "")

    if not column_names or _SCHEMA_MARKER not in contents or _ROWS_MARKER not in contents:
        return Table(table_id=table_id, table_name=table_name, columns=[])

    _header_part, rest = contents.split(_SCHEMA_MARKER, 1)
    _schema_line, rows_block = rest.split(_ROWS_MARKER, 1)

    n_cols = len(column_names)
    row_lines = rows_block.split("\n") if rows_block else []

    values_per_column: list[list[str]] = [[] for _ in range(n_cols)]
    skipped_rows = 0

    for row_line in row_lines:
        # maxsplit = n_cols - 1 so a stray " | " inside the *last* cell
        # doesn't over-split -- doesn't help for stray " | " in earlier
        # cells, which is the known limitation noted above.
        cells = row_line.split(_CELL_SEP, n_cols - 1)

        if len(cells) != n_cols:
            skipped_rows += 1
            continue

        for i, cell in enumerate(cells):
            values_per_column[i].append(cell)

    if skipped_rows:
        print(
            f"[table_from_corpus_record] table {table_id}: "
            f"skipped {skipped_rows} ragged row(s)"
        )

    columns = [
        Column(header=column_names[i], cells=values_per_column[i])
        for i in range(n_cols)
    ]

    return Table(table_id=table_id, table_name=table_name, columns=columns)


def iter_tables_from_jsonl(path: str) -> Iterator[Table]:
    """Streams corpus.jsonl line by line, yielding one Table at a time."""

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            table = table_from_corpus_record(record)

            if table.num_columns > 0:
                yield table


def batch_tables(
    tables: Iterator[Table],
    batch_size: int,
) -> Iterator[list[Table]]:
    """Groups a stream of Tables into fixed-size batches (last batch may
    be smaller). Simple generator -- no shuffling; wrap the input
    iterator yourself (e.g. shuffle file order per epoch) if needed."""

    batch: list[Table] = []

    for table in tables:
        batch.append(table)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch