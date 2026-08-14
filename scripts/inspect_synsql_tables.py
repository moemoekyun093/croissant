"""
Prints a random sample of real SynSQL tables (headers + a few example
cell values per column, plus a primary/foreign-key flag from the live
SQLite schema) so you can actually look at what the data looks like
before a pilot run -- e.g. to judge how many columns are IDs/keys,
where a same-column swap corruption (src/data/electra_corruption.py)
carries little or no learnable signal (an ID swap is often just another
arbitrary-looking valid ID, with no "typical value" pattern for the
discriminator to key off of).

This is read-only inspection, no training. Doesn't touch CellEncoder/
TableEncoder at all.

Usage:
    python -m scripts.inspect_synsql_tables \
        --tables_json /path/to/synsql/tables.json \
        --databases_root /path/to/synsql/databases \
        --n_tables 15

    # only look at tables from a specific db:
    python -m scripts.inspect_synsql_tables \
        --databases_root /path/to/synsql/databases \
        --db_id some_db_id
"""

import argparse
import random
import re
import sqlite3

from src.data.synsql_dataset import SynSQLTableDataset, get_sqlite_path

# Heuristic only -- flags a column as "possibly a key" by header name
# (id, _id, id_, rowid, ...uuid, ...key) OR by actually being a
# declared PRIMARY KEY / FOREIGN KEY per the live schema (more
# reliable than the name heuristic, used as the primary signal below).
_ID_NAME_RE = re.compile(r"(^|_)(id|uuid|guid|key)($|_)", re.IGNORECASE)


def _key_columns(databases_root: str, db_id: str, table_name: str) -> tuple[set[str], set[str]]:
    """Returns (primary_key_columns, foreign_key_columns) for one table,
    read directly from the live SQLite schema -- ground truth, not a
    name-based guess."""
    conn = sqlite3.connect(get_sqlite_path(databases_root, db_id))
    cur = conn.cursor()

    pk_cols = set()
    cur.execute(f'PRAGMA table_info("{table_name}")')
    for row in cur.fetchall():
        # row: (cid, name, type, notnull, dflt_value, pk) -- pk > 0 means
        # part of the primary key (order within a composite key)
        if row[5] > 0:
            pk_cols.add(row[1])

    fk_cols = set()
    cur.execute(f'PRAGMA foreign_key_list("{table_name}")')
    for row in cur.fetchall():
        # row: (id, seq, table, from, to, on_update, on_delete, match)
        fk_cols.add(row[3])

    conn.close()
    return pk_cols, fk_cols


def print_table(table_dataset: SynSQLTableDataset, databases_root: str, db_id: str, table_name: str, n_examples: int = 5) -> None:
    table = table_dataset.get_table(db_id, table_name)
    pk_cols, fk_cols = _key_columns(databases_root, db_id, table_name)

    print(f"\n=== {db_id} :: {table_name} === ({table.num_columns} cols x {table.num_rows} rows sampled)")
    for col in table.columns:
        flags = []
        if col.header in pk_cols:
            flags.append("PRIMARY KEY")
        if col.header in fk_cols:
            flags.append("FOREIGN KEY")
        if not flags and _ID_NAME_RE.search(col.header):
            flags.append("id-like name")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""

        examples = col.cells[:n_examples]
        examples_str = ", ".join(repr(v) for v in examples)
        n_distinct = len(set(col.cells))
        print(f"  {col.header!r:30s}{flag_str:20s} n_distinct={n_distinct}/{len(col.cells)}  examples: {examples_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", default=None)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--n_tables", type=int, default=15)
    parser.add_argument("--n_examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db_id", default=None, help="restrict to one db_id instead of sampling across all of them")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    table_dataset = SynSQLTableDataset(
        tables_json=args.tables_json,
        databases_root=args.databases_root,
    )

    if args.db_id is not None:
        table_keys = [(args.db_id, t) for t in table_dataset.tables_in_db(args.db_id)]
    else:
        db_ids = table_dataset.db_ids()
        table_keys = [(db_id, t) for db_id in db_ids for t in table_dataset.tables_in_db(db_id)]

    rng.shuffle(table_keys)
    table_keys = table_keys[: args.n_tables]

    print(f"showing {len(table_keys)} table(s)")

    n_id_like_total = 0
    n_cols_total = 0

    for db_id, table_name in table_keys:
        print_table(table_dataset, args.databases_root, db_id, table_name, n_examples=args.n_examples)
        table = table_dataset.get_table(db_id, table_name)
        pk_cols, fk_cols = _key_columns(args.databases_root, db_id, table_name)
        for col in table.columns:
            n_cols_total += 1
            if col.header in pk_cols or col.header in fk_cols or _ID_NAME_RE.search(col.header):
                n_id_like_total += 1

    print(f"\n=== summary: {n_id_like_total}/{n_cols_total} columns across this sample are key/id-like ({100*n_id_like_total/max(1,n_cols_total):.1f}%) ===")
