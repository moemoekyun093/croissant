"""
SynSQL data loading: pulls real cell values LIVE from the SQLite files
under `databases/<db_id>/<db_id>.sqlite`, and pairs real natural-language
queries (`questions_with_tables.json`) with their positive table(s) for
finetuning.

Table/column NAMES are read directly from each SQLite database's own
schema (`sqlite_master` + `PRAGMA table_info`) -- the same approach
scripts/build_bird_jsonl.py already uses elsewhere in this repo -- rather
than from `tables.json`. Confirmed against the real SynSQL-2.5M dump:
`tables.json` is Spider-style, a LIST of per-DATABASE records (not
per-table), each with a `ddls` field (raw CREATE TABLE strings) and a
`column_names` field that turned out to hold human-readable per-column
DESCRIPTIONS ("Unique identifier for each passenger"), not the actual
column identifiers ("passenger_id") -- parsing real column names out of
that would mean either regex-scraping the ddls or trusting Spider's
index-aligned column_names_original convention (not confirmed present in
this dump). Reading the live SQLite schema instead sidesteps all of that
ambiguity and can never drift out of sync with the real data, at the
cost of one extra `PRAGMA table_info` call per table (cached after the
first access).

Confirmed on-disk layout:

    <root>/databases/<db_id>/<db_id>.sqlite      # real data + real schema
    <root>/questions_with_tables.json            # query -> positive table(s)
    <root>/tables.json                           # Spider-style schema dump;
                                                  # only used here (optionally)
                                                  # to enumerate db_ids -- NOT
                                                  # for column names.

Confirmed `questions_with_tables.json` schema (list of records -- verified
against the real file):
    [
        {
            "db_id": "...",
            "question": "...",
            "required_tables": ["...", ...],
            "style": "...",       # e.g. "Vague" / "Colloquial" / "Imperative" -- unused here
            "complexity": "..."   # e.g. "Simple" / "Moderate" / "Complex" -- unused here
        },
        ...
    ]

If your actual SynSQL dump uses different key names for db_id/question/
required_tables, pass the `*_key` constructor args rather than editing
the parsing logic -- these are deliberately factored out for exactly
that reason.
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
from dataclasses import dataclass
from typing import Iterator

from src.data.table import Column, Table


def clean_text(x) -> str:
    """Same normalization used throughout the corpus-building scripts
    (build_bird_jsonl.py, build_bird_augmented_corpus.py) -- collapse
    whitespace, stringify, strip. Kept identical here so SynSQL tables
    look like every other source in the corpus."""
    if x is None:
        return ""
    return " ".join(str(x).split()).strip()


def get_sqlite_path(db_root: str, db_id: str) -> str:
    """Same convention as scripts/build_bird_jsonl.py::get_sqlite_path --
    confirmed to match SynSQL-2.5M's actual on-disk layout
    (databases/<db_id>/<db_id>.sqlite)."""
    return f"{db_root}/{db_id}/{db_id}.sqlite"


def _select_columns_by_name(
    cur: sqlite3.Cursor, table_name: str, column_names: list[str], rowids: list[int]
) -> list[tuple]:
    """Explicit `SELECT "col1", "col2", ... FROM table WHERE rowid IN (...)`
    -- selecting by declared name (not `SELECT *`), matching the exact
    column order PRAGMA table_info reported."""
    if not rowids:
        return []
    cols_sql = ", ".join(f'"{c}"' for c in column_names)
    placeholders = ",".join("?" * len(rowids))
    cur.execute(
        f'SELECT {cols_sql} FROM "{table_name}" WHERE rowid IN ({placeholders})',
        rowids,
    )
    return cur.fetchall()


# ==========================================================
# TABLE DATASET (live SQLite schema + data join)
# ==========================================================

class SynSQLTableDataset:
    """
    Materializes a real Table (header + sampled real cell values) for
    any (db_id, table_name) pair on demand, reading BOTH schema (table
    and column names) and data directly from that database's own SQLite
    file -- nothing precomputed or cached to disk. Call get_table()
    (or iterate the whole dataset via iter_tables()) whenever the data
    is actually needed.

    One sqlite3 connection is opened per db_id, cached for the lifetime
    of this object (databases are commonly reused across many tables and
    many queries in the same epoch -- reopening per table would be pure
    waste). Table/column names are also cached per db_id / (db_id,
    table_name) after their first PRAGMA lookup.
    """

    def __init__(
        self,
        databases_root: str,
        tables_json: str | None = None,
        max_rows: int = 50,
        seed: int = 42,
        db_id_key: str = "db_id",
    ):
        """
        databases_root: root of the databases/<db_id>/<db_id>.sqlite tree.
        tables_json:    optional -- if given, used ONLY to enumerate the
                        set of db_ids (via each record's `db_id_key`,
                        e.g. "db_id"). If omitted, db_ids are discovered
                        by listing subdirectories of databases_root
                        directly, which works just as well since every
                        db_id IS a subdirectory name -- tables.json isn't
                        actually required for anything else here (see
                        module docstring for why its column_names field
                        specifically is NOT used).
        """
        self.databases_root = databases_root
        self.max_rows = max_rows
        self._rng = random.Random(seed)

        if tables_json is not None:
            with open(tables_json, "r", encoding="utf-8") as f:
                raw = json.load(f)
            records = raw.values() if isinstance(raw, dict) else raw
            self._db_ids = [info[db_id_key] for info in records]
        else:
            self._db_ids = sorted(
                d for d in os.listdir(databases_root)
                if os.path.isdir(os.path.join(databases_root, d))
            )

        self._conn_cache: dict[str, sqlite3.Connection] = {}
        self._table_names_cache: dict[str, list[str]] = {}
        self._column_names_cache: dict[tuple[str, str], list[str]] = {}
        self._foreign_key_columns_cache: dict[tuple[str, str], set[str]] = {}

    def __len__(self) -> int:
        """Total (db_id, table_name) pairs across every known database --
        forces schema discovery for every db the first time this is
        called (each db_id's table list gets cached after that)."""
        return sum(len(self._table_names(db_id)) for db_id in self._db_ids)

    def db_ids(self) -> list[str]:
        return list(self._db_ids)

    def table_keys(self) -> list[tuple[str, str]]:
        """All (db_id, table_name) pairs this dataset knows about."""
        return [(db_id, t) for db_id in self._db_ids for t in self._table_names(db_id)]

    def tables_in_db(self, db_id: str) -> list[str]:
        return list(self._table_names(db_id))

    def has_table(self, db_id: str, table_name: str) -> bool:
        return db_id in self._db_ids and table_name in self._table_names(db_id)

    def _connection(self, db_id: str) -> sqlite3.Connection:
        if db_id not in self._conn_cache:
            path = get_sqlite_path(self.databases_root, db_id)
            self._conn_cache[db_id] = sqlite3.connect(path)
        return self._conn_cache[db_id]

    def _table_names(self, db_id: str) -> list[str]:
        """Real table names, read from this database's own sqlite_master
        -- not from tables.json (see module docstring)."""
        if db_id not in self._table_names_cache:
            cur = self._connection(db_id).cursor()
            cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            self._table_names_cache[db_id] = [r[0] for r in cur.fetchall()]
        return self._table_names_cache[db_id]

    def _column_names(self, db_id: str, table_name: str) -> list[str]:
        """Real column names, in the table's own declared order, read via
        PRAGMA table_info -- same convention scripts/build_bird_jsonl.py
        already uses for BIRD."""
        key = (db_id, table_name)
        if key not in self._column_names_cache:
            cur = self._connection(db_id).cursor()
            cur.execute(f'PRAGMA table_info("{table_name}")')
            # PRAGMA table_info row shape: (cid, name, type, notnull, dflt_value, pk)
            self._column_names_cache[key] = [row[1] for row in cur.fetchall()]
        return self._column_names_cache[key]

    def _foreign_key_columns(self, db_id: str, table_name: str) -> set[str]:
        """Column names declared as FOREIGN KEY in this table, read via
        PRAGMA foreign_key_list -- used to populate Column.is_foreign_key
        so the ELECTRA pretraining loss can exclude them (see
        src/data/electra_corruption.py::build_non_fk_mask for why)."""
        key = (db_id, table_name)
        if key not in self._foreign_key_columns_cache:
            cur = self._connection(db_id).cursor()
            cur.execute(f'PRAGMA foreign_key_list("{table_name}")')
            # PRAGMA foreign_key_list row shape:
            # (id, seq, table, from, to, on_update, on_delete, match)
            self._foreign_key_columns_cache[key] = {row[3] for row in cur.fetchall()}
        return self._foreign_key_columns_cache[key]

    def get_table(self, db_id: str, table_name: str) -> Table:
        """
        Fetches up to `max_rows` real rows for (db_id, table_name) from
        its SQLite database and returns a fully-populated Table. Random
        row sampling uses this dataset's own seeded RNG (not SQLite's
        own RANDOM(), which ignores Python's random.seed() -- same fix
        build_bird_jsonl.py::sample_base_rows already applies), via a
        bounded rowid scan + WHERE rowid IN (...) lookup so this stays
        cheap even for a huge table.
        """
        if not self.has_table(db_id, table_name):
            raise KeyError(f"unknown table: db_id={db_id!r} table_name={table_name!r}")

        column_names = self._column_names(db_id, table_name)
        cur = self._connection(db_id).cursor()

        cur.execute(f'SELECT rowid FROM "{table_name}"')
        all_rowids = [r[0] for r in cur.fetchall()]

        if not all_rowids:
            chosen: list[int] = []
        elif len(all_rowids) <= self.max_rows:
            chosen = all_rowids
        else:
            chosen = self._rng.sample(all_rowids, self.max_rows)

        rows = _select_columns_by_name(cur, table_name, column_names, chosen)

        columns_values: list[list[str]] = [[] for _ in column_names]
        for row in rows:
            for i, val in enumerate(row):
                columns_values[i].append(clean_text(val))

        fk_columns = self._foreign_key_columns(db_id, table_name)
        columns = [
            Column(
                header=column_names[i],
                cells=columns_values[i],
                is_foreign_key=column_names[i] in fk_columns,
            )
            for i in range(len(column_names))
        ]

        return Table(
            table_id=f"{db_id}#sep#{table_name}",
            table_name=f"{db_id}#sep#{table_name}",
            columns=columns,
        )

    def iter_tables(self) -> Iterator[Table]:
        """Materializes every table this dataset knows about, one at a
        time -- convenient for building a pretraining stream in one
        pass, but note this issues SQLite queries per table."""
        for db_id in self._db_ids:
            for table_name in self._table_names(db_id):
                yield self.get_table(db_id, table_name)


# ==========================================================
# QUERY DATASET (real query -> positive table(s), for finetuning)
# ==========================================================

@dataclass
class QueryTableExample:
    question: str
    db_id: str
    table_names: list[str]  # one or more positive tables for this query


class SynSQLQueryDataset:
    """
    Reads questions_with_tables.json and pairs each natural-language
    question with its positive table(s) (the "required_tables" field --
    confirmed against the real file), materialized via a
    SynSQLTableDataset -- this is the real query->table supervision used
    for finetuning (as opposed to pretraining's self-supervised ELECTRA
    cell-corruption task, which needs no queries at all).
    """

    def __init__(
        self,
        questions_json: str,
        table_dataset: SynSQLTableDataset,
        question_key: str = "question",
        db_id_key: str = "db_id",
        table_names_key: str = "required_tables",
    ):
        self.table_dataset = table_dataset

        with open(questions_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.examples: list[QueryTableExample] = []
        skipped = 0
        for record in raw:
            db_id = record[db_id_key]
            table_names = record[table_names_key]
            if isinstance(table_names, str):
                table_names = [table_names]

            # drop any positive table this database doesn't actually
            # contain, rather than crashing at train time -- logged once
            # at the end, not per-example (real datasets can have
            # thousands of these).
            valid = [t for t in table_names if table_dataset.has_table(db_id, t)]
            if not valid:
                skipped += 1
                continue

            self.examples.append(
                QueryTableExample(question=record[question_key], db_id=db_id, table_names=valid)
            )

        if skipped:
            print(
                f"[SynSQLQueryDataset] skipped {skipped} question(s) whose "
                f"positive table(s) weren't found in the live database schema"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[str, list[Table]]:
        """returns (question_text, [positive Table, ...])"""
        ex = self.examples[idx]
        tables = [self.table_dataset.get_table(ex.db_id, t) for t in ex.table_names]
        return ex.question, tables

    def iter_batches(self, batch_size: int, shuffle: bool = True) -> Iterator[list[int]]:
        """Yields lists of example indices -- kept as index batches (not
        materialized Tables) so shuffling is cheap and materialization
        only happens for the batch actually being trained on right now."""
        order = list(range(len(self.examples)))
        if shuffle:
            random.shuffle(order)
        for i in range(0, len(order), batch_size):
            yield order[i : i + batch_size]
