"""Materialize a deterministic, row-capped, FK-consistent SQL corpus.

The ordinary ``SynSQLTableDataset`` sampler caps every table independently.
That is suitable for unrelated web tables, but a random child-table sample
will usually reference parent rows absent from the parent's independent
sample.  This script samples a whole database jointly:

1. Parent tables are sampled before their children.
2. A child row is eligible only when every non-null outgoing FK tuple points
   to a row retained in the corresponding parent table.
3. Cyclic/self-referencing components are repeatedly pruned to a fixed point.

Consequently every emitted table has at most ``--max_rows`` rows and the
sampled database has no dangling declared foreign keys.  Some child tables
may have fewer than the cap; satisfying both an unconditional exact row count
and strict FK closure is not generally possible under a hard parent-table cap.

The source SQLite files are read-only and never modified.  Output uses the
same JSON representation consumed by ``SynSQLTableDataset.load_corpus``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import tempfile
import time
from typing import Iterable

from src.data.synsql_dataset import clean_text


@dataclass(frozen=True)
class ForeignKey:
    child: str
    parent: str
    child_columns: tuple[str, ...]
    parent_columns: tuple[str, ...]


@dataclass(frozen=True)
class SampledRow:
    # Integer SQLite rowid for ordinary tables; primary-key tuple (or a
    # deterministic scan ordinal) for WITHOUT ROWID/virtual tables.
    rowid: object
    values: tuple[object, ...]


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False)
    os.replace(partial, path)


def _stable_rng(seed: int, db_id: str, table_name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\0{db_id}\0{table_name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _sqlite_path(dataset_root: Path, db_id: str) -> Path:
    direct = dataset_root / "databases" / db_id / f"{db_id}.sqlite"
    if direct.exists():
        return direct
    matches = sorted((dataset_root / "databases" / db_id).glob("*.sqlite"))
    if len(matches) != 1:
        raise FileNotFoundError(f"could not resolve SQLite file for {db_id!r}")
    return matches[0]


def _stage_sqlite(source: Path, staging_dir: Path, db_id: str) -> Path:
    """Copy one NAS database sequentially so sampling uses local random I/O."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    size = source.stat().st_size
    free = shutil.disk_usage(staging_dir).free
    if size > free:
        raise OSError(
            f"not enough free space in {staging_dir}: database {db_id!r} needs "
            f"{size / 2**30:.2f} GiB, only {free / 2**30:.2f} GiB is free"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{db_id}.", suffix=".sqlite", dir=staging_dir
    )
    os.close(descriptor)
    staged = Path(temporary_name)
    copied = 0
    last_report = time.monotonic()
    try:
        with source.open("rb") as source_file, staged.open("wb") as destination:
            while True:
                chunk = source_file.read(16 * 1024 * 1024)
                if not chunk:
                    break
                destination.write(chunk)
                copied += len(chunk)
                now = time.monotonic()
                if now - last_report >= 10:
                    print(
                        f"[fk-sample] staging {db_id}: {copied / 2**30:.2f}/"
                        f"{size / 2**30:.2f} GiB",
                        flush=True,
                    )
                    last_report = now
        return staged
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _table_names(connection: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [
        row[1]
        for row in connection.execute(f"PRAGMA table_info({_quote(table)})")
    ]


def _primary_key_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    info = connection.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
    return [row[1] for row in sorted(info, key=lambda value: value[5]) if row[5]]


def _pragma_foreign_keys(
    connection: sqlite3.Connection, table: str, canonical_tables: dict[str, str]
) -> list[ForeignKey]:
    grouped: dict[int, list[tuple]] = defaultdict(list)
    for row in connection.execute(f"PRAGMA foreign_key_list({_quote(table)})"):
        grouped[int(row[0])].append(row)

    result = []
    for rows in grouped.values():
        rows.sort(key=lambda value: int(value[1]))
        parent = canonical_tables.get(str(rows[0][2]).lower())
        if parent is None:
            continue
        child_columns = tuple(str(row[3]) for row in rows)
        parent_columns = tuple(str(row[4]) for row in rows if row[4] is not None)
        if len(parent_columns) != len(child_columns):
            parent_columns = tuple(_primary_key_columns(connection, parent))
        if len(parent_columns) != len(child_columns):
            continue
        result.append(ForeignKey(table, parent, child_columns, parent_columns))
    return result


def _schema_foreign_keys(
    schema: dict | None,
    live_tables: list[str],
    live_columns: dict[str, list[str]],
) -> list[ForeignKey]:
    """Fallback for Spider/BIRD databases whose SQLite DDL omits FKs."""
    if not schema:
        return []
    schema_tables = schema.get("table_names_original") or schema.get("table_names") or []
    schema_columns = schema.get("column_names_original") or schema.get("column_names") or []
    live_by_lower = {name.lower(): name for name in live_tables}
    result = []
    for pair in schema.get("foreign_keys", []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        child_index, parent_index = pair
        if not (
            isinstance(child_index, int)
            and isinstance(parent_index, int)
            and 0 <= child_index < len(schema_columns)
            and 0 <= parent_index < len(schema_columns)
        ):
            continue
        child_spec, parent_spec = schema_columns[child_index], schema_columns[parent_index]
        if len(child_spec) < 2 or len(parent_spec) < 2:
            continue
        child_table_index, child_column = child_spec[0], str(child_spec[1])
        parent_table_index, parent_column = parent_spec[0], str(parent_spec[1])
        if not (
            isinstance(child_table_index, int)
            and isinstance(parent_table_index, int)
            and 0 <= child_table_index < len(schema_tables)
            and 0 <= parent_table_index < len(schema_tables)
        ):
            continue
        child = live_by_lower.get(str(schema_tables[child_table_index]).lower())
        parent = live_by_lower.get(str(schema_tables[parent_table_index]).lower())
        if child is None or parent is None:
            continue
        child_live = {name.lower(): name for name in live_columns[child]}
        parent_live = {name.lower(): name for name in live_columns[parent]}
        child_column = child_live.get(child_column.lower())
        parent_column = parent_live.get(parent_column.lower())
        if child_column is None or parent_column is None:
            continue
        # Official Spider schema JSON does not retain composite-FK group IDs;
        # each pair is enforced independently, which is conservative and still
        # guarantees every emitted scalar reference is represented.
        result.append(ForeignKey(child, parent, (child_column,), (parent_column,)))
    return result


def _discover_foreign_keys(
    connection: sqlite3.Connection,
    tables: list[str],
    columns: dict[str, list[str]],
    schema: dict | None,
) -> list[ForeignKey]:
    canonical = {name.lower(): name for name in tables}
    pragma = [
        foreign_key
        for table in tables
        for foreign_key in _pragma_foreign_keys(connection, table, canonical)
    ]
    fallback = _schema_foreign_keys(schema, tables, columns)
    by_signature = {
        (
            fk.child.lower(), fk.parent.lower(),
            tuple(value.lower() for value in fk.child_columns),
            tuple(value.lower() for value in fk.parent_columns),
        ): fk
        for fk in fallback
    }
    # Live SQLite declarations take precedence over schema-JSON fallbacks.
    for fk in pragma:
        by_signature[
            (
                fk.child.lower(), fk.parent.lower(),
                tuple(value.lower() for value in fk.child_columns),
                tuple(value.lower() for value in fk.parent_columns),
            )
        ] = fk
    validated = []
    for fk in by_signature.values():
        child_live = {name.lower(): name for name in columns.get(fk.child, [])}
        parent_live = {name.lower(): name for name in columns.get(fk.parent, [])}
        missing_child = [
            name for name in fk.child_columns if name.lower() not in child_live
        ]
        missing_parent = [
            name for name in fk.parent_columns if name.lower() not in parent_live
        ]
        if missing_child or missing_parent:
            print(
                f"[fk-sample] WARNING: ignoring malformed FK "
                f"{fk.child}{fk.child_columns} -> {fk.parent}{fk.parent_columns}; "
                f"missing child columns={missing_child}, "
                f"missing parent columns={missing_parent}",
                flush=True,
            )
            continue
        validated.append(
            ForeignKey(
                child=fk.child,
                parent=fk.parent,
                child_columns=tuple(child_live[name.lower()] for name in fk.child_columns),
                parent_columns=tuple(
                    parent_live[name.lower()] for name in fk.parent_columns
                ),
            )
        )
    return sorted(
        validated,
        key=lambda fk: (fk.child.lower(), fk.parent.lower(), fk.child_columns),
    )


def _parent_first_order(tables: list[str], foreign_keys: list[ForeignKey]) -> list[str]:
    parents = {table: set() for table in tables}
    children: dict[str, set[str]] = defaultdict(set)
    for fk in foreign_keys:
        if fk.child != fk.parent:
            parents[fk.child].add(fk.parent)
            children[fk.parent].add(fk.child)
    remaining_parents = {table: len(values) for table, values in parents.items()}
    queue = deque(sorted(table for table, count in remaining_parents.items() if count == 0))
    order = []
    while queue:
        parent = queue.popleft()
        order.append(parent)
        for child in sorted(children.get(parent, ())):
            remaining_parents[child] -= 1
            if remaining_parents[child] == 0:
                queue.append(child)
    # Cyclic/self-referencing tables are initialized last and then pruned.
    order.extend(sorted(table for table in tables if table not in set(order)))
    return order


def _column_indices(columns: list[str], selected: Iterable[str]) -> tuple[int, ...]:
    by_lower = {name.lower(): index for index, name in enumerate(columns)}
    return tuple(by_lower[name.lower()] for name in selected)


def _keys(
    rows: list[SampledRow], columns: list[str], selected: tuple[str, ...]
) -> set[tuple[object, ...]]:
    indices = _column_indices(columns, selected)
    return {
        tuple(row.values[index] for index in indices)
        for row in rows
        if all(row.values[index] is not None for index in indices)
    }


def _eligibility_sql(
    outgoing: list[ForeignKey],
    sampled: dict[str, list[SampledRow]],
    columns: dict[str, list[str]],
) -> tuple[str, list[object]]:
    clauses = []
    parameters: list[object] = []
    for fk in outgoing:
        if fk.parent not in sampled:
            continue
        parent_keys = sorted(
            _keys(sampled[fk.parent], columns[fk.parent], fk.parent_columns),
            key=repr,
        )
        nullable = " OR ".join(f"{_quote(name)} IS NULL" for name in fk.child_columns)
        matches = []
        for key in parent_keys:
            matches.append(
                "(" + " AND ".join(
                    f"{_quote(name)} = ?" for name in fk.child_columns
                ) + ")"
            )
            parameters.extend(key)
        allowed = " OR ".join(matches)
        clauses.append(f"({nullable}{' OR ' + allowed if allowed else ''})")
    return (" AND ".join(clauses) if clauses else "1"), parameters


def _sample_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
    where_sql: str,
    parameters: list[object],
    max_rows: int,
    rng: random.Random,
) -> list[SampledRow]:
    quoted_columns = ", ".join(_quote(name) for name in columns)
    quoted_table = _quote(table)
    try:
        minimum, maximum = connection.execute(
            f"SELECT MIN(rowid), MAX(rowid) FROM {quoted_table}"
        ).fetchone()
    except sqlite3.OperationalError:
        return _sample_rows_without_rowid(
            connection,
            table,
            columns,
            where_sql,
            parameters,
            max_rows,
            rng,
        )
    if minimum is None or maximum is None:
        return []

    found: dict[int, SampledRow] = {}
    # Rejection sampling avoids ORDER BY RANDOM() and full-table shuffles.
    for _ in range(12):
        if len(found) >= max_rows:
            break
        draw_count = max(256, 8 * (max_rows - len(found)))
        candidates = {
            rng.randint(int(minimum), int(maximum)) for _ in range(draw_count)
        } - set(found)
        if not candidates:
            continue
        placeholders = ",".join("?" for _ in candidates)
        query = (
            f"SELECT rowid, {quoted_columns} FROM {quoted_table} "
            f"WHERE rowid IN ({placeholders}) AND ({where_sql})"
        )
        for result in connection.execute(query, [*sorted(candidates), *parameters]):
            found[int(result[0])] = SampledRow(int(result[0]), tuple(result[1:]))

    # Sparse FK eligibility can defeat rejection sampling.  A bounded,
    # deterministic fallback retrieves a candidate pool, not the full table.
    if len(found) < max_rows:
        pool_limit = max(1000, max_rows * 20)
        query = (
            f"SELECT rowid, {quoted_columns} FROM {quoted_table} "
            f"WHERE ({where_sql}) ORDER BY rowid LIMIT ?"
        )
        for result in connection.execute(query, [*parameters, pool_limit]):
            found.setdefault(
                int(result[0]), SampledRow(int(result[0]), tuple(result[1:]))
            )

    candidates = sorted(found.values(), key=lambda row: row.rowid)
    if len(candidates) > max_rows:
        candidates = rng.sample(candidates, max_rows)
    return sorted(candidates, key=lambda row: row.rowid)


def _sample_rows_without_rowid(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
    where_sql: str,
    parameters: list[object],
    max_rows: int,
    rng: random.Random,
) -> list[SampledRow]:
    """Uniform deterministic reservoir sample for WITHOUT ROWID tables.

    WITHOUT ROWID tables are physically ordered by their declared primary
    key, so this is a sequential local scan when ``--sqlite_staging_dir`` is
    used.  It is intentionally limited to the uncommon tables for which the
    much faster random-rowid path is unavailable.
    """
    quoted_table = _quote(table)
    quoted_columns = ", ".join(_quote(name) for name in columns)
    primary_key = _primary_key_columns(connection, table)
    if primary_key:
        order_sql = ", ".join(_quote(name) for name in primary_key)
        key_indices = _column_indices(columns, primary_key)
    else:
        # SQLite requires a primary key for true WITHOUT ROWID tables, but a
        # virtual table can also reject ``rowid``. Preserve deterministic scan
        # order as far as that module permits and use the scan ordinal.
        order_sql = ", ".join(_quote(name) for name in columns)
        key_indices = ()
    query = (
        f"SELECT {quoted_columns} FROM {quoted_table} "
        f"WHERE ({where_sql}) ORDER BY {order_sql}"
    )
    print(
        f"[fk-sample] table {table}: no implicit rowid; deterministic "
        f"reservoir sampling by {primary_key or 'full-row order'}",
        flush=True,
    )
    reservoir: list[SampledRow] = []
    seen = 0
    for result in connection.execute(query, parameters):
        values = tuple(result)
        locator: object
        if key_indices:
            locator = tuple(values[index] for index in key_indices)
        else:
            locator = seen
        row = SampledRow(locator, values)
        seen += 1
        if len(reservoir) < max_rows:
            reservoir.append(row)
            continue
        replacement = rng.randrange(seen)
        if replacement < max_rows:
            reservoir[replacement] = row
    return sorted(reservoir, key=lambda row: repr(row.rowid))


def _prune_to_fk_closure(
    sampled: dict[str, list[SampledRow]],
    columns: dict[str, list[str]],
    foreign_keys: list[ForeignKey],
) -> int:
    removed_total = 0
    while True:
        removed_this_round = 0
        for fk in foreign_keys:
            parent_keys = _keys(
                sampled.get(fk.parent, []), columns[fk.parent], fk.parent_columns
            )
            child_indices = _column_indices(columns[fk.child], fk.child_columns)
            retained = []
            for row in sampled.get(fk.child, []):
                key = tuple(row.values[index] for index in child_indices)
                # SQL FK semantics: any NULL component exempts the row.
                if any(value is None for value in key) or key in parent_keys:
                    retained.append(row)
                else:
                    removed_this_round += 1
            sampled[fk.child] = retained
        removed_total += removed_this_round
        if removed_this_round == 0:
            return removed_total


def _verify_fk_closure(
    sampled: dict[str, list[SampledRow]],
    columns: dict[str, list[str]],
    foreign_keys: list[ForeignKey],
    max_rows: int,
) -> None:
    for table, rows in sampled.items():
        if len(rows) > max_rows:
            raise AssertionError(f"row cap violated for {table!r}: {len(rows)}")
    violations = []
    for fk in foreign_keys:
        parent_keys = _keys(sampled[fk.parent], columns[fk.parent], fk.parent_columns)
        child_indices = _column_indices(columns[fk.child], fk.child_columns)
        for row in sampled[fk.child]:
            key = tuple(row.values[index] for index in child_indices)
            if not any(value is None for value in key) and key not in parent_keys:
                violations.append((fk, row.rowid, key))
    if violations:
        raise AssertionError(f"sample contains dangling foreign keys: {violations[:5]!r}")


def sample_database(
    connection: sqlite3.Connection,
    db_id: str,
    schema: dict | None,
    max_rows: int,
    seed: int,
) -> tuple[dict[str, list[SampledRow]], dict[str, list[str]], list[ForeignKey], int]:
    tables = _table_names(connection)
    columns = {table: _columns(connection, table) for table in tables}
    foreign_keys = _discover_foreign_keys(connection, tables, columns, schema)
    outgoing: dict[str, list[ForeignKey]] = defaultdict(list)
    for fk in foreign_keys:
        outgoing[fk.child].append(fk)

    sampled: dict[str, list[SampledRow]] = {}
    for table in _parent_first_order(tables, foreign_keys):
        where_sql, parameters = _eligibility_sql(
            outgoing.get(table, []), sampled, columns
        )
        sampled[table] = _sample_rows(
            connection,
            table,
            columns[table],
            where_sql,
            parameters,
            max_rows,
            _stable_rng(seed, db_id, table),
        )

    removed = _prune_to_fk_closure(sampled, columns, foreign_keys)
    _verify_fk_closure(sampled, columns, foreign_keys, max_rows)
    return sampled, columns, foreign_keys, removed


CHECKPOINT_FORMAT = "croissant_fk_database_checkpoint_v1"


def _database_checkpoint_path(directory: Path, db_id: str) -> Path:
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in db_id
    )
    digest = hashlib.sha256(db_id.encode()).hexdigest()[:12]
    return directory / f"{safe}.{digest}.json"


def _load_database_checkpoint(
    path: Path,
    db_id: str,
    requested_tables: list[str],
    max_rows: int,
    seed: int,
) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    expected = {
        "format": CHECKPOINT_FORMAT,
        "db_id": db_id,
        "max_rows": max_rows,
        "seed": seed,
        "requested_tables": requested_tables,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"stale/incompatible database checkpoint {path}: {mismatches}; "
            "use a new --database_checkpoint_dir or remove this checkpoint"
        )
    return payload


def materialize(
    dataset_root: Path,
    output_path: Path,
    max_rows: int,
    seed: int,
    sqlite_staging_dir: Path | None = None,
    database_checkpoint_dir: Path | None = None,
) -> dict:
    with (dataset_root / "configs/splits/corpus.json").open(encoding="utf-8") as source:
        corpus = json.load(source)
    with (dataset_root / "tables.json").open(encoding="utf-8") as source:
        raw_schemas = json.load(source)
    schema_records = raw_schemas.values() if isinstance(raw_schemas, dict) else raw_schemas
    schemas = {record["db_id"]: record for record in schema_records if "db_id" in record}

    requested_by_db: dict[str, list[str]] = defaultdict(list)
    for record in corpus["tables"]:
        requested_by_db[str(record["db_id"])].append(str(record["table_name"]))

    if database_checkpoint_dir is None:
        database_checkpoint_dir = output_path.with_name(
            output_path.name + ".db_checkpoints"
        )
    database_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    emitted_by_id = {}
    total_fks = total_rows = pruned_rows = empty_tables = 0
    for position, db_id in enumerate(sorted(requested_by_db), start=1):
        requested_tables = requested_by_db[db_id]
        checkpoint_path = _database_checkpoint_path(database_checkpoint_dir, db_id)
        checkpoint = _load_database_checkpoint(
            checkpoint_path, db_id, requested_tables, max_rows, seed
        )
        if checkpoint is not None:
            for table_record in checkpoint["tables"]:
                _record_db, _, table = table_record["table_id"].partition("#sep#")
                emitted_by_id[(db_id, table)] = table_record
            total_rows += int(checkpoint["n_rows"])
            total_fks += int(checkpoint["n_foreign_keys"])
            pruned_rows += int(checkpoint["n_rows_pruned_for_fk_closure"])
            empty_tables += int(checkpoint["n_empty_tables"])
            print(
                f"[fk-sample] resumed database {position}/{len(requested_by_db)}: "
                f"{db_id} from {checkpoint_path}",
                flush=True,
            )
            continue

        source_path = _sqlite_path(dataset_root, db_id)
        staged_path = None
        print(
            f"[fk-sample] database {position}/{len(requested_by_db)}: {db_id} "
            f"({source_path.stat().st_size / 2**20:.1f} MiB)"
            + ("; staging locally" if sqlite_staging_dir is not None else ""),
            flush=True,
        )
        try:
            database_path = source_path
            if sqlite_staging_dir is not None:
                staged_path = _stage_sqlite(source_path, sqlite_staging_dir, db_id)
                database_path = staged_path
            connection = sqlite3.connect(str(database_path))
            connection.execute("PRAGMA query_only=ON")
            try:
                sampled, columns, foreign_keys, removed = sample_database(
                    connection, db_id, schemas.get(db_id), max_rows, seed
                )
            finally:
                connection.close()
        finally:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)
        fk_child_columns = {
            (fk.child, column)
            for fk in foreign_keys
            for column in fk.child_columns
        }
        database_tables = []
        database_rows = 0
        database_empty_tables = 0
        for table in requested_tables:
            rows = sampled.get(table, [])
            table_columns = columns.get(table, [])
            if not rows:
                database_empty_tables += 1
            database_rows += len(rows)
            table_record = {
                "table_id": f"{db_id}#sep#{table}",
                "table_name": f"{db_id}#sep#{table}",
                "columns": [
                    {
                        "header": column,
                        "cells": [clean_text(row.values[index]) for row in rows],
                        "is_foreign_key": (table, column) in fk_child_columns,
                    }
                    for index, column in enumerate(table_columns)
                ],
            }
            database_tables.append(table_record)
            emitted_by_id[(db_id, table)] = table_record
        checkpoint = {
            "format": CHECKPOINT_FORMAT,
            "db_id": db_id,
            "max_rows": max_rows,
            "seed": seed,
            "requested_tables": requested_tables,
            "n_rows": database_rows,
            "n_foreign_keys": len(foreign_keys),
            "n_rows_pruned_for_fk_closure": removed,
            "n_empty_tables": database_empty_tables,
            "tables": database_tables,
        }
        _atomic_json(checkpoint_path, checkpoint)
        total_rows += database_rows
        total_fks += len(foreign_keys)
        pruned_rows += removed
        empty_tables += database_empty_tables
        print(
            f"[fk-sample] completed {position}/{len(requested_by_db)} databases; "
            f"rows={total_rows}, FKs={total_fks}, pruned={pruned_rows}; "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )

    emitted = [
        emitted_by_id[(str(record["db_id"]), str(record["table_name"]))]
        for record in corpus["tables"]
        if (str(record["db_id"]), str(record["table_name"])) in emitted_by_id
    ]
    _atomic_json(output_path, emitted)
    manifest = {
        "format": "croissant_fk_consistent_materialized_corpus",
        "dataset_root": str(dataset_root.resolve()),
        "output_path": str(output_path.resolve()),
        "seed": seed,
        "max_rows": max_rows,
        "n_databases": len(requested_by_db),
        "n_tables": len(emitted),
        "n_rows": total_rows,
        "n_foreign_keys": total_fks,
        "n_rows_pruned_for_fk_closure": pruned_rows,
        "n_empty_tables": empty_tables,
        "strict_fk_closure": True,
        "sqlite_staging_dir": (
            str(sqlite_staging_dir.resolve()) if sqlite_staging_dir is not None else None
        ),
        "database_checkpoint_dir": str(database_checkpoint_dir.resolve()),
    }
    _atomic_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_rows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sqlite_staging_dir",
        default=None,
        help="copy one database at a time here before sampling; use local /tmp "
        "to replace random NAS reads with one sequential NAS read",
    )
    parser.add_argument(
        "--database_checkpoint_dir",
        default=None,
        help="per-database resumable JSON checkpoints; defaults next to output_path",
    )
    args = parser.parse_args()
    if args.max_rows <= 0:
        parser.error("--max_rows must be positive")
    output_path = Path(args.output_path)
    if output_path.exists():
        parser.error(f"refusing to overwrite existing output: {output_path}")
    materialize(
        Path(args.dataset_root),
        output_path,
        args.max_rows,
        args.seed,
        Path(args.sqlite_staging_dir) if args.sqlite_staging_dir else None,
        Path(args.database_checkpoint_dir) if args.database_checkpoint_dir else None,
    )


if __name__ == "__main__":
    main()
