"""Normalize official BIRD and Spider downloads for Croissant training.

The output for each dataset is compatible with ``SynSQLTableDataset`` and
``SynSQLQueryDataset``::

    <output_root>/<bird|spider>/
      databases/<db_id>/<db_id>.sqlite
      databases.schema_cache.json
      questions_with_tables.json
      questions_with_tables.resolved.json
      tables.json
      configs/splits/query_split.json
      configs/splits/corpus.json
      normalization_manifest.json

Only queries are split.  Every split uses the same complete table corpus.
Database files are symlinked by default so normalization does not duplicate
the official SQLite downloads on the NAS.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sqlite3
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
    os.replace(partial, path)


def _load_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _first_existing(candidates: Iterable[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    rendered = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not find {label}; checked:\n  {rendered}")


def _sqlite_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"database directory does not exist: {root}")
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*.sqlite")):
        db_id = path.stem
        previous = result.get(db_id)
        if previous is not None and previous.resolve() != path.resolve():
            raise ValueError(
                f"database id {db_id!r} resolves to two SQLite files: "
                f"{previous} and {path}"
            )
        result[db_id] = path
    if not result:
        raise ValueError(f"no .sqlite files found under {root}")
    return result


def _merge_sqlite_roots(roots: Iterable[Path]) -> dict[str, Path]:
    merged: dict[str, Path] = {}
    for root in roots:
        for db_id, path in _sqlite_files(root).items():
            previous = merged.get(db_id)
            if previous is not None and previous.resolve() != path.resolve():
                raise ValueError(
                    f"database id {db_id!r} occurs in multiple source roots: "
                    f"{previous} and {path}"
                )
            merged[db_id] = path
    return merged


def _schema_from_sqlite(path: Path) -> tuple[list[str], dict[str, list[str]], dict]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        table_names = [row[0] for row in rows]
        columns: dict[str, list[str]] = {}
        column_names_original = [[-1, "*"]]
        column_types = ["text"]
        for table_index, table_name in enumerate(table_names):
            escaped = table_name.replace('"', '""')
            info = connection.execute(f'PRAGMA table_info("{escaped}")').fetchall()
            columns[table_name] = [row[1] for row in info]
            for row in info:
                column_names_original.append([table_index, row[1]])
                column_types.append((row[2] or "text").lower())
        schema = {
            "db_id": path.stem,
            "table_names_original": table_names,
            "table_names": table_names,
            "column_names_original": column_names_original,
            "column_names": column_names_original,
            "column_types": column_types,
        }
        return table_names, columns, schema
    finally:
        connection.close()


def _canonical_tables(names: Iterable[str], actual_names: list[str]) -> list[str]:
    by_lower = {name.lower(): name for name in actual_names}
    result = []
    seen = set()
    for name in names:
        canonical = by_lower.get(str(name).lower())
        if canonical is not None and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def _tables_from_spider_ast(sql_tree, schema_record: dict) -> list[str]:
    """Collect every table unit, including nested/union/intersect queries."""
    table_names = schema_record.get("table_names_original", [])
    indices: set[int] = set()

    def visit(value) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            if (
                len(value) >= 2
                and value[0] == "table_unit"
                and isinstance(value[1], int)
            ):
                indices.add(value[1])
            for child in value:
                visit(child)

    visit(sql_tree)
    return [table_names[index] for index in sorted(indices) if index < len(table_names)]


def _tables_from_connection(
    sql: str, connection: sqlite3.Connection, actual_names: list[str]
) -> list[str]:
    """Ask SQLite which base tables a read-only query references."""
    reads: list[str] = []

    def authorize(action, arg1, _arg2, _db_name, _trigger):
        if action == sqlite3.SQLITE_READ and arg1:
            reads.append(arg1)
        return sqlite3.SQLITE_OK

    try:
        connection.set_authorizer(authorize)
        connection.execute("EXPLAIN " + sql).fetchall()
    except sqlite3.Error:
        # BIRD contains a small number of dialect/annotation irregularities.
        # The conservative name matcher below is only a fallback for those.
        pass
    finally:
        connection.set_authorizer(None)

    canonical = _canonical_tables(reads, actual_names)
    if canonical:
        return canonical

    # Remove string literals before matching schema table names so a value
    # such as 'concerts' is not mistaken for the concerts table.
    scrubbed = re.sub(r"'(?:''|[^'])*'", " ", sql)
    matched = []
    for name in sorted(actual_names, key=len, reverse=True):
        quoted = re.escape(name)
        patterns = (
            rf'"{quoted}"',
            rf'`{quoted}`',
            rf'\[{quoted}\]',
            rf'(?<![A-Za-z0-9_]){quoted}(?![A-Za-z0-9_])',
        )
        if any(re.search(pattern, scrubbed, flags=re.IGNORECASE) for pattern in patterns):
            matched.append(name)
    return matched


class SQLiteTableResolver:
    """Resolve SQL table references with a bounded reusable connection LRU."""

    def __init__(
        self,
        sqlite_paths: dict[str, Path],
        table_names: dict[str, list[str]],
        max_open_connections: int = 32,
    ):
        self.sqlite_paths = sqlite_paths
        self.table_names = table_names
        self.max_open_connections = max_open_connections
        self.connections: OrderedDict[str, sqlite3.Connection] = OrderedDict()

    def _connection(self, db_id: str) -> sqlite3.Connection:
        connection = self.connections.pop(db_id, None)
        if connection is not None:
            self.connections[db_id] = connection
            return connection
        while len(self.connections) >= self.max_open_connections:
            _old_db, old_connection = self.connections.popitem(last=False)
            old_connection.close()
        connection = sqlite3.connect(self.sqlite_paths[db_id])
        connection.execute("PRAGMA query_only=ON")
        self.connections[db_id] = connection
        return connection

    def tables(self, db_id: str, sql: str) -> list[str]:
        return _tables_from_connection(
            sql, self._connection(db_id), self.table_names[db_id]
        )

    def close(self) -> None:
        for connection in self.connections.values():
            connection.close()
        self.connections.clear()


def _question_record(
    record: dict,
    source_dataset: str,
    source_split: str,
    actual_names: list[str],
    schema_record: dict,
    resolver: SQLiteTableResolver,
) -> dict | None:
    question = str(record.get("question", "")).strip()
    db_id = str(record.get("db_id", "")).strip()
    sql = record.get("SQL") or record.get("query") or ""
    if not question or not db_id or not sql:
        return None

    tables = []
    if source_dataset == "spider" and record.get("sql") is not None:
        tables = _tables_from_spider_ast(record["sql"], schema_record)
    if not tables:
        tables = resolver.tables(db_id, str(sql))
    tables = _canonical_tables(tables, actual_names)
    if not tables:
        return None

    return {
        "question": question,
        "db_id": db_id,
        "required_tables": tables,
        "style": source_dataset,
        "source_split": source_split,
        "sql": str(sql),
    }


def _split_queries(
    questions: list[dict], seed: int, train_frac: float, val_frac: float, test_frac: float
) -> dict:
    total_fraction = train_frac + val_frac + test_frac
    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError("train/val/test fractions must sum to one")
    indices = list(range(len(questions)))
    random.Random(seed).shuffle(indices)
    n_train = round(len(indices) * train_frac)
    n_val = round(len(indices) * val_frac)
    partitions = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    def split_records(selected: list[int]) -> list[dict]:
        return [
            {
                "question": questions[index]["question"],
                "db_id": questions[index]["db_id"],
                "table_names": questions[index]["required_tables"],
            }
            for index in selected
        ]

    return {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "n_total": len(indices),
        "n_train": len(partitions["train"]),
        "n_val": len(partitions["val"]),
        "n_test": len(partitions["test"]),
        **{name: split_records(values) for name, values in partitions.items()},
    }


def _install_database(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise FileExistsError(
            f"refusing to replace existing normalized database: {destination}"
        )
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def normalize_dataset(
    dataset: str,
    question_sources: list[tuple[str, Path]],
    sqlite_sources: dict[str, Path],
    source_schema_records: dict[str, dict],
    output_dir: Path,
    seed: int,
    fractions: tuple[float, float, float],
    database_mode: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    database_dir = output_dir / "databases"

    schemas: dict[str, dict] = {}
    table_names_by_db: dict[str, list[str]] = {}
    for position, (db_id, sqlite_path) in enumerate(sorted(sqlite_sources.items()), start=1):
        actual_names, _columns, generated_schema = _schema_from_sqlite(sqlite_path)
        generated_schema["db_id"] = db_id
        schemas[db_id] = source_schema_records.get(db_id, generated_schema)
        # Always use SQLite's canonical names for supervision and corpus.
        schemas[db_id]["db_id"] = db_id
        table_names_by_db[db_id] = actual_names
        _install_database(
            sqlite_path, database_dir / db_id / f"{db_id}.sqlite", database_mode
        )
        if position % 100 == 0 or position == len(sqlite_sources):
            print(
                f"[normalize-{dataset}] installed {position}/{len(sqlite_sources)} databases",
                flush=True,
            )

    questions: list[dict] = []
    skipped = Counter()
    seen = set()
    resolver = SQLiteTableResolver(sqlite_sources, table_names_by_db)
    try:
        for source_split, question_path in question_sources:
            records = _load_json(question_path)
            for record in records:
                db_id = str(record.get("db_id", "")).strip()
                if db_id not in sqlite_sources:
                    skipped["missing_database"] += 1
                    continue
                normalized = _question_record(
                    record,
                    dataset,
                    source_split,
                    table_names_by_db[db_id],
                    source_schema_records.get(db_id, schemas[db_id]),
                    resolver,
                )
                if normalized is None:
                    skipped["missing_question_sql_or_tables"] += 1
                    continue
                key = (
                    normalized["question"],
                    normalized["db_id"],
                    tuple(sorted(normalized["required_tables"])),
                )
                if key in seen:
                    skipped["duplicate"] += 1
                    continue
                seen.add(key)
                questions.append(normalized)
    finally:
        resolver.close()

    if not questions:
        raise ValueError(f"normalization produced no valid {dataset} questions")

    train_frac, val_frac, test_frac = fractions
    split = _split_queries(
        questions, seed, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
    )
    corpus_tables = [
        {"db_id": db_id, "table_name": table_name}
        for db_id in sorted(table_names_by_db)
        for table_name in sorted(table_names_by_db[db_id])
    ]
    resolved = [
        {
            "question": record["question"],
            "db_id": record["db_id"],
            "table_names": record["required_tables"],
        }
        for record in questions
    ]

    _atomic_json(output_dir / "questions_with_tables.json", questions)
    _atomic_json(output_dir / "questions_with_tables.resolved.json", resolved)
    _atomic_json(output_dir / "tables.json", [schemas[key] for key in sorted(schemas)])
    _atomic_json(output_dir / "databases.schema_cache.json", table_names_by_db)
    _atomic_json(output_dir / "configs/splits/query_split.json", split)
    _atomic_json(
        output_dir / "configs/splits/corpus.json",
        {"seed": seed, "corpus_size": None, "tables": corpus_tables},
    )
    manifest = {
        "dataset": dataset,
        "database_mode": database_mode,
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "n_databases": len(sqlite_sources),
        "n_tables": len(corpus_tables),
        "n_questions": len(questions),
        "n_train": split["n_train"],
        "n_val": split["n_val"],
        "n_test": split["n_test"],
        "skipped": dict(skipped),
        "question_sources": [str(path.resolve()) for _name, path in question_sources],
    }
    _atomic_json(output_dir / "normalization_manifest.json", manifest)
    print(
        f"[normalize-{dataset}] complete: {len(sqlite_sources)} databases, "
        f"{len(corpus_tables)} tables, {len(questions)} questions; "
        f"split={split['n_train']}/{split['n_val']}/{split['n_test']}; "
        f"skipped={dict(skipped)}",
        flush=True,
    )
    return manifest


def _schema_records(paths: Iterable[Path]) -> dict[str, dict]:
    result = {}
    for path in paths:
        raw = _load_json(path)
        records = raw.values() if isinstance(raw, dict) else raw
        for record in records:
            if "db_id" in record:
                result[record["db_id"]] = record
    return result


def _bird_inputs(root: Path):
    train_dir = root / "train"
    dev_dir = root / "dev_20240627"
    train_json = _first_existing((train_dir / "train.json", root / "train.json"), "BIRD train.json")
    dev_json = _first_existing((dev_dir / "dev.json", root / "dev.json"), "BIRD dev.json")
    train_tables = _first_existing(
        (train_dir / "train_tables.json", root / "train_tables.json"),
        "BIRD train_tables.json",
    )
    dev_tables = _first_existing(
        (dev_dir / "dev_tables.json", root / "dev_tables.json"),
        "BIRD dev_tables.json",
    )
    train_databases = _first_existing(
        (train_dir / "train_databases", root / "train_databases"),
        "BIRD train_databases",
    )
    dev_databases = _first_existing(
        (dev_dir / "dev_databases", root / "dev_databases"),
        "BIRD dev_databases",
    )
    return (
        [("official_train", train_json), ("official_dev", dev_json)],
        _merge_sqlite_roots((train_databases, dev_databases)),
        _schema_records((train_tables, dev_tables)),
    )


def _spider_inputs(root: Path):
    data = root / "spider_data" if (root / "spider_data").is_dir() else root
    train_json = _first_existing((data / "train_spider.json",), "Spider train_spider.json")
    dev_json = _first_existing((data / "dev.json",), "Spider dev.json")
    tables_json = _first_existing((data / "tables.json",), "Spider tables.json")
    databases = _first_existing(
        (data / "database", data / "databases"), "Spider database directory"
    )
    return (
        [("official_train", train_json), ("official_dev", dev_json)],
        _sqlite_files(databases),
        _schema_records((tables_json,)),
    )


def _materialize_normalized_corpus(dataset_dir: Path) -> None:
    """Build the standard sequential table-content cache once."""
    from src.data.synsql_dataset import SynSQLTableDataset

    corpus_path = dataset_dir / "configs/splits/corpus.json"
    cache_path = dataset_dir / "configs/splits/corpus.materialized.json"
    if cache_path.exists():
        print(
            f"[normalize-{dataset_dir.name}] materialized corpus already exists: "
            f"{cache_path}",
            flush=True,
        )
        return
    table_dataset = SynSQLTableDataset(
        databases_root=str(dataset_dir / "databases"),
        tables_json=str(dataset_dir / "tables.json"),
        max_rows=50,
        seed=42,
        max_open_connections=32,
    )
    tables = table_dataset.load_corpus(
        str(corpus_path), materialized_cache_path=str(cache_path)
    )
    table_dataset.close_connections()
    print(
        f"[normalize-{dataset_dir.name}] materialized {len(tables)} tables to "
        f"{cache_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird_root", default=None)
    parser.add_argument("--spider_root", default=None)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument(
        "--database_mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="symlink avoids duplicating the official SQLite files",
    )
    parser.add_argument(
        "--materialize_corpus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also build configs/splits/corpus.materialized.json (default: on)",
    )
    args = parser.parse_args()
    if args.bird_root is None and args.spider_root is None:
        parser.error("pass --bird_root and/or --spider_root")

    fractions = (args.train_frac, args.val_frac, args.test_frac)
    output_root = Path(args.output_root)
    manifests = {}
    if args.bird_root is not None:
        bird_output = output_root / "bird"
        manifests["bird"] = normalize_dataset(
            "bird",
            *_bird_inputs(Path(args.bird_root)),
            bird_output,
            args.seed,
            fractions,
            args.database_mode,
        )
        if args.materialize_corpus:
            _materialize_normalized_corpus(bird_output)
    if args.spider_root is not None:
        spider_output = output_root / "spider"
        manifests["spider"] = normalize_dataset(
            "spider",
            *_spider_inputs(Path(args.spider_root)),
            spider_output,
            args.seed,
            fractions,
            args.database_mode,
        )
        if args.materialize_corpus:
            _materialize_normalized_corpus(spider_output)
    _atomic_json(output_root / "normalization_summary.json", manifests)


if __name__ == "__main__":
    main()
