"""CPU-only deterministic closure check for FK-consistent row sampling."""

from __future__ import annotations

import sqlite3
from pathlib import Path
import tempfile

from scripts.materialize_fk_consistent_corpus import (
    _atomic_json,
    _stage_sqlite,
    materialize,
    sample_database,
)


def _snapshot(sampled):
    return {
        table: [(row.rowid, row.values) for row in rows]
        for table, rows in sampled.items()
    }


def main() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        PRAGMA foreign_keys=OFF;
        CREATE TABLE parent(id INTEGER PRIMARY KEY, label TEXT);
        CREATE TABLE child(
            id INTEGER PRIMARY KEY,
            parent_id INTEGER,
            value TEXT,
            FOREIGN KEY(parent_id) REFERENCES parent(id)
        );
        CREATE TABLE grandchild(
            id INTEGER PRIMARY KEY,
            child_id INTEGER,
            FOREIGN KEY(child_id) REFERENCES child(id)
        );
        CREATE TABLE node(
            id INTEGER PRIMARY KEY,
            parent_id INTEGER,
            FOREIGN KEY(parent_id) REFERENCES node(id)
        );
        CREATE TABLE schema_parent(id INTEGER PRIMARY KEY);
        CREATE TABLE schema_child(id INTEGER PRIMARY KEY, parent_id INTEGER);
        CREATE TABLE wr_parent(
            left_id INTEGER,
            right_id INTEGER,
            label TEXT,
            PRIMARY KEY(left_id, right_id)
        ) WITHOUT ROWID;
        CREATE TABLE wr_child(
            id INTEGER PRIMARY KEY,
            left_id INTEGER,
            right_id INTEGER,
            FOREIGN KEY(left_id, right_id)
              REFERENCES wr_parent(left_id, right_id)
        );
        CREATE TABLE malformed_parent(id INTEGER PRIMARY KEY);
        CREATE TABLE malformed_child(
            id INTEGER PRIMARY KEY,
            team_id INTEGER,
            FOREIGN KEY(team_id) REFERENCES malformed_parent(team_id)
        );
        """
    )
    connection.executemany(
        "INSERT INTO parent(id, label) VALUES (?, ?)",
        [(index, f"p{index}") for index in range(1, 101)],
    )
    connection.executemany(
        "INSERT INTO child(id, parent_id, value) VALUES (?, ?, ?)",
        [
            (index, (index - 1) % 100 + 1, f"c{index}")
            for index in range(1, 1001)
        ]
        + [(1001, 9999, "broken"), (1002, None, "nullable")],
    )
    connection.executemany(
        "INSERT INTO grandchild(id, child_id) VALUES (?, ?)",
        [(index, index) for index in range(1, 1001)]
        + [(1001, 9999), (1002, None)],
    )
    connection.executemany(
        "INSERT INTO node(id, parent_id) VALUES (?, ?)",
        [(1, None)] + [(index, index - 1) for index in range(2, 101)],
    )
    connection.executemany(
        "INSERT INTO schema_parent(id) VALUES (?)",
        [(index,) for index in range(1, 101)],
    )
    connection.executemany(
        "INSERT INTO schema_child(id, parent_id) VALUES (?, ?)",
        [(index, (index - 1) % 100 + 1) for index in range(1, 1001)],
    )
    connection.executemany(
        "INSERT INTO wr_parent(left_id, right_id, label) VALUES (?, ?, ?)",
        [(index, index + 1000, f"wr{index}") for index in range(1, 101)],
    )
    connection.executemany(
        "INSERT INTO wr_child(id, left_id, right_id) VALUES (?, ?, ?)",
        [(index, (index - 1) % 100 + 1, (index - 1) % 100 + 1001)
         for index in range(1, 1001)],
    )
    connection.executemany(
        "INSERT INTO malformed_parent(id) VALUES (?)", [(1,), (2,)]
    )
    connection.executemany(
        "INSERT INTO malformed_child(id, team_id) VALUES (?, ?)", [(1, 1), (2, 2)]
    )

    # Spider-style schema fallback: this FK is absent from the SQLite DDL.
    schema = {
        "table_names_original": ["schema_parent", "schema_child"],
        "column_names_original": [
            [-1, "*"],
            [0, "id"],
            [1, "id"],
            [1, "parent_id"],
        ],
        "foreign_keys": [[3, 1]],
    }

    first, _columns, foreign_keys, _removed = sample_database(
        connection, "toy", schema, max_rows=7, seed=42
    )
    second, _columns, _foreign_keys, _removed = sample_database(
        connection, "toy", schema, max_rows=7, seed=42
    )
    assert _snapshot(first) == _snapshot(second)
    assert foreign_keys
    assert not any(
        fk.child == "malformed_child" and fk.parent == "malformed_parent"
        for fk in foreign_keys
    )
    assert all(len(rows) <= 7 for rows in first.values())

    parent_ids = {row.values[0] for row in first["parent"]}
    child_ids = {row.values[0] for row in first["child"]}
    assert all(
        row.values[1] is None or row.values[1] in parent_ids
        for row in first["child"]
    )
    assert all(
        row.values[1] is None or row.values[1] in child_ids
        for row in first["grandchild"]
    )
    node_ids = {row.values[0] for row in first["node"]}
    assert all(
        row.values[1] is None or row.values[1] in node_ids
        for row in first["node"]
    )
    schema_parent_ids = {row.values[0] for row in first["schema_parent"]}
    assert all(
        row.values[1] is None or row.values[1] in schema_parent_ids
        for row in first["schema_child"]
    )
    wr_parent_keys = {(row.values[0], row.values[1]) for row in first["wr_parent"]}
    assert all(
        (row.values[1], row.values[2]) in wr_parent_keys
        for row in first["wr_child"]
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.sqlite"
        source.write_bytes(b"sqlite-staging-test" * 1024)
        staged = _stage_sqlite(source, root / "stage", "toy")
        assert staged.read_bytes() == source.read_bytes()
        staged.unlink()

    # A failed database is recorded without publishing an incomplete corpus;
    # after fixing it, the next invocation resumes the successful checkpoint.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "configs/splits").mkdir(parents=True)
        requested = {
            "tables": [
                {"db_id": "good", "table_name": "items"},
                {"db_id": "missing", "table_name": "items"},
            ]
        }
        schemas = [
            {"db_id": name, "table_names_original": ["items"], "foreign_keys": []}
            for name in ("good", "missing")
        ]
        _atomic_json(root / "configs/splits/corpus.json", requested)
        _atomic_json(root / "tables.json", schemas)

        def create_database(name: str) -> None:
            path = root / "databases" / name / f"{name}.sqlite"
            path.parent.mkdir(parents=True)
            database = sqlite3.connect(path)
            database.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, value TEXT)")
            database.executemany(
                "INSERT INTO items(value) VALUES (?)", [("a",), ("b",), ("c",)]
            )
            database.commit()
            database.close()

        create_database("good")
        output = root / "corpus.materialized.json"
        checkpoints = root / "checkpoints"
        incomplete = materialize(
            root, output, max_rows=2, seed=42,
            database_checkpoint_dir=checkpoints, max_database_attempts=1,
        )
        assert incomplete["failed_databases"] == 1
        assert not output.exists()

        create_database("missing")
        complete = materialize(
            root, output, max_rows=2, seed=42,
            database_checkpoint_dir=checkpoints, max_database_attempts=1,
        )
        assert complete["n_databases"] == 2
        assert output.exists()
    print("FK-consistent sampling test passed")


if __name__ == "__main__":
    main()
