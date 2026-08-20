"""CPU-only checks for NFS-free SQL/schema normalization fast paths."""

from __future__ import annotations

from pathlib import Path
import tempfile

from scripts.normalize_sql_datasets import (
    SQLiteTableResolver,
    _schema_table_names,
    _sqlite_files_from_schema,
    _tables_from_sql_syntax,
)


def main() -> None:
    names = ["students", "classes", "class rooms", "unused"]
    sql = """
        WITH active AS (
          SELECT s.id FROM students AS s
          JOIN classes c ON c.student_id = s.id
        )
        SELECT * FROM active
        JOIN "class rooms" AS r ON r.id = active.id
        WHERE r.note = 'unused, FROM unused'
    """
    assert set(_tables_from_sql_syntax(sql, names)) == {
        "students", "classes", "class rooms"
    }
    assert set(
        _tables_from_sql_syntax(
            "SELECT * FROM main.students s, [classes] c WHERE s.id=c.student_id",
            names,
        )
    ) == {"students", "classes"}
    assert _tables_from_sql_syntax('SELECT * FROM "from"', ["from"]) == ["from"]
    assert _schema_table_names(
        {"table_names_original": ["Students", "Classes"]}
    ) == ["Students", "Classes"]

    # A nonexistent SQLite path proves the resolver did not open SQLite when
    # syntax extraction accounts for every schema-table name in the SQL.
    resolver = SQLiteTableResolver(
        {"toy": __import__("pathlib").Path("/path/that/does/not/exist.sqlite")},
        {"toy": names},
    )
    assert resolver.tables("toy", sql) == ["students", "classes", "class rooms"]
    assert resolver.fast_path_queries == 1
    assert resolver.sqlite_fallback_queries == 0

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        database = root / "toy" / "toy.sqlite"
        database.parent.mkdir()
        database.touch()
        assert _sqlite_files_from_schema(
            (root,), {"toy": {"db_id": "toy"}}
        ) == {"toy": database}
    print("Normalization metadata/SQL fast-path tests passed")


if __name__ == "__main__":
    main()
