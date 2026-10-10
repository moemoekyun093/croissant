"""Filesystem path conventions for on-disk SQLite databases."""


def get_sqlite_path(db_root: str, db_id: str) -> str:
    """Return ``<db_root>/<db_id>/<db_id>.sqlite`` (BIRD and SynSQL-2.5M layout)."""
    return f"{db_root}/{db_id}/{db_id}.sqlite"
