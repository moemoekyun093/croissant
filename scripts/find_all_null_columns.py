"""
Finds every column, WITHIN A GIVEN SET OF TABLES, that is ENTIRELY null
across the WHOLE table -- and reports each one's DECLARED SQL type
(from PRAGMA table_info), plus a summary count by type. Numeric/date
types have a well-defined "random value in a plausible range" fallback;
text types don't (no real content anywhere to infer format/vocabulary
from) -- this survey tells you how much of each category you're
actually dealing with before deciding a strategy.

Deliberately scoped to a specific table list (default: the 7 tables
already found to have 0.0% complete rows via count_complete_rows.py)
rather than re-scanning the full 75-table BIRD corpus -- no reason to
re-check tables already known not to have this problem.

Usage:
    python -m scripts.find_all_null_columns \
        --bird_db_root /mnt/nas/ayane/tables/dev_database
    # (uses the default 7-table list -- override with --full_table_names)
"""

import argparse
import sqlite3
from collections import Counter

DEFAULT_TABLES = [
    "california_schools#sep#schools",
    "card_games#sep#cards",
    "card_games#sep#sets",
    "thrombosis_prediction#sep#Laboratory",
    "student_club#sep#income",
    "codebase_community#sep#comments",
    "codebase_community#sep#posts",
]


def get_sqlite_path(db_root: str, db_id: str) -> str:
    return f"{db_root}/{db_id}/{db_id}.sqlite"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird_db_root", required=True)
    parser.add_argument(
        "--full_table_names", nargs="+", default=DEFAULT_TABLES,
        help="one or more 'db_id#sep#table_name' strings "
        "(default: the 7 tables already known to have 0.0% complete rows)",
    )
    args = parser.parse_args()

    all_null_columns = []  # (table_name, column_name, declared_type, total_rows)

    for full_table_name in args.full_table_names:
        if "#sep#" not in full_table_name:
            print(f"[skip] {full_table_name!r} not in 'db_id#sep#table_name' form")
            continue
        db_id, table_name = full_table_name.split("#sep#", 1)

        try:
            conn = sqlite3.connect(get_sqlite_path(args.bird_db_root, db_id))
            cur = conn.cursor()

            cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            total = cur.fetchone()[0]

            cur.execute(f'PRAGMA table_info("{table_name}")')
            schema_info = cur.fetchall()  # (cid, name, type, notnull, dflt_value, pk)

            for _, col_name, declared_type, _, _, _ in schema_info:
                cur.execute(f'SELECT COUNT("{col_name}") FROM "{table_name}"')
                non_null = cur.fetchone()[0]
                if non_null == 0 and total > 0:
                    all_null_columns.append((full_table_name, col_name, declared_type, total))

            conn.close()
        except Exception as e:
            print(f"[ERROR] {full_table_name}: {e}")
            continue

    print(f"found {len(all_null_columns)} fully-null columns across {len(args.full_table_names)} tables\n")

    print(f"{'table':<45} {'column':<30} {'declared_type':<15} {'rows':>8}")
    print("-" * 100)
    for table_name, col_name, declared_type, total in all_null_columns:
        print(f"{table_name:<45} {col_name:<30} {declared_type or '(none)':<15} {total:>8}")

    print()
    type_counts = Counter(
        (dt.upper() if dt else "(none)") for _, _, dt, _ in all_null_columns
    )
    print("== summary by declared type ==")
    for t, count in type_counts.most_common():
        print(f"  {t:<20} {count}")