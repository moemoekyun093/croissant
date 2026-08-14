"""
For one or more tables, breaks down null percentage and distinct-value
count PER COLUMN -- shows exactly which column(s) drive a table's low
"complete row" percentage from count_complete_rows.py, rather than just
knowing the table has a completeness problem without knowing where.

Usage (accepts multiple tables in one call):
    python -m scripts.table_null_breakdown \
        --bird_db_root /mnt/nas/ayane/tables/dev_database \
        --full_table_names "california_schools#sep#schools" "card_games#sep#cards" \
            "card_games#sep#sets" "thrombosis_prediction#sep#Laboratory" \
            "student_club#sep#income" "codebase_community#sep#comments" \
            "codebase_community#sep#posts"
"""

import argparse
import sqlite3


def get_sqlite_path(db_root: str, db_id: str) -> str:
    return f"{db_root}/{db_id}/{db_id}.sqlite"


def breakdown_one_table(bird_db_root: str, full_table_name: str) -> None:
    if "#sep#" not in full_table_name:
        print(f"[skip] {full_table_name!r} not in 'db_id#sep#table_name' form")
        return

    db_id, table_name = full_table_name.split("#sep#", 1)

    conn = sqlite3.connect(get_sqlite_path(bird_db_root, db_id))
    cur = conn.cursor()

    cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    total = cur.fetchone()[0]

    cur.execute(f'SELECT * FROM "{table_name}" LIMIT 0')
    columns = [desc[0] for desc in cur.description]

    print(f"\n== {full_table_name}  ({total} rows) ==")
    print(f"{'column':<45} {'non-null':>10} {'null %':>8} {'distinct':>10}")
    print("-" * 76)

    rows_out = []
    for col in columns:
        cur.execute(f'SELECT COUNT("{col}"), COUNT(DISTINCT "{col}") FROM "{table_name}"')
        non_null, distinct = cur.fetchone()
        null_pct = 100 * (total - non_null) / total if total > 0 else 0
        rows_out.append((col, non_null, null_pct, distinct))

    rows_out.sort(key=lambda r: -r[2])  # worst (most null) first

    for col, non_null, null_pct, distinct in rows_out:
        print(f"{col:<45} {non_null:>10} {null_pct:>7.1f}% {distinct:>10}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird_db_root", required=True)
    parser.add_argument(
        "--full_table_names", required=True, nargs="+",
        help="one or more 'db_id#sep#table_name' strings",
    )
    args = parser.parse_args()

    for full_table_name in args.full_table_names:
        try:
            breakdown_one_table(args.bird_db_root, full_table_name)
        except Exception as e:
            print(f"\n== {full_table_name} == [ERROR: {e}]")