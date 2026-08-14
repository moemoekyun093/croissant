"""
For every table in a given database, compares the DIVERSITY (distinct
value count per column) of the first N rows (SQLite's default order --
what build_bird_jsonl.py's sample_table_rows currently pulls) against a
genuinely random N-row sample -- checks whether the corpus's displayed
content is an unrepresentative slice of a table, which would starve the
model of real signal for that table regardless of anything else.

Usage:
    python -m scripts.check_sample_diversity \
        --bird_db_root /mnt/nas/ayane/tables/dev_database \
        --db_id debit_card_specializing \
        --sample_size 50
"""

import argparse
import sqlite3


def get_sqlite_path(db_root: str, db_id: str) -> str:
    return f"{db_root}/{db_id}/{db_id}.sqlite"


def check_one_table(cur: sqlite3.Cursor, table_name: str, sample_size: int) -> None:
    cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    total_rows = cur.fetchone()[0]

    cur.execute(f'PRAGMA table_info("{table_name}")')
    columns = [row[1] for row in cur.fetchall()]

    cur.execute(f'SELECT * FROM "{table_name}" LIMIT {sample_size}')
    first_n = cur.fetchall()

    cur.execute(f'SELECT * FROM "{table_name}" ORDER BY RANDOM() LIMIT {sample_size}')
    random_n = cur.fetchall()

    print(f"\n== {table_name} ({total_rows} total rows, sample size {sample_size}) ==")
    print(f"{'column':<30} {'distinct (first-N)':>20} {'distinct (random-N)':>22}")
    print("-" * 74)

    any_gap = False
    for i, col in enumerate(columns):
        distinct_first = len(set(r[i] for r in first_n))
        distinct_random = len(set(r[i] for r in random_n))
        flag = "  <-- gap" if distinct_random > distinct_first * 2 and distinct_random > 3 else ""
        if flag:
            any_gap = True
        print(f"{col:<30} {distinct_first:>20} {distinct_random:>22}{flag}")

    if not any_gap:
        print("(no notable diversity gap for this table)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird_db_root", required=True)
    parser.add_argument("--db_id", required=True)
    parser.add_argument("--sample_size", type=int, default=50)
    args = parser.parse_args()

    conn = sqlite3.connect(get_sqlite_path(args.bird_db_root, args.db_id))
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = [row[0] for row in cur.fetchall()]

    print(f"found {len(table_names)} tables in {args.db_id}: {table_names}")

    for table_name in table_names:
        check_one_table(cur, table_name, args.sample_size)

    conn.close()