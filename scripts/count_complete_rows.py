"""
For each BIRD table, counts how many rows would remain if only rows
with ALL columns non-null were kept -- checks whether a "sample only
complete rows" data-side strategy would leave enough rows to work with,
before deciding whether it's worth pursuing.

Usage:
    python -m scripts.count_complete_rows \
        --corpus_jsonl /mnt/nas/ayane/tables/big_corpus.jsonl \
        --bird_db_root /mnt/nas/ayane/tables/dev_database
"""

import argparse
import json
import sqlite3


def get_sqlite_path(db_root: str, db_id: str) -> str:
    return f"{db_root}/{db_id}/{db_id}.sqlite"


def load_bird_records(corpus_jsonl: str) -> list[dict]:
    records = []
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("source") == "bird":
                records.append(record)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_jsonl", required=True)
    parser.add_argument("--bird_db_root", required=True)
    args = parser.parse_args()

    records = load_bird_records(args.corpus_jsonl)
    print(f"{'table':<55} {'total':>8} {'complete':>10} {'pct':>7}")
    print("-" * 82)

    for record in records:
        full_table_name = record["table_name"]
        if "#sep#" not in full_table_name:
            continue
        db_id, table_name = full_table_name.split("#sep#", 1)

        try:
            conn = sqlite3.connect(get_sqlite_path(args.bird_db_root, db_id))
            cur = conn.cursor()

            cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            total = cur.fetchone()[0]

            cur.execute(f'SELECT * FROM "{table_name}" LIMIT 0')
            columns = [desc[0] for desc in cur.description]

            where_clause = " AND ".join(f'"{c}" IS NOT NULL' for c in columns)
            cur.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE {where_clause}')
            complete = cur.fetchone()[0]

            conn.close()
        except Exception as e:
            print(f"{full_table_name:<55} [ERROR: {e}]")
            continue

        pct = 100 * complete / total if total > 0 else 0
        print(f"{full_table_name:<55} {total:>8} {complete:>10} {pct:>6.1f}%")