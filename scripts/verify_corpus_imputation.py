"""
Checks null/empty-cell percentage PER COLUMN directly in the generated
corpus JSONL -- the actual data training reads -- rather than the
original SQLite databases (which are never modified, and always show
the same null pattern regardless of any corpus-generation fix).

Usage:
    python -m scripts.verify_corpus_imputation \
        --corpus_jsonl /mnt/nas/ayane/tables/big_corpus.jsonl \
        --full_table_names "codebase_community#sep#posts" "card_games#sep#cards"
    # (omit --full_table_names to check ALL BIRD tables in the corpus)
"""

import argparse
import json

from src.data.corpus_loader import table_from_corpus_record

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_jsonl", required=True)
    parser.add_argument(
        "--full_table_names", nargs="*", default=None,
        help="specific 'db_id#sep#table_name' strings to check "
        "(default: all BIRD tables found in the corpus)",
    )
    args = parser.parse_args()

    wanted = set(args.full_table_names) if args.full_table_names else None

    print(f"{'table':<45} {'column':<30} {'total':>7} {'empty':>7} {'empty %':>8}")
    print("-" * 100)

    checked_tables = 0

    with open(args.corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("source") != "bird":
                continue
            if wanted is not None and record["table_name"] not in wanted:
                continue

            table = table_from_corpus_record(record)
            if table.num_columns == 0:
                continue

            checked_tables += 1
            for col in table.columns:
                total = len(col.cells)
                empty = sum(1 for c in col.cells if c.strip() == "")
                if empty > 0:
                    pct = 100 * empty / total if total > 0 else 0
                    print(
                        f"{table.table_name:<45} {col.header:<30} "
                        f"{total:>7} {empty:>7} {pct:>7.1f}%"
                    )

    print(f"\nchecked {checked_tables} BIRD table(s) from the corpus")
    print("(only columns with remaining empty cells are listed above --")
    print(" no rows printed for a table means it's fully imputed)")