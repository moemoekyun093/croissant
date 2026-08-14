"""
Checks the saved bird_closest_matches.json for "attractor" columns --
target columns that show up as the top-1 match for many DIFFERENT,
structurally unrelated query columns. If a small number of columns
absorb a disproportionate share of matches, that's a symptom of
representation collapse: some vectors becoming generically "close to
everything" rather than genuinely discriminative.

Usage:
    python -m scripts.analyze_column_collapse \
        --matches_file eval/report_runs/run_v4_ctx_headers/bird_closest_matches.json
"""

import argparse
import json
from collections import Counter, defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches_file", required=True)
    parser.add_argument(
        "--top_n", type=int, default=15, help="how many top attractor columns to print"
    )
    args = parser.parse_args()

    with open(args.matches_file, "r", encoding="utf-8") as f:
        all_matches = json.load(f)

    # count how often each (table, column) appears as a MATCHED (target)
    # column, across the TOP-1 match's column_alignment for every query
    target_counts: Counter = Counter()
    target_source_queries: dict[tuple, set[str]] = defaultdict(set)

    total_alignments = 0

    for entry in all_matches:
        if not entry["top_matches"]:
            continue
        top1 = entry["top_matches"][0]
        for a in top1["column_alignment"]:
            key = (top1["table_name"], a["matched_column"])
            target_counts[key] += 1
            target_source_queries[key].add(
                f"{entry['query_table_name']}.{a['query_column']}"
            )
            total_alignments += 1

    print(f"total (query_column -> matched_column) pairs analyzed: {total_alignments}")
    print(f"distinct matched columns used: {len(target_counts)}")
    print()

    print(f"== top {args.top_n} most-frequently-matched columns ==")
    print("(a column matched by many DIFFERENT, unrelated query columns is a")
    print(" sign of representation collapse -- check whether the source query")
    print(" columns below actually look related to each other)")
    print()

    for (table_name, col), count in target_counts.most_common(args.top_n):
        sources = target_source_queries[(table_name, col)]
        print(f"'{table_name}'.'{col}': matched {count} times")
        for s in sorted(sources)[:8]:
            print(f"    <- {s}")
        if len(sources) > 8:
            print(f"    ... and {len(sources) - 8} more")
        print()