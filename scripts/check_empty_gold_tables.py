"""
Checks whether any query's gold (positive) table is silently unusable at
validation/test time due to being empty (0 rows or 0 columns).

Why this can happen: scripts/build_query_splits.py guarantees every
query's gold table(s) are included in the written corpus.json (see its
build_corpus()'s `required_keys` -- this is checked via
table_dataset.has_table(), NOT whether the table is empty). But
src/data/synsql_dataset.py's SynSQLTableDataset.load_corpus() silently
DROPS any table with 0 rows or 0 columns at load time (see
_drop_empty_tables, added after build_query_splits.py's guarantee was
written). So a query whose only gold table happens to be empty can end
up with zero positives in the ACTUAL loaded corpus_tables at runtime,
even though corpus.json still lists it -- src/eval/retrieval_metrics.py
then silently EXCLUDES that query from the MAP/MRR average (a query with
zero positives in the corpus isn't scored as 0, it's dropped from the
mean entirely -- see compute_ranking_metrics' docstring), rather than
that query ever being counted as a ranking failure.

This script re-derives, for every split (train/val/test), which queries
this actually affects: for each query, checks whether ALL of its gold
tables are empty (query becomes fully unanswerable -- silently excluded
from MAP/MRR) or only SOME are (query keeps at least one valid positive,
degraded but not silently dropped). Also cross-checks that every gold
table referenced by a query is actually present in corpus.json at all
(a separate, independent thing from being empty).

Usage (same flags as every other script in this repo):
    python -m scripts.check_empty_gold_tables \\
        --databases_root /path/to/synsql/databases \\
        --questions_json /path/to/synsql/questions_with_tables.json \\
        --tables_json /path/to/synsql/tables.json \\
        --split_json configs/splits/query_split.json \\
        --corpus_json configs/splits/corpus.json
"""

import argparse
import json

from src.data.synsql_dataset import SynSQLQueryDataset, SynSQLTableDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", default=None)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--questions_json", required=True)
    parser.add_argument("--split_json", default="configs/splits/query_split.json")
    parser.add_argument("--corpus_json", default="configs/splits/corpus.json")
    args = parser.parse_args()

    print(f"indexing tables from {args.tables_json} / {args.databases_root} ...")
    table_dataset = SynSQLTableDataset(
        tables_json=args.tables_json,
        databases_root=args.databases_root,
    )

    query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    print(f"loaded {len(query_dataset)} query -> table example(s)")

    print(f"loading split from {args.split_json} ...")
    resolved = query_dataset.resolve_split(args.split_json)

    # Raw list from corpus.json BEFORE the empty-table filter -- used
    # only to distinguish "genuinely missing from corpus.json" (a table
    # build_query_splits.py never included at all) from "listed in
    # corpus.json but dropped as empty" (the case this script actually
    # cares about) below.
    with open(args.corpus_json, "r", encoding="utf-8") as f:
        corpus_raw = json.load(f)
    corpus_ids_raw = {
        f"{t['db_id']}#sep#{t['table_name']}" for t in corpus_raw["tables"]
    }
    print(f"corpus.json lists {len(corpus_ids_raw)} table(s) (pre-filter)")

    # The corpus is ALREADY materialized -- load_corpus() loads it from
    # its own JSON cache (instant, no SQLite reads) if one exists, or
    # materializes it once from SQLite (with progress logging) and
    # caches it for next time either way. It also already runs the same
    # empty-table filter this script cares about internally, printing
    # how many it dropped. No live per-query SQLite fetches needed at
    # all: every gold table is guaranteed (by build_query_splits.py's
    # build_corpus()) to be listed in corpus.json, so checking whether a
    # gold table is empty is just checking whether it survived this ONE
    # already-cached load -- pure in-memory set/dict lookups from here.
    print(f"loading corpus from {args.corpus_json} (materialized cache if present) ...")
    corpus_tables = table_dataset.load_corpus(args.corpus_json)
    corpus_by_id = {t.table_id: t for t in corpus_tables}
    print(f"corpus: {len(corpus_by_id)} table(s) survived the empty-table filter")

    def is_empty(db_id: str, table_name: str) -> bool:
        """A gold table is empty iff it's listed in corpus.json but did
        NOT survive load_corpus()'s internal _drop_empty_tables filter
        -- no SQLite call needed, corpus_by_id already reflects that
        filter's outcome for every table in corpus.json."""
        table_id = f"{db_id}#sep#{table_name}"
        return table_id in corpus_ids_raw and table_id not in corpus_by_id

    grand_total_fully_unanswerable = 0
    grand_total_degraded = 0
    all_empty_gold_tables: set[tuple[str, str]] = set()
    all_missing_from_corpus: set[tuple[str, str]] = set()

    for split_name in ("train", "val", "test"):
        indices = resolved[split_name]
        fully_unanswerable = []  # queries where EVERY gold table is empty
        degraded = []            # queries where SOME (not all) gold tables are empty
        missing_from_corpus = set()

        for idx in indices:
            ex = query_dataset.examples[idx]
            statuses = []
            for table_name in ex.table_names:
                empty = is_empty(ex.db_id, table_name)
                statuses.append(empty)
                if empty:
                    all_empty_gold_tables.add((ex.db_id, table_name))

                table_id = f"{ex.db_id}#sep#{table_name}"
                if table_id not in corpus_ids_raw:
                    missing_from_corpus.add((ex.db_id, table_name))
                    all_missing_from_corpus.add((ex.db_id, table_name))

            if all(statuses):
                fully_unanswerable.append(idx)
            elif any(statuses):
                degraded.append(idx)

        grand_total_fully_unanswerable += len(fully_unanswerable)
        grand_total_degraded += len(degraded)

        print(f"\n=== {split_name} ({len(indices)} quer(ies)) ===")
        print(
            f"  fully unanswerable (ALL gold tables empty -- silently "
            f"excluded from MAP/MRR): {len(fully_unanswerable)}"
        )
        print(
            f"  degraded (SOME gold tables empty, at least 1 valid "
            f"positive remains): {len(degraded)}"
        )
        print(
            f"  gold table(s) referenced but missing from corpus.json "
            f"entirely (separate issue): {len(missing_from_corpus)}"
        )
        if fully_unanswerable[:5]:
            sample = [query_dataset.examples[i].question[:80] for i in fully_unanswerable[:5]]
            print(f"  sample fully-unanswerable question(s): {sample}")

    print(f"\n=== summary across all splits ===")
    print(f"  total fully-unanswerable queries: {grand_total_fully_unanswerable}")
    print(f"  total degraded queries: {grand_total_degraded}")
    print(f"  distinct empty gold tables involved: {len(all_empty_gold_tables)}")
    if all_empty_gold_tables:
        print(f"    sample: {sorted(all_empty_gold_tables)[:10]}")
    print(f"  distinct gold tables missing from corpus.json entirely: {len(all_missing_from_corpus)}")
    if all_missing_from_corpus:
        print(f"    sample: {sorted(all_missing_from_corpus)[:10]}")

    if grand_total_fully_unanswerable == 0 and not all_missing_from_corpus:
        print("\nNo issue found -- every query's gold table(s) are present and non-empty.")


if __name__ == "__main__":
    main()
