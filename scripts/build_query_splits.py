"""
Builds two persisted artifacts every training script (ours and every
baseline) should load instead of re-deriving their own: a train/val/test
split of the QUERIES in questions_with_tables.json, and the FIXED table
corpus used as the retrieval candidate pool for every split.

Per instruction: the table corpus is never split -- every split
(train/val/test) ranks against the exact same, full set of candidate
tables. Only which QUERIES belong to train vs. val vs. test is decided
here. Run this ONCE per (questions_json, seed, fracs, corpus_size)
combination; every training script should then point at the same two
output files so results stay comparable across models and across runs.

Usage:
    python -m scripts.build_query_splits \
        --tables_json /path/to/synsql/tables.json \
        --databases_root /path/to/synsql/databases \
        --questions_json /path/to/synsql/questions_with_tables.json \
        --split_output configs/splits/query_split.json \
        --corpus_output configs/splits/corpus.json

    # pilot scale: cap the corpus instead of using the entire dataset.
    # ALWAYS includes every positive table referenced by any query in
    # any split (train/val/test) first, then fills up to --corpus_size
    # with random extra "distractor" tables -- so no query ever ends up
    # with its correct answer missing from the corpus just because the
    # corpus was capped for a pilot.
    python -m scripts.build_query_splits \
        --tables_json /path/to/synsql/tables.json \
        --databases_root /path/to/synsql/databases \
        --questions_json /path/to/synsql/questions_with_tables.json \
        --n_examples 500 --corpus_size 2000
"""

import argparse
import json
import os
import random

from src.data.synsql_dataset import SynSQLQueryDataset, SynSQLTableDataset


def build_query_split(
    query_dataset: SynSQLQueryDataset,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    n_examples: int | None,
) -> dict:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/val/test fracs must sum to 1.0, got {total}")

    n = len(query_dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    if n_examples is not None and n_examples < len(indices):
        indices = indices[:n_examples]
        print(f"pilot run: capping to {len(indices)} query example(s)")

    n_total = len(indices)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))
    # test gets the remainder, so rounding never drops/duplicates an example
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def to_records(idx_list):
        records = []
        for i in idx_list:
            ex = query_dataset.examples[i]
            records.append(
                {"question": ex.question, "db_id": ex.db_id, "table_names": ex.table_names}
            )
        return records

    return {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "n_total": n_total,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train": to_records(train_idx),
        "val": to_records(val_idx),
        "test": to_records(test_idx),
    }


def build_corpus(
    table_dataset: SynSQLTableDataset,
    split: dict,
    corpus_size: int | None,
    seed: int,
) -> list[dict]:
    """
    Returns a list of {"db_id":..., "table_name":...} records -- the
    FIXED candidate pool every split ranks against.

    ALWAYS includes every positive table referenced by any query across
    every split (train/val/test) first -- this is non-negotiable,
    otherwise a query's own correct answer could be missing from the
    corpus it's evaluated against, silently making that query
    unanswerable rather than just hard. If corpus_size is given and
    larger than that required set, fills up to corpus_size with
    randomly sampled additional "distractor" tables from the rest of
    the dataset. If corpus_size is None, uses every table in the
    dataset (the true full corpus -- expensive at SynSQL-2.5M scale,
    intended for a real full run, not a pilot).
    """
    required_keys = set()
    for split_name in ("train", "val", "test"):
        for record in split[split_name]:
            db_id = record["db_id"]
            for table_name in record["table_names"]:
                if table_dataset.has_table(db_id, table_name):
                    required_keys.add((db_id, table_name))

    if corpus_size is None:
        all_keys = table_dataset.table_keys()
        print(f"corpus: using the full dataset ({len(all_keys)} tables, no cap)")
        corpus_keys = set(all_keys)
    else:
        if corpus_size < len(required_keys):
            raise ValueError(
                f"--corpus_size {corpus_size} is smaller than the "
                f"{len(required_keys)} table(s) required as true positives "
                f"for the queries in this split -- raise --corpus_size to "
                f"at least {len(required_keys)}."
            )
        rng = random.Random(seed)
        all_keys = table_dataset.table_keys()
        rng.shuffle(all_keys)

        corpus_keys = set(required_keys)
        for key in all_keys:
            if len(corpus_keys) >= corpus_size:
                break
            corpus_keys.add(key)
        print(
            f"corpus: {len(required_keys)} required (query positives) + "
            f"{len(corpus_keys) - len(required_keys)} distractor(s) = "
            f"{len(corpus_keys)} total"
        )

    return [{"db_id": db_id, "table_name": table_name} for db_id, table_name in sorted(corpus_keys)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", default=None)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--questions_json", required=True)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_examples", type=int, default=None,
        help="pilot run: cap the total number of query examples before splitting "
             "(applied before train/val/test fracs, on the shuffled full set)",
    )
    parser.add_argument(
        "--corpus_size", type=int, default=None,
        help="pilot run: cap the retrieval corpus size. Always includes every "
             "query's true positive table(s) first, fills the rest with random "
             "distractors. Omit for the full, uncapped dataset as the corpus.",
    )
    parser.add_argument(
        "--exclude_styles", default=None,
        help="comma-separated 'style' values to drop (e.g. 'Multi-turn Dialogue,Vague'). "
             "Default: SynSQLQueryDataset's own default (currently just "
             "'Multi-turn Dialogue' -- those collapse a whole conversation into "
             "one question string, not a standalone query). Pass an empty string "
             "('') to disable exclusion entirely and keep every style.",
    )
    parser.add_argument("--split_output", default="configs/splits/query_split.json")
    parser.add_argument("--corpus_output", default="configs/splits/corpus.json")
    args = parser.parse_args()

    table_dataset = SynSQLTableDataset(
        tables_json=args.tables_json,
        databases_root=args.databases_root,
    )
    if args.exclude_styles is None:
        query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    else:
        styles = {s.strip() for s in args.exclude_styles.split(",") if s.strip()}
        query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset, exclude_styles=styles or None)
    print(f"loaded {len(query_dataset)} valid query -> table example(s)")

    split = build_query_split(
        query_dataset,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        n_examples=args.n_examples,
    )

    corpus = build_corpus(table_dataset, split, corpus_size=args.corpus_size, seed=args.seed)

    os.makedirs(os.path.dirname(args.split_output) or ".", exist_ok=True)
    with open(args.split_output, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(
        f"wrote split to {args.split_output}: "
        f"{split['n_train']} train / {split['n_val']} val / {split['n_test']} test "
        f"(seed={args.seed})"
    )

    os.makedirs(os.path.dirname(args.corpus_output) or ".", exist_ok=True)
    with open(args.corpus_output, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "corpus_size": args.corpus_size, "tables": corpus}, f, indent=2)
    print(f"wrote corpus to {args.corpus_output}: {len(corpus)} table(s)")
