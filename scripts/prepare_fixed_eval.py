"""Prepare deterministic lookup-free validation or test artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random

from scripts.finetune_query_table import cap_columns
from scripts.prepare_fixed_batches import RandomProjectionPreparer, file_sha256
from src.data.prepared_eval import (
    PreparedEvalQueries,
    PreparedEvalTables,
    write_eval_shard,
)
from src.data.synsql_dataset import SynSQLQueryDataset, SynSQLTableDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--questions_json", required=True)
    parser.add_argument("--tables_json", default=None)
    parser.add_argument("--split_json", default="configs/splits/query_split.json")
    parser.add_argument("--corpus_json", default="configs/splits/corpus.json")
    parser.add_argument("--materialized_corpus_cache_path", required=True)
    parser.add_argument("--text_cache_path", required=True)
    parser.add_argument("--query_cache_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection_seed", type=int, default=42)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--query_sample_size", type=int, default=None)
    parser.add_argument("--corpus_sample_size", type=int, default=None)
    parser.add_argument("--n_hard_negatives_per_db", type=int, default=2)
    parser.add_argument("--query_shard_size", type=int, default=512)
    parser.add_argument("--table_shard_size", type=int, default=32)
    parser.add_argument("--max_rows", type=int, default=50)
    parser.add_argument("--max_columns", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--max_text_batch_size", type=int, default=8192)
    parser.add_argument("--exclude_special_tokens", action="store_true")
    args = parser.parse_args()

    for name in ("query_shard_size", "table_shard_size", "projection_dim"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.query_sample_size is not None and args.query_sample_size <= 0:
        parser.error("--query_sample_size must be positive or omitted")
    if args.corpus_sample_size is not None and args.corpus_sample_size <= 0:
        parser.error("--corpus_sample_size must be positive or omitted")
    if not os.path.exists(args.materialized_corpus_cache_path):
        parser.error("materialized corpus cache does not exist")
    os.makedirs(args.output_dir, exist_ok=True)
    if glob.glob(os.path.join(args.output_dir, "*.pkl")) or os.path.exists(
        os.path.join(args.output_dir, "PREPARATION_COMPLETE")
    ):
        parser.error("output_dir already contains prepared evaluation files")

    table_dataset = SynSQLTableDataset(
        databases_root=args.databases_root,
        tables_json=args.tables_json,
        max_rows=args.max_rows,
        seed=args.seed,
        max_open_connections=64,
    )
    corpus = table_dataset.load_corpus(
        args.corpus_json,
        materialized_cache_path=args.materialized_corpus_cache_path,
    )
    corpus = [cap_columns(table, args.max_columns) for table in corpus]
    corpus_by_id = {table.table_id: table for table in corpus}

    query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    resolved_splits = query_dataset.resolve_split(args.split_json)
    split_indices = resolved_splits[args.split]
    with open(args.split_json, encoding="utf-8") as source:
        persisted_split = json.load(source)
    print(
        f"[prepare-eval] persisted query split: "
        f"train={len(resolved_splits['train'])}, "
        f"val={len(resolved_splits['val'])}, "
        f"test={len(resolved_splits['test'])}; "
        f"fractions={persisted_split.get('train_frac')}/"
        f"{persisted_split.get('val_frac')}/"
        f"{persisted_split.get('test_frac')}",
        flush=True,
    )
    rng = random.Random(args.seed)
    if args.query_sample_size is not None and args.query_sample_size < len(split_indices):
        query_indices = sorted(rng.sample(split_indices, args.query_sample_size))
    else:
        query_indices = list(split_indices)

    examples = []
    required_ids: set[str] = set()
    for index in query_indices:
        example = query_dataset.examples[index]
        gold = tuple(
            f"{example.db_id}#sep#{name}"
            for name in example.table_names
            if f"{example.db_id}#sep#{name}" in corpus_by_id
        )
        if gold:
            examples.append((example.question, example.db_id, gold))
            required_ids.update(gold)

    if args.corpus_sample_size is None:
        selected_ids = set(corpus_by_id)
    else:
        # Match train_model.py's validation policy exactly: the requested
        # corpus size is a target, never a hard cap. Every positive and the
        # configured same-database hard negatives are retained even when
        # they already exceed the target; random distractors only fill the
        # remaining capacity. Silently trimming hard negatives here would
        # change the validation task relative to the original pipeline.
        selected_ids = set(required_ids)
        tables_by_db: dict[str, list[str]] = {}
        for table_id in corpus_by_id:
            db_id, _, _table_name = table_id.partition("#sep#")
            tables_by_db.setdefault(db_id, []).append(table_id)
        for db_id in sorted({db_id for _question, db_id, _gold in examples}):
            alternatives = [
                table_id
                for table_id in tables_by_db.get(db_id, [])
                if table_id not in required_ids
            ]
            rng.shuffle(alternatives)
            selected_ids.update(alternatives[: args.n_hard_negatives_per_db])
        remaining = [table_id for table_id in corpus_by_id if table_id not in selected_ids]
        rng.shuffle(remaining)
        selected_ids.update(remaining[: max(0, args.corpus_sample_size - len(selected_ids))])

    selected_tables = [table for table in corpus if table.table_id in selected_ids]
    selected_table_ids = {table.table_id for table in selected_tables}
    examples = [
        (question, db_id, gold)
        for question, db_id, gold in examples
        if any(table_id in selected_table_ids for table_id in gold)
    ]

    preparer = RandomProjectionPreparer(
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
        max_length=args.max_length,
        max_text_batch_size=args.max_text_batch_size,
        device=args.device,
        exclude_special_tokens=args.exclude_special_tokens,
        text_cache_path=args.text_cache_path,
        query_cache_path=args.query_cache_path,
    )
    metadata = {
        "split": args.split,
        "seed": args.seed,
        "projection_seed": args.projection_seed,
        "projection_dim": args.projection_dim,
        "model_name": args.model_name,
        "max_rows": args.max_rows,
        "max_columns": args.max_columns,
        "max_length": args.max_length,
        "split_sha256": file_sha256(args.split_json),
        "questions_sha256": file_sha256(args.questions_json),
        "train_frac": persisted_split.get("train_frac"),
        "val_frac": persisted_split.get("val_frac"),
        "test_frac": persisted_split.get("test_frac"),
        "n_queries": len(examples),
        "n_tables": len(selected_tables),
    }

    for shard, start in enumerate(range(0, len(examples), args.query_shard_size)):
        chunk = examples[start : start + args.query_shard_size]
        questions = [item[0] for item in chunk]
        features, mask = preparer.query_features(questions)
        payload = PreparedEvalQueries(
            query_texts=tuple(questions),
            gold_table_ids=tuple(item[2] for item in chunk),
            features=features,
            mask=mask,
        )
        path = os.path.join(args.output_dir, f"queries_{shard:05d}.pkl")
        write_eval_shard(path, metadata, payload)
        print(f"[prepare-eval] wrote {path} ({start + len(chunk)}/{len(examples)} queries)", flush=True)

    for shard, start in enumerate(range(0, len(selected_tables), args.table_shard_size)):
        tables = selected_tables[start : start + args.table_shard_size]
        (
            cells,
            cell_scatter,
            headers,
            header_scatter,
            row_mask,
            col_mask,
            cell_mask,
        ) = preparer.table_features(tables)
        payload = PreparedEvalTables(
            table_ids=tuple(table.table_id for table in tables),
            cell_features=cells,
            cell_scatter=cell_scatter,
            header_features=headers,
            header_scatter=header_scatter,
            row_mask=row_mask,
            col_mask=col_mask,
            cell_mask=cell_mask,
        )
        path = os.path.join(args.output_dir, f"tables_{shard:05d}.pkl")
        write_eval_shard(path, metadata, payload)
        print(f"[prepare-eval] wrote {path} ({start + len(tables)}/{len(selected_tables)} tables)", flush=True)

    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
    with open(os.path.join(args.output_dir, "PREPARATION_COMPLETE"), "w", encoding="utf-8") as output:
        output.write("complete\n")
    table_dataset.close_connections()
    print(
        f"[prepare-eval] complete: split={args.split}, queries={len(examples)}, "
        f"tables={len(selected_tables)}, query_shards={math.ceil(len(examples)/args.query_shard_size)}, "
        f"table_shards={math.ceil(len(selected_tables)/args.table_shard_size)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
