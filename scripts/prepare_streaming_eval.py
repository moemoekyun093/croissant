"""Prepare lookup-free full-test streaming retrieval artifacts.

Unlike ``prepare_fixed_eval.py`` (one global query/corpus sample), this
preserves the original BIG evaluator's protocol:

* every valid query in the requested split is retained;
* queries are stored in deterministic streaming chunks;
* each query sees every table in its own database plus a deterministic
  number of random distractors;
* table features are stored once in one global indexed corpus.

Candidate indices and visibility/positive masks are fixed on disk, so every
model receives exactly the same ranking task without runtime data lookup.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random

import torch

from scripts.finetune_query_table import cap_columns
from scripts.prepare_fixed_batches import RandomProjectionPreparer, file_sha256
from src.data.prepared_eval import (
    PreparedEvalTables,
    PreparedStreamingEvalQueries,
    read_eval_shard,
    write_eval_shard,
)
from src.data.synsql_dataset import SynSQLQueryDataset, SynSQLTableDataset


def _db_of(table_id: str) -> str:
    return table_id.partition("#sep#")[0]


def _build_candidate_plan(
    chunk: list[tuple[str, str, tuple[str, ...]]],
    db_to_table_indices: dict[str, list[int]],
    table_ids: list[str],
    table_index: dict[str, int],
    n_distractors: int,
    seed: int,
    query_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the old evaluator's pool/visibility policy deterministically."""
    pool_indices: list[int] = []
    seen: set[int] = set()
    chunk_databases = sorted({db_id for _question, db_id, _gold in chunk})
    for db_id in chunk_databases:
        for index in db_to_table_indices.get(db_id, ()):  # every own-DB table
            if index not in seen:
                seen.add(index)
                pool_indices.append(index)

    if n_distractors > 0:
        rng = random.Random(seed * 7919 + query_offset)
        reservoir = rng.sample(
            range(len(table_ids)), min(len(table_ids), 2 * n_distractors)
        )
        for index in reservoir:
            if index not in seen:
                seen.add(index)
                pool_indices.append(index)

    if not pool_indices:
        raise ValueError(f"query chunk at offset {query_offset} has no candidates")

    database_names = sorted(
        set(chunk_databases) | {_db_of(table_ids[index]) for index in pool_indices}
    )
    database_codes = {name: code for code, name in enumerate(database_names)}
    pool_database_codes = torch.tensor(
        [database_codes[_db_of(table_ids[index])] for index in pool_indices],
        dtype=torch.int32,
    )
    pool_position = {global_index: position for position, global_index in enumerate(pool_indices)}
    visible = torch.zeros(len(chunk), len(pool_indices), dtype=torch.bool)
    positive = torch.zeros_like(visible)

    for query_position, (_question, db_id, gold_ids) in enumerate(chunk):
        own = pool_database_codes == database_codes[db_id]
        visible[query_position] |= own
        if n_distractors > 0:
            other_positions = (~own).nonzero(as_tuple=True)[0]
            if other_positions.numel() > n_distractors:
                generator = torch.Generator().manual_seed(
                    seed * 1_000_003 + query_offset + query_position
                )
                permutation = torch.randperm(
                    other_positions.numel(), generator=generator
                )[:n_distractors]
                other_positions = other_positions[permutation]
            visible[query_position, other_positions] = True

        for table_id in gold_ids:
            global_index = table_index[table_id]
            candidate_position = pool_position.get(global_index)
            if candidate_position is None:
                raise RuntimeError(
                    f"positive {table_id!r} missing from its own-database pool"
                )
            positive[query_position, candidate_position] = True

    if not torch.all(positive.any(dim=1)):
        raise RuntimeError("streaming candidate construction omitted a positive")
    if torch.any(positive & ~visible):
        raise RuntimeError("streaming candidate construction hid a positive")
    return torch.tensor(pool_indices, dtype=torch.int32), visible, positive


def _validate_resumed_metadata(expected: dict, actual: dict, path: str) -> None:
    keys = (
        "evaluation_mode", "split", "seed", "projection_seed", "projection_dim",
        "model_name", "max_rows", "max_columns", "max_length", "split_sha256",
        "questions_sha256", "query_chunk_size", "n_distractors",
        "table_shard_size", "n_queries", "n_tables",
    )
    mismatches = {
        key: (actual.get(key), expected.get(key))
        for key in keys
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        raise ValueError(f"incompatible resumed streaming shard {path}: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--questions_json", required=True)
    parser.add_argument("--tables_json", default=None)
    parser.add_argument("--split_json", default="configs/splits/query_split.json")
    parser.add_argument("--corpus_json", default="configs/splits/corpus.json")
    parser.add_argument("--materialized_corpus_cache_path", required=True)
    parser.add_argument("--text_cache_path", default=None)
    parser.add_argument("--query_cache_path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection_seed", type=int, default=42)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--query_chunk_size", type=int, default=2000)
    parser.add_argument("--n_distractors", type=int, default=2000)
    parser.add_argument("--table_shard_size", type=int, default=256)
    parser.add_argument("--max_rows", type=int, default=50)
    parser.add_argument("--max_columns", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--max_text_batch_size", type=int, default=8192)
    parser.add_argument("--exclude_special_tokens", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate and reuse already published query/table shards",
    )
    args = parser.parse_args()

    for name in (
        "projection_dim", "query_chunk_size", "table_shard_size",
        "max_rows", "max_columns", "max_length", "max_text_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.n_distractors < 0:
        parser.error("--n_distractors must be non-negative")
    if not os.path.exists(args.materialized_corpus_cache_path):
        parser.error("materialized corpus cache does not exist")

    os.makedirs(args.output_dir, exist_ok=True)
    existing_pickles = glob.glob(os.path.join(args.output_dir, "*.pkl"))
    complete_marker = os.path.join(args.output_dir, "PREPARATION_COMPLETE")
    if (existing_pickles or os.path.exists(complete_marker)) and not args.resume:
        parser.error("output_dir already contains prepared streaming files")
    if os.path.exists(complete_marker) and args.resume:
        print(
            f"[prepare-streaming] already complete: {args.output_dir}", flush=True
        )
        return

    table_dataset = SynSQLTableDataset(
        databases_root=args.databases_root,
        tables_json=args.tables_json,
        max_rows=args.max_rows,
        seed=args.seed,
        max_open_connections=1,
    )
    corpus = table_dataset.load_corpus(
        args.corpus_json,
        materialized_cache_path=args.materialized_corpus_cache_path,
    )
    corpus = [cap_columns(table, args.max_columns) for table in corpus]
    table_ids = [table.table_id for table in corpus]
    table_index = {table_id: index for index, table_id in enumerate(table_ids)}
    if len(table_index) != len(table_ids):
        raise ValueError("prepared streaming corpus contains duplicate table IDs")
    db_to_table_indices: dict[str, list[int]] = {}
    for index, table_id in enumerate(table_ids):
        db_to_table_indices.setdefault(_db_of(table_id), []).append(index)

    query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    resolved_splits = query_dataset.resolve_split(args.split_json)
    with open(args.split_json, encoding="utf-8") as source:
        persisted_split = json.load(source)

    examples: list[tuple[str, str, tuple[str, ...]]] = []
    skipped_without_positive = 0
    for index in resolved_splits[args.split]:
        example = query_dataset.examples[index]
        gold = tuple(
            f"{example.db_id}#sep#{name}"
            for name in example.table_names
            if f"{example.db_id}#sep#{name}" in table_index
        )
        if not gold:
            skipped_without_positive += 1
            continue
        examples.append((example.question, example.db_id, gold))
    # Match the old evaluator: group queries by database before chunking.
    examples.sort(key=lambda item: item[1])

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
        "evaluation_mode": "streaming_per_query_candidates",
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
        "query_chunk_size": args.query_chunk_size,
        "n_distractors": args.n_distractors,
        "table_shard_size": args.table_shard_size,
        "n_queries": len(examples),
        "n_tables": len(corpus),
        "skipped_queries_without_positive": skipped_without_positive,
    }

    n_query_chunks = math.ceil(len(examples) / args.query_chunk_size)
    for chunk_number, start in enumerate(
        range(0, len(examples), args.query_chunk_size)
    ):
        chunk = examples[start : start + args.query_chunk_size]
        candidate_indices, visible, positive = _build_candidate_plan(
            chunk,
            db_to_table_indices,
            table_ids,
            table_index,
            args.n_distractors,
            args.seed,
            start,
        )
        path = os.path.join(args.output_dir, f"query_chunks_{chunk_number:05d}.pkl")
        if args.resume and os.path.exists(path):
            resumed_metadata, resumed_payload = read_eval_shard(path)
            _validate_resumed_metadata(metadata, resumed_metadata, path)
            if not isinstance(resumed_payload, PreparedStreamingEvalQueries):
                raise TypeError(f"resumed query shard has wrong payload: {path}")
            if resumed_payload.features.shape[0] != len(chunk):
                raise ValueError(f"resumed query shard has wrong query count: {path}")
            if not torch.equal(
                resumed_payload.candidate_table_indices, candidate_indices
            ) or not torch.equal(resumed_payload.visible_mask, visible) or not torch.equal(
                resumed_payload.positive_mask, positive
            ):
                raise ValueError(f"resumed query candidate plan differs: {path}")
            print(
                f"[prepare-streaming] resumed {path} "
                f"({chunk_number + 1}/{n_query_chunks} chunks)",
                flush=True,
            )
            continue
        features, mask = preparer.query_features([item[0] for item in chunk])
        payload = PreparedStreamingEvalQueries(
            query_texts=tuple(item[0] for item in chunk),
            gold_table_ids=tuple(item[2] for item in chunk),
            features=features,
            mask=mask,
            candidate_table_indices=candidate_indices,
            visible_mask=visible,
            positive_mask=positive,
        )
        write_eval_shard(path, metadata, payload)
        print(
            f"[prepare-streaming] wrote {path}: queries {start + len(chunk)}/"
            f"{len(examples)}, pool={candidate_indices.numel()}, "
            f"visible/query<={args.n_distractors}+own-db "
            f"({chunk_number + 1}/{n_query_chunks} chunks)",
            flush=True,
        )

    n_table_shards = math.ceil(len(corpus) / args.table_shard_size)
    for shard, start in enumerate(range(0, len(corpus), args.table_shard_size)):
        tables = corpus[start : start + args.table_shard_size]
        path = os.path.join(args.output_dir, f"tables_{shard:05d}.pkl")
        if args.resume and os.path.exists(path):
            resumed_metadata, resumed_payload = read_eval_shard(path)
            _validate_resumed_metadata(metadata, resumed_metadata, path)
            if not isinstance(resumed_payload, PreparedEvalTables):
                raise TypeError(f"resumed table shard has wrong payload: {path}")
            expected_ids = tuple(table.table_id for table in tables)
            if resumed_payload.table_ids != expected_ids:
                raise ValueError(f"resumed table shard has wrong table IDs: {path}")
            print(
                f"[prepare-streaming] resumed {path} "
                f"({shard + 1}/{n_table_shards} shards)",
                flush=True,
            )
            continue
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
        write_eval_shard(path, metadata, payload)
        print(
            f"[prepare-streaming] wrote {path}: tables {start + len(tables)}/"
            f"{len(corpus)} ({shard + 1}/{n_table_shards} shards)",
            flush=True,
        )

    metadata.update(
        {"n_query_chunks": n_query_chunks, "n_table_shards": n_table_shards}
    )
    manifest_partial = os.path.join(args.output_dir, "manifest.json.partial")
    with open(manifest_partial, "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
    os.replace(manifest_partial, os.path.join(args.output_dir, "manifest.json"))
    with open(
        os.path.join(args.output_dir, "PREPARATION_COMPLETE"),
        "w",
        encoding="utf-8",
    ) as output:
        output.write("complete\n")
    table_dataset.close_connections()
    print(
        f"[prepare-streaming] complete: split={args.split}, "
        f"queries={len(examples)}, query_chunks={n_query_chunks}, "
        f"tables={len(corpus)}, table_shards={n_table_shards}, "
        f"distractors/query={args.n_distractors}",
        flush=True,
    )


if __name__ == "__main__":
    main()
