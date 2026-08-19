"""Prepare deterministic, fully materialized real-valued 128-D batches.

All stochastic choices and all frozen-BERT work happen here.  The resulting
pickle stream contains final tensors and a final multi-positive mask; the
training process performs no string/table lookup, sampling, candidate
de-duplication, or label reconstruction.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random

import torch
from transformers import AutoModel, AutoTokenizer

from scripts.finetune_query_table import count_batches, resolve_train_batches
from src.data.prepared_batches import PreparedBatch, PreparedBatchWriter
from src.data.synsql_dataset import SynSQLQueryDataset, SynSQLTableDataset


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RandomProjectionPreparer:
    def __init__(
        self,
        model_name: str,
        projection_dim: int,
        projection_seed: int,
        max_length: int,
        max_text_batch_size: int,
        device: str,
        exclude_special_tokens: bool,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name).to(device).eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.device = torch.device(device)
        self.projection_dim = projection_dim
        self.max_length = max_length
        self.max_text_batch_size = max_text_batch_size
        self.exclude_special_tokens = exclude_special_tokens

        generator = torch.Generator(device="cpu").manual_seed(projection_seed)
        hidden_size = self.backbone.config.hidden_size
        self.projection_cpu = torch.randn(
            hidden_size, projection_dim, generator=generator, dtype=torch.float32
        ) / math.sqrt(projection_dim)
        self.projection = self.projection_cpu.to(device)
        # Preparation-only cache.  Training never sees or consults this
        # dictionary; every requested vector is embedded directly in its
        # final PreparedBatch record.
        self._cls_cache: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def _encode_missing_cls(self, strings: list[str]) -> None:
        missing = sorted({text for text in strings if text not in self._cls_cache})
        for start in range(0, len(missing), self.max_text_batch_size):
            texts = missing[start : start + self.max_text_batch_size]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.backbone(**encoded).last_hidden_state[:, 0]
            projected = (hidden @ self.projection).to(torch.float16).cpu()
            for i, text in enumerate(texts):
                self._cls_cache[text] = projected[i].clone()

    def cls_features(self, strings: list[str]) -> torch.Tensor:
        if not strings:
            return torch.empty(0, self.projection_dim, dtype=torch.float16)
        self._encode_missing_cls(strings)
        return torch.stack([self._cls_cache[text] for text in strings])

    @torch.inference_mode()
    def query_features(self, questions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.backbone(**encoded).last_hidden_state
        features = (hidden @ self.projection).to(torch.float16).cpu()
        mask = encoded["attention_mask"].bool()
        if self.exclude_special_tokens:
            special = torch.tensor(
                [
                    self.tokenizer.get_special_tokens_mask(
                        ids, already_has_special_tokens=True
                    )
                    for ids in encoded["input_ids"].tolist()
                ],
                device=self.device,
                dtype=torch.bool,
            )
            mask = mask & ~special
        return features, mask.cpu()

    def table_features(self, tables) -> tuple[torch.Tensor, ...]:
        bt = len(tables)
        max_n = max(table.num_columns for table in tables)
        max_m = max(table.num_rows for table in tables)
        r = self.projection_dim

        header_strings = [column.header for table in tables for column in table.columns]
        cell_strings = [
            column.cells[row]
            for table in tables
            for column in table.columns
            for row in range(table.num_rows)
        ]
        header_flat = self.cls_features(header_strings)
        cell_flat = self.cls_features(cell_strings)

        header_scatter = []
        cell_scatter = []
        col_mask = torch.zeros(bt, max_n, dtype=torch.bool)
        row_mask = torch.zeros(bt, max_m, dtype=torch.bool)
        cell_mask = torch.zeros(bt, max_n, max_m, dtype=torch.bool)

        hi = 0
        ci = 0
        for table_i, table in enumerate(tables):
            n, m = table.num_columns, table.num_rows
            col_mask[table_i, :n] = True
            row_mask[table_i, :m] = True
            for col_i, column in enumerate(table.columns):
                header_scatter.append(table_i * max_n + col_i)
                hi += 1
                for row_i in range(m):
                    cell_scatter.append((table_i * max_n + col_i) * max_m + row_i)
                    if column.cells[row_i].strip():
                        cell_mask[table_i, col_i, row_i] = True
                ci += m
        return (
            cell_flat,
            torch.tensor(cell_scatter, dtype=torch.int32),
            header_flat,
            torch.tensor(header_scatter, dtype=torch.int32),
            row_mask,
            col_mask,
            cell_mask,
        )

    def prepare_batch(self, resolved_batch) -> PreparedBatch:
        pairs, hard_negatives, gold_table_ids = resolved_batch
        questions = [question for question, _table in pairs]

        tables = []
        seen = set()
        for table in [table for _question, table in pairs] + list(hard_negatives):
            if table.table_id not in seen:
                seen.add(table.table_id)
                tables.append(table)
        candidate_ids = [table.table_id for table in tables]
        positive_mask = torch.tensor(
            [[table_id in gold for table_id in candidate_ids] for gold in gold_table_ids],
            dtype=torch.bool,
        )
        if not torch.all(positive_mask.any(dim=1)):
            raise ValueError("preparation produced a query with no positive candidate")

        query_features, query_mask = self.query_features(questions)
        (
            cells,
            cell_scatter,
            headers,
            header_scatter,
            row_mask,
            col_mask,
            cell_mask,
        ) = self.table_features(tables)
        return PreparedBatch(
            query_features=query_features,
            query_mask=query_mask,
            cell_features=cells,
            cell_scatter=cell_scatter,
            header_features=headers,
            header_scatter=header_scatter,
            row_mask=row_mask,
            col_mask=col_mask,
            cell_mask=cell_mask,
            positive_mask=positive_mask,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--questions_json", required=True)
    parser.add_argument("--tables_json", default=None)
    parser.add_argument("--split_json", default="configs/splits/query_split.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection_seed", type=int, default=None)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--batches_per_shard",
        type=int,
        default=20,
        help="atomically publish this many prepared batches per pickle so "
             "training can consume completed shards while preparation continues",
    )
    parser.add_argument("--n_hard_negatives", type=int, default=2)
    parser.add_argument("--train_sample_size", type=int, default=40000)
    parser.add_argument("--max_rows", type=int, default=50)
    parser.add_argument("--max_columns", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--max_text_batch_size", type=int, default=2048)
    parser.add_argument("--exclude_special_tokens", action="store_true")
    args = parser.parse_args()

    if args.projection_dim <= 0 or args.batch_size <= 1 or args.batches_per_shard <= 0:
        parser.error(
            "projection_dim/batches_per_shard must be positive and batch_size must exceed one"
        )
    projection_seed = args.seed if args.projection_seed is None else args.projection_seed

    table_dataset = SynSQLTableDataset(
        databases_root=args.databases_root,
        tables_json=args.tables_json,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    train_indices = query_dataset.resolve_split(args.split_json)["train"]
    shuffled = list(train_indices)
    random.Random(args.seed).shuffle(shuffled)
    chunk_size = args.train_sample_size or len(shuffled)
    train_chunks = [shuffled[i : i + chunk_size] for i in range(0, len(shuffled), chunk_size)]

    preparer = RandomProjectionPreparer(
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        projection_seed=projection_seed,
        max_length=args.max_length,
        max_text_batch_size=args.max_text_batch_size,
        device=args.device,
        exclude_special_tokens=args.exclude_special_tokens,
    )
    rng = random.Random(args.seed)
    common_metadata = {
        "seed": args.seed,
        "projection_seed": projection_seed,
        "projection_dim": args.projection_dim,
        "projection_matrix": preparer.projection_cpu,
        "batch_size": args.batch_size,
        "batches_per_shard": args.batches_per_shard,
        "n_hard_negatives": args.n_hard_negatives,
        "max_rows": args.max_rows,
        "max_columns": args.max_columns,
        "max_length": args.max_length,
        "model_name": args.model_name,
        "split_sha256": file_sha256(args.split_json),
        "questions_sha256": file_sha256(args.questions_json),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    existing_outputs = glob.glob(os.path.join(args.output_dir, "epoch_*"))
    if existing_outputs or os.path.exists(os.path.join(args.output_dir, "PREPARATION_COMPLETE")):
        parser.error(
            f"output_dir {args.output_dir!r} already contains prepared epoch files; "
            "use an empty directory so a streaming trainer cannot consume stale shards"
        )
    for epoch, indices in enumerate(train_chunks):
        expected = count_batches(len(indices), args.batch_size)
        resolved = resolve_train_batches(
            query_dataset,
            table_dataset,
            indices,
            args.batch_size,
            args.max_columns,
            rng,
            n_hard_negatives=args.n_hard_negatives,
        )
        writer = None
        partial_path = final_path = None
        total_written = 0
        shard = -1
        try:
            for batch_index, batch in enumerate(resolved):
                if batch_index % args.batches_per_shard == 0:
                    shard += 1
                    final_path = os.path.join(
                        args.output_dir, f"epoch_{epoch:03d}_shard_{shard:05d}.pkl"
                    )
                    partial_path = final_path + ".partial"
                    metadata = dict(
                        common_metadata,
                        epoch=epoch,
                        shard=shard,
                        n_queries=len(indices),
                        expected_batches=expected,
                    )
                    writer = PreparedBatchWriter(partial_path, metadata)

                prepared = preparer.prepare_batch(batch)
                if batch_index == 0:
                    utilization = prepared.cell_features.shape[0] / prepared.cell_mask.numel()
                    print(
                        f"[prepare] epoch {epoch}: packed {prepared.cell_features.shape[0]} "
                        f"real table cells instead of {prepared.cell_mask.numel()} padded "
                        f"slots ({100 * utilization:.1f}% feature utilization)",
                        flush=True,
                    )
                writer.write(prepared)
                total_written += 1

                shard_full = writer.count >= args.batches_per_shard
                final_batch = batch_index + 1 >= expected
                if shard_full or final_batch:
                    writer.close()
                    os.replace(partial_path, final_path)
                    print(
                        f"[prepare] published {final_path} "
                        f"({writer.count} batches; {total_written}/{expected} this epoch)",
                        flush=True,
                    )
                    writer = None

                if (batch_index + 1) % 20 == 0:
                    print(
                        f"[prepare] epoch {epoch}: {batch_index + 1}/{expected} batches, "
                        f"{len(preparer._cls_cache)} unique cell/header strings",
                        flush=True,
                    )
        except BaseException:
            if writer is not None:
                writer.close()
            raise
        if writer is not None:
            # The arithmetic expected count can exceed the actual yielded
            # count when invalid/empty examples are dropped. Publish the
            # final short shard after normal generator exhaustion.
            writer.close()
            os.replace(partial_path, final_path)
            print(
                f"[prepare] published {final_path} "
                f"({writer.count} batches; {total_written} this epoch)",
                flush=True,
            )

        with open(
            os.path.join(args.output_dir, f"epoch_{epoch:03d}.complete"), "w", encoding="utf-8"
        ) as f:
            json.dump({"epoch": epoch, "batches": total_written, "shards": shard + 1}, f)
        print(f"[prepare] epoch {epoch} complete: {total_written} batch(es)", flush=True)

    manifest = {
        key: value for key, value in common_metadata.items() if key != "projection_matrix"
    }
    manifest["epochs"] = len(train_chunks)
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.output_dir, "PREPARATION_COMPLETE"), "w", encoding="utf-8") as f:
        f.write("complete\n")


if __name__ == "__main__":
    main()
