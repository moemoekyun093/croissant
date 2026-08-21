"""Lookup-free prepared validation/test query and corpus shards."""

from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
from typing import Iterator

import torch


FORMAT_NAME = "croissant_prepared_eval"
FORMAT_VERSION = 1


@dataclass
class PreparedEvalQueries:
    query_texts: tuple[str, ...]
    gold_table_ids: tuple[tuple[str, ...], ...]
    features: torch.Tensor  # [B,L,R] float16 CPU
    mask: torch.Tensor  # [B,L] bool

    def validate(self, projection_dim: int) -> None:
        if self.features.ndim != 3 or self.features.shape[-1] != projection_dim:
            raise ValueError("bad prepared evaluation query features")
        if self.mask.shape != self.features.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("bad prepared evaluation query mask")
        if len(self.query_texts) != self.features.shape[0]:
            raise ValueError("query text count does not match features")
        if len(self.gold_table_ids) != self.features.shape[0]:
            raise ValueError("query gold-label count does not match features")
        if any(not labels for labels in self.gold_table_ids):
            raise ValueError("evaluation query has no gold table")


@dataclass
class PreparedStreamingEvalQueries:
    """One fixed full-test query chunk and its candidate decisions.

    ``candidate_table_indices`` points into the single globally prepared
    table corpus (the concatenation of ``tables_*.pkl`` in filename order).
    The two masks therefore freeze the old streaming evaluator's per-query
    candidate visibility and relevance decisions without duplicating table
    features in every query chunk.
    """

    query_texts: tuple[str, ...]          # audit only
    gold_table_ids: tuple[tuple[str, ...], ...]  # audit only
    features: torch.Tensor               # [B,L,R] float16 CPU
    mask: torch.Tensor                   # [B,L] bool
    candidate_table_indices: torch.Tensor  # [P] int32, global corpus positions
    visible_mask: torch.Tensor           # [B,P] bool
    positive_mask: torch.Tensor          # [B,P] bool

    def validate(self, projection_dim: int) -> None:
        if self.features.ndim != 3 or self.features.shape[-1] != projection_dim:
            raise ValueError("bad prepared streaming query features")
        if self.mask.shape != self.features.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("bad prepared streaming query mask")
        batch_size = self.features.shape[0]
        pool_size = self.candidate_table_indices.numel()
        if len(self.query_texts) != batch_size:
            raise ValueError("streaming query text count does not match features")
        if len(self.gold_table_ids) != batch_size:
            raise ValueError("streaming query gold-label count does not match features")
        if self.candidate_table_indices.ndim != 1:
            raise ValueError("streaming candidate indices must be one-dimensional")
        if self.candidate_table_indices.dtype != torch.int32:
            raise TypeError("streaming candidate indices must be int32")
        if self.visible_mask.shape != (batch_size, pool_size):
            raise ValueError("bad streaming candidate visibility shape")
        if self.positive_mask.shape != (batch_size, pool_size):
            raise ValueError("bad streaming positive-mask shape")
        if self.visible_mask.dtype != torch.bool or self.positive_mask.dtype != torch.bool:
            raise TypeError("streaming visibility/positive masks must be bool")
        if pool_size == 0:
            raise ValueError("streaming query chunk has an empty candidate pool")
        if int(self.candidate_table_indices.min()) < 0:
            raise ValueError("streaming candidate index is negative")
        if not torch.all(self.positive_mask.any(dim=1)):
            raise ValueError("streaming evaluation query has no positive candidate")
        if torch.any(self.positive_mask & ~self.visible_mask):
            raise ValueError("streaming positive candidate is hidden")


@dataclass
class PreparedEvalTables:
    table_ids: tuple[str, ...]
    cell_features: torch.Tensor
    cell_scatter: torch.Tensor
    header_features: torch.Tensor
    header_scatter: torch.Tensor
    row_mask: torch.Tensor
    col_mask: torch.Tensor
    cell_mask: torch.Tensor

    def validate(self, projection_dim: int) -> None:
        bt, max_n = self.col_mask.shape
        max_m = self.row_mask.shape[1]
        if len(self.table_ids) != bt:
            raise ValueError("evaluation table-ID count does not match masks")
        if self.cell_mask.shape != (bt, max_n, max_m):
            raise ValueError("bad prepared evaluation cell mask")
        if self.cell_features.ndim != 2 or self.cell_features.shape[1] != projection_dim:
            raise ValueError("bad prepared evaluation cell features")
        if self.header_features.ndim != 2 or self.header_features.shape[1] != projection_dim:
            raise ValueError("bad prepared evaluation header features")
        if self.cell_scatter.dtype != torch.int32 or self.header_scatter.dtype != torch.int32:
            raise TypeError("evaluation scatter indices must be int32")
        if self.cell_scatter.shape != (self.cell_features.shape[0],):
            raise ValueError("evaluation cell scatter length mismatch")
        if self.header_scatter.shape != (self.header_features.shape[0],):
            raise ValueError("evaluation header scatter length mismatch")

    def materialize(self, device: str | torch.device):
        device = torch.device(device)
        row_mask = self.row_mask.to(device)
        col_mask = self.col_mask.to(device)
        cell_mask = self.cell_mask.to(device)
        bt, max_n, max_m = cell_mask.shape
        dim = self.cell_features.shape[1]
        cells = torch.zeros(bt * max_n * max_m, dim, device=device)
        headers = torch.zeros(bt * max_n, dim, device=device)
        cells.index_copy_(
            0,
            self.cell_scatter.to(device=device, dtype=torch.long),
            self.cell_features.to(device=device, dtype=torch.float32),
        )
        headers.index_copy_(
            0,
            self.header_scatter.to(device=device, dtype=torch.long),
            self.header_features.to(device=device, dtype=torch.float32),
        )
        return (
            cells.view(bt, max_n, max_m, dim),
            headers.view(bt, max_n, dim),
            row_mask,
            col_mask,
            cell_mask,
        )


def write_eval_shard(path: str, metadata: dict, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = dict(metadata)
    header.update({"format": FORMAT_NAME, "format_version": FORMAT_VERSION})
    partial = path + ".partial"
    with open(partial, "wb") as output:
        pickle.dump(header, output, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(payload, output, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(partial, path)


def read_eval_shard(path: str):
    with open(path, "rb") as source:
        metadata = pickle.load(source)
        payload = pickle.load(source)
    if metadata.get("format") != FORMAT_NAME or metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported prepared evaluation shard: {path}")
    projection_dim = int(metadata["projection_dim"])
    payload.validate(projection_dim)
    return metadata, payload


def iter_eval_shards(paths: list[str]) -> Iterator[tuple[dict, object]]:
    for path in paths:
        yield read_eval_shard(path)
