"""
Shared data structures and utilities for all table-encoder baselines.

Every encoder in this package implements the same contract so they can be
swapped into a retrieval pipeline interchangeably:

    encoder = SomeTableEncoder(...)
    out = encoder(headers=["Name", "Country"], rows=[["Alice", "US"], ["Bo", "CN"]])
    out.cell_embeddings   # [n_rows, n_cols, dim]
    out.row_embeddings    # [n_rows, dim]
    out.col_embeddings    # [n_cols, dim]
    out.table_embedding   # [dim]

This makes it trivial to plug any of them into a query-table or table-table
similarity module: the downstream code only ever depends on `TableEncoding`,
never on model-specific internals.

Deliberately NOT shared here: how headers get folded into cell text. Each
paper does this differently (TABBIE treats the header as literal row 0;
StruBERT emits "[header] [type] [value]" per cell; TURL keeps header tokens
separate from cell tokens and relies on its visibility matrix instead of
string concatenation; TAPAS's tokenizer takes headers as DataFrame column
names; HyTrel gives the header text directly to the column-hyperedge, not
to the cell/node). Folding that into one shared `serialize_cell()` would
have hidden a real per-paper design choice behind a fake abstraction, so
each encoder file owns its own header-handling logic instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class TableEncoding:
    """Uniform output contract returned by every encoder's forward()."""

    cell_embeddings: torch.Tensor  # [n_rows, n_cols, dim]
    row_embeddings: torch.Tensor  # [n_rows, dim]
    col_embeddings: torch.Tensor  # [n_cols, dim]
    table_embedding: torch.Tensor  # [dim]

    def to(self, device) -> "TableEncoding":
        return TableEncoding(
            cell_embeddings=self.cell_embeddings.to(device),
            row_embeddings=self.row_embeddings.to(device),
            col_embeddings=self.col_embeddings.to(device),
            table_embedding=self.table_embedding.to(device),
        )


def clean_cell(value: object) -> str:
    """Normalize a raw cell value into a short string. Cheap but consistent
    across every encoder so results are comparable."""
    s = "" if value is None else str(value)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "[EMPTY]"


class BaseTableEncoder(nn.Module):
    """Abstract base class. Subclasses implement `forward`.

    Parameters shared by all encoders:
        model_name: HF backbone checkpoint used to initialize token embeddings.
        device: torch device the module (and its outputs) should live on.
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def forward(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str] = None,
    ) -> TableEncoding:
        raise NotImplementedError

    @torch.no_grad()
    def encode(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str] = None,
    ) -> TableEncoding:
        """Convenience inference-mode wrapper around forward()."""
        self.eval()
        return self.forward(headers, rows, caption)


def mean_pool_span(hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Mean-pool a [seq_len, dim] tensor over token span [start, end)."""
    if end <= start:
        return torch.zeros(hidden.size(-1), device=hidden.device, dtype=hidden.dtype)
    return hidden[start:end].mean(dim=0)


def validate_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    n_cols = len(headers)
    for i, row in enumerate(rows):
        if len(row) != n_cols:
            raise ValueError(
                f"Row {i} has {len(row)} cells but table has {n_cols} headers"
            )
    if n_cols == 0 or len(rows) == 0:
        raise ValueError("Table must have at least one row and one column")
