"""
Baseline #4 — TAPAS (Herzig et al., 2020, "TAPAS: Weakly Supervised Table
Parsing via Pre-training").

Unlike TABBIE/StruBERT/TURL/HyTrel, TAPAS is *natively supported* in
`transformers` (`TapasTokenizer` + `TapasModel`), so there is no need to
port code from google-research/tapas — we use the library directly.

Architecture recap: the whole table is linearized into a single sequence
(everything squashed into one linear representation, as you described), but
each token additionally gets `token_type_ids` encoding structural position:
segment id, column id, row id, rank id, etc. Attention is standard full
self-attention (like plain BERT) — the structural bias comes entirely from
these extra embeddings added at the input, not from any masking or
multi-stage attention.

Since `TapasModel` doesn't give row/column/table embeddings out of the box
(only per-token), we reconstruct them the same way Observatory
(https://github.com/superctj/observatory, VLDB'24) does: aggregate token
embeddings by their `column_ids`/`row_ids` token_type channels.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import torch
from transformers import TapasModel, TapasTokenizer

from .common import BaseTableEncoder, TableEncoding, clean_cell, validate_table


class TapasTableEncoder(BaseTableEncoder):
    def __init__(
        self,
        model_name: str = "google/tapas-base",
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        super().__init__(model_name, device)
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.backbone = TapasModel.from_pretrained(model_name).to(self.device)
        self.hidden_size = self.backbone.config.hidden_size
        self.max_length = max_length

        # Frozen feature extractor -- see bert_baseline.py's comment for
        # the full rationale. Like plain BERT, TAPAS has no separate
        # on-top layer of its own (the pretrained backbone IS the whole
        # model for this baseline too -- see adapter.py's
        # _NUM_LAYERS_KWARG comment), so this leaves the encoder itself
        # with zero trainable parameters; DiscriminatorHead/QueryEncoder/
        # MultiScorer remain the trainable parts for this baseline.
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        """Keep the frozen backbone permanently in eval mode -- see
        bert_baseline.py's train() override for the full rationale."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str] = None,
        query: str = "",
    ) -> TableEncoding:
        validate_table(headers, rows)
        n_rows, n_cols = len(rows), len(headers)

        df = pd.DataFrame(
            [[clean_cell(v) for v in row] for row in rows],
            columns=[clean_cell(h) for h in headers],
        ).astype(str)

        enc = self.tokenizer(
            table=df,
            queries=query or "",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.backbone(**enc)
        hidden = out.last_hidden_state[0]  # [seq_len, D]

        token_type_ids = enc["token_type_ids"][0]  # [seq_len, 7]: TAPAS's 7 structural channels
        # channel indices per HF TapasTokenizer: 0 segment, 1 column, 2 row, 3 prev-label,
        # 4 column-rank, 5 inv-column-rank, 6 numeric-relation
        column_ids = token_type_ids[:, 1]
        row_ids = token_type_ids[:, 2]

        # Vectorized scatter-mean over every token at once, instead of a
        # Python loop over every token with an individual .item() call
        # each iteration (forces a GPU->CPU sync per token -- with up to
        # 512 tokens/table and ~192 tables/step, that's up to ~98,000
        # blocking syncs per training step) plus a scalar += into
        # cell_emb. index_add_ does the same accumulation as one
        # vectorized op, entirely on-device.
        valid = (row_ids >= 1) & (column_ids >= 1) & (row_ids <= n_rows) & (column_ids <= n_cols)
        flat_idx = (row_ids[valid] - 1) * n_cols + (column_ids[valid] - 1)  # [n_valid]
        valid_hidden = hidden[valid]  # [n_valid, D]

        cell_emb_flat = torch.zeros(n_rows * n_cols, self.hidden_size, device=self.device, dtype=hidden.dtype)
        counts_flat = torch.zeros(n_rows * n_cols, device=self.device, dtype=hidden.dtype)
        cell_emb_flat.index_add_(0, flat_idx, valid_hidden)
        counts_flat.index_add_(0, flat_idx, torch.ones(flat_idx.size(0), device=self.device, dtype=hidden.dtype))

        counts = counts_flat.clamp(min=1).unsqueeze(-1)
        cell_emb = (cell_emb_flat / counts).view(n_rows, n_cols, self.hidden_size)

        row_emb = cell_emb.mean(dim=1)
        col_emb = cell_emb.mean(dim=0)
        table_emb = hidden[0]  # [CLS]

        return TableEncoding(cell_emb, row_emb, col_emb, table_emb)
