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

from typing import List, Optional, Sequence, Tuple

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

    def _pool_cells(
        self,
        hidden: torch.Tensor,
        token_type_ids: torch.Tensor,
        n_rows: int,
        n_cols: int,
    ) -> torch.Tensor:
        """Vectorized scatter-mean over every token at once, instead of a
        Python loop over every token with an individual .item() call
        each iteration (forces a GPU->CPU sync per token -- with up to
        512 tokens/table and ~192 tables/step, that's up to ~98,000
        blocking syncs per training step) plus a scalar += into
        cell_emb. index_add_ does the same accumulation as one
        vectorized op, entirely on-device.

        Shared by forward() (single table, hidden/token_type_ids = that
        table's own) and forward_batch() (hidden/token_type_ids = one
        table's own SLICE out of a batched backbone output, real tokens
        only -- see forward_batch's docstring).
        """
        # channel indices per HF TapasTokenizer: 0 segment, 1 column, 2 row, 3 prev-label,
        # 4 column-rank, 5 inv-column-rank, 6 numeric-relation
        column_ids = token_type_ids[:, 1]
        row_ids = token_type_ids[:, 2]

        valid = (row_ids >= 1) & (column_ids >= 1) & (row_ids <= n_rows) & (column_ids <= n_cols)
        flat_idx = (row_ids[valid] - 1) * n_cols + (column_ids[valid] - 1)  # [n_valid]
        valid_hidden = hidden[valid]  # [n_valid, D]

        cell_emb_flat = torch.zeros(n_rows * n_cols, self.hidden_size, device=self.device, dtype=hidden.dtype)
        counts_flat = torch.zeros(n_rows * n_cols, device=self.device, dtype=hidden.dtype)
        cell_emb_flat.index_add_(0, flat_idx, valid_hidden)
        counts_flat.index_add_(0, flat_idx, torch.ones(flat_idx.size(0), device=self.device, dtype=hidden.dtype))

        counts = counts_flat.clamp(min=1).unsqueeze(-1)
        return (cell_emb_flat / counts).view(n_rows, n_cols, self.hidden_size)

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
        token_type_ids = enc["token_type_ids"][0]  # [seq_len, 7]

        cell_emb = self._pool_cells(hidden, token_type_ids, n_rows, n_cols)
        row_emb = cell_emb.mean(dim=1)
        col_emb = cell_emb.mean(dim=0)
        table_emb = hidden[0]  # [CLS]

        return TableEncoding(cell_emb, row_emb, col_emb, table_emb)

    def forward_batch(
        self,
        tables: List[Tuple[Sequence[str], Sequence[Sequence[object]], Optional[str]]],
    ) -> List[TableEncoding]:
        """Batched version of forward(): runs ONE TapasModel forward pass
        for MULTIPLE tables instead of one pass per table -- same fix,
        same rationale as bert_baseline.py's forward_batch (see its
        docstring for the full "why ours was faster" explanation).

        TapasTokenizer's row/column/rank structural ids are inherently
        PER-TABLE (they encode that specific table's own shape), so
        tokenization itself can't be batched across different tables in
        one tokenizer call the way bert_baseline.py's plain-text
        tokenization can -- each table still gets its own tokenizer call
        here (CPU-side work, not the GPU bottleneck). What DOES get
        batched is the expensive part: instead of calling
        self.backbone(**enc) once per table (batch-of-1 each time), every
        table's already-tokenized input_ids/attention_mask/token_type_ids
        are padded to a common length and stacked into one batch, and
        self.backbone(...) is called exactly ONCE for the whole batch.

        tables: list of (headers, rows, caption) tuples, one per table
                (caption is accepted for API parity with forward() but
                unused here, same as forward()'s own caption parameter
                on the current adapter call path -- TAPAS has no comparable
                caption/query use in this codebase's finetuning setup).
        returns: list of TableEncoding, one per table, SAME ORDER as the
                 input list.
        """
        if not tables:
            return []

        per_table_input_ids: List[torch.Tensor] = []
        per_table_attn: List[torch.Tensor] = []
        per_table_token_type: List[torch.Tensor] = []
        per_table_shape: List[Tuple[int, int]] = []

        for headers, rows, _caption in tables:
            validate_table(headers, rows)
            df = pd.DataFrame(
                [[clean_cell(v) for v in row] for row in rows],
                columns=[clean_cell(h) for h in headers],
            ).astype(str)
            enc = self.tokenizer(
                table=df,
                queries="",
                padding=False,  # pad ONCE below, across the whole batch -- not per table here
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            per_table_input_ids.append(enc["input_ids"][0])
            per_table_attn.append(enc["attention_mask"][0])
            per_table_token_type.append(enc["token_type_ids"][0])  # [seq_len_i, 7]
            per_table_shape.append((len(rows), len(headers)))

        B = len(tables)
        maxlen = max(t.size(0) for t in per_table_input_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        ids_t = torch.full((B, maxlen), pad_id, dtype=torch.long)
        attn_t = torch.zeros((B, maxlen), dtype=torch.long)
        # 7 structural channels, padded with 0 -- a 0 in the row/column
        # channels is exactly what _pool_cells' `valid` mask already
        # treats as "not a real cell" (row_ids/column_ids >= 1 required),
        # so padded positions are automatically excluded from pooling
        # without any extra handling, on top of attention_mask=0 already
        # excluding them from the backbone's attention entirely.
        token_type_t = torch.zeros((B, maxlen, 7), dtype=torch.long)
        for i in range(B):
            L = per_table_input_ids[i].size(0)
            ids_t[i, :L] = per_table_input_ids[i]
            attn_t[i, :L] = per_table_attn[i]
            token_type_t[i, :L] = per_table_token_type[i]

        ids_t = ids_t.to(self.device)
        attn_t = attn_t.to(self.device)
        token_type_t = token_type_t.to(self.device)

        # Defensive checks BEFORE the backbone call -- same rationale as
        # bert_baseline.py's forward_batch (see its comment for the full
        # explanation of why this matters: an out-of-range id here would
        # otherwise surface as an unattributed, ASYNCHRONOUSLY-reported
        # CUDA device-side assert at some unrelated later call).
        vocab_size = getattr(self.backbone.config, "vocab_size", None)
        if vocab_size is not None and bool((ids_t.ge(vocab_size) | ids_t.lt(0)).any()):
            bad = [
                (i, tid.item())
                for i in range(B)
                for tid in per_table_input_ids[i]
                if tid.item() < 0 or tid.item() >= vocab_size
            ]
            raise ValueError(
                f"forward_batch: token id(s) outside this backbone's vocab_size "
                f"({vocab_size}) -- would have crashed as an unattributed CUDA "
                f"device-side assert instead. offending "
                f"(table_index_in_this_call, token_id): {bad[:5]}"
            )

        max_pos = getattr(self.backbone.config, "max_position_embeddings", None)
        if max_pos is not None and maxlen > max_pos:
            offending = [
                (i, per_table_input_ids[i].size(0))
                for i in range(B)
                if per_table_input_ids[i].size(0) > max_pos
            ]
            raise ValueError(
                f"forward_batch: batch's max sequence length ({maxlen}) exceeds "
                f"this backbone's max_position_embeddings ({max_pos}) -- "
                f"self.max_length={self.max_length} should prevent this via "
                f"the tokenizer's own truncation, so either self.max_length is "
                f"configured above max_pos, or there's an edge case letting a "
                f"sequence through un-truncated. offending "
                f"(table_index_in_this_call, seq_len): {offending[:5]}"
            )

        with torch.no_grad():
            out = self.backbone(input_ids=ids_t, attention_mask=attn_t, token_type_ids=token_type_t)
        hidden_batch = out.last_hidden_state  # [B, maxlen, D]

        results: List[TableEncoding] = []
        for i in range(B):
            seq_len = per_table_input_ids[i].size(0)
            hidden = hidden_batch[i, :seq_len]
            token_type_ids = token_type_t[i, :seq_len]
            n_rows, n_cols = per_table_shape[i]
            cell_emb = self._pool_cells(hidden, token_type_ids, n_rows, n_cols)
            row_emb = cell_emb.mean(dim=1)
            col_emb = cell_emb.mean(dim=0)
            table_emb = hidden[0]
            results.append(TableEncoding(cell_emb, row_emb, col_emb, table_emb))

        return results
