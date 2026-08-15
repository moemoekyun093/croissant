"""
Baseline #5 — TURL (Deng et al., 2020/2021, "TURL: Table Understanding
through Representation Learning", VLDB'21).

Reference implementation: https://github.com/sunlab-osu/TURL (custom
`pytorch-pretrained-bert`-based structure-aware Transformer). TURL's entity
linking / pretraining machinery is out of scope for a retrieval baseline;
what we port faithfully is the core structural idea: a **visibility
matrix** used as an additive attention mask, so that a token can only
attend to tokens/entities that are "structurally visible" to it, instead
of full (BERT) or row/column-alternating (TABBIE) attention.

Visibility rule (Sec 4.3-4.4 of the paper): caption and header tokens are
globally visible to everything (they provide shared context); a cell is
visible to every other cell in the *same row* and every other cell in the
*same column*; a cell is NOT visible to a cell in neither the same row nor
same column. This is strictly sparser than TABBIE's alternating full
row/column attention (TABBIE contextualizes a cell against *every* other
cell in its row AND *every* cell in its column at every layer; TURL applies
one single joint mask per layer, which is a cheaper, single-pass version of
the same locality bias, but does not average two separate row/column
representations).

We build token embeddings from BERT's embedding table (so vocabulary /
subword behavior matches BERT) but the contextualization itself is done by
a small stack of masked Transformer encoder layers using the visibility
matrix as an additive attention bias -- this is what TURL calls its
"structure-aware Transformer encoder".
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .common import BaseTableEncoder, TableEncoding, clean_cell, validate_table


class _MaskedEncoderLayer(nn.Module):
    """A single Transformer layer that accepts an additive [seq, seq] bias
    (the visibility matrix, as -inf / 0) instead of the usual padding-only
    attention mask."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        # x: [1, seq, D], attn_bias: [seq, seq] additive mask
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_bias)
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.linear2(F.gelu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff))
        return x


class TurlTableEncoder(BaseTableEncoder):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_layers: int = 4,
        num_heads: int = 8,
        cell_max_tokens: int = 8,
        device: Optional[str] = None,
    ):
        super().__init__(model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Reuse BERT's pretrained *token* embeddings only (TURL initializes
        # its embedding table from BERT); contextualization is our own
        # visibility-masked stack, not BERT's full-attention encoder.
        bert = AutoModel.from_pretrained(model_name)
        self.token_embed = bert.embeddings.word_embeddings.to(self.device)
        self.hidden_size = bert.config.hidden_size
        self.cell_max_tokens = cell_max_tokens

        # Frozen, same as every other baseline's backbone -- see
        # bert_baseline.py's comment. This is just BERT's word-embedding
        # LOOKUP table (not a full backbone forward pass, already cheap),
        # but freezing it keeps "only the layers on top train" consistent
        # across every baseline. self.layers (the visibility-masked
        # attention stack) remains fully trainable.
        for p in self.token_embed.parameters():
            p.requires_grad = False

        self.layers = nn.ModuleList(
            [
                _MaskedEncoderLayer(self.hidden_size, num_heads, self.hidden_size * 4).to(self.device)
                for _ in range(num_layers)
            ]
        )

    def _tokenize_cell(self, text: str) -> List[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids[: self.cell_max_tokens] if ids else [self.tokenizer.unk_token_id]

    def _build_visibility(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str],
    ) -> Tuple[torch.Tensor, List[int], List[List[Tuple[int, int]]], Tuple[int, int]]:
        """Returns:
        input_ids: flat list of token ids
        cell_spans[i][j] = (start, end) token span for cell (i, j)
        cls_span: token span of the [CLS]/caption/header block (globally visible)
        """
        n_rows, n_cols = len(rows), len(headers)
        input_ids: List[int] = [self.tokenizer.cls_token_id]
        token_row = [-1]  # -1 = globally visible (CLS/caption/headers)
        token_col = [-1]

        if caption:
            ids = self._tokenize_cell(clean_cell(caption))
            input_ids += ids
            token_row += [-1] * len(ids)
            token_col += [-1] * len(ids)

        header_spans = []
        for j, h in enumerate(headers):
            ids = self._tokenize_cell(clean_cell(h))
            start = len(input_ids)
            input_ids += ids
            token_row += [-1] * len(ids)  # headers are globally visible, but tagged with their column too
            token_col += [j] * len(ids)
            header_spans.append((start, len(input_ids)))

        cell_spans: List[List[Tuple[int, int]]] = [[(0, 0)] * n_cols for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_cols):
                ids = self._tokenize_cell(clean_cell(rows[i][j]))
                start = len(input_ids)
                input_ids += ids
                token_row += [i] * len(ids)
                token_col += [j] * len(ids)
                cell_spans[i][j] = (start, len(input_ids))

        seq_len = len(input_ids)
        row_t = torch.tensor(token_row)
        col_t = torch.tensor(token_col)

        # visibility[a, b] = True if token b is visible from token a
        same_row = row_t.unsqueeze(0) == row_t.unsqueeze(1)
        same_col = col_t.unsqueeze(0) == col_t.unsqueeze(1)
        globally_visible = (row_t == -1).unsqueeze(0) | (row_t == -1).unsqueeze(1)
        self_visible = torch.eye(seq_len, dtype=torch.bool)
        visible = same_row | same_col | globally_visible | self_visible

        attn_bias = torch.zeros(seq_len, seq_len)
        attn_bias.masked_fill_(~visible, float("-inf"))

        return attn_bias.to(self.device), input_ids, cell_spans, (0, len(input_ids))

    def forward(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str] = None,
    ) -> TableEncoding:
        validate_table(headers, rows)
        n_rows, n_cols = len(rows), len(headers)

        attn_bias, input_ids, cell_spans, _ = self._build_visibility(headers, rows, caption)
        ids_t = torch.tensor(input_ids, device=self.device)
        x = self.token_embed(ids_t).unsqueeze(0)  # [1, seq, D]

        for layer in self.layers:
            x = layer(x, attn_bias)
        hidden = x.squeeze(0)  # [seq, D]

        # Vectorized mean-pool over every cell's token span at once via a
        # prefix-sum trick, instead of one mean_pool_span (small GPU op)
        # call per cell in a Python double loop -- same fix as
        # bert_baseline.py, see that file's comment for the full
        # rationale. Every cell span here is guaranteed non-empty
        # (_tokenize_cell always returns at least [unk_token_id]), so no
        # empty-span fallback branch is needed, unlike bert_baseline.py.
        zero_row = torch.zeros(1, self.hidden_size, device=self.device, dtype=hidden.dtype)
        csum = torch.cat([zero_row, hidden.cumsum(dim=0)], dim=0)  # [seq_len+1, dim]
        starts = torch.tensor(
            [[cell_spans[i][j][0] for j in range(n_cols)] for i in range(n_rows)],
            device=self.device,
        )
        ends = torch.tensor(
            [[cell_spans[i][j][1] for j in range(n_cols)] for i in range(n_rows)],
            device=self.device,
        )
        lengths = (ends - starts).clamp(min=1).unsqueeze(-1).to(hidden.dtype)
        cell_emb = (csum[ends] - csum[starts]) / lengths

        row_emb = cell_emb.mean(dim=1)
        col_emb = cell_emb.mean(dim=0)
        table_emb = hidden[0]  # CLS, globally visible to (and from) every cell

        return TableEncoding(cell_emb, row_emb, col_emb, table_emb)
