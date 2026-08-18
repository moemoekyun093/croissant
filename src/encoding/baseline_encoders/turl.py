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

Visibility rule (Sec. 4.2-4.3 of the paper): caption/topic elements are
globally visible; header tokens and entity cells are visible only to elements
in their *same row* or *same column*. Thus, a header conditions cells in its
column but does not become a table-wide shortcut. This is strictly sparser
than TABBIE's alternating full row/column attention.

TURL's input is a mixed sequence: metadata stays as individual tokens,
whereas each entity cell is ONE input item.  The original model combines a
learned entity embedding with a mean-pooled entity-mention embedding before
contextualization.  This text-only retrieval adaptation has no linked entity
IDs, so each cell item is its mean-pooled word embedding.  Those already
grouped cell vectors -- not their individual word pieces -- then enter the
visibility-masked Transformer alongside caption/header tokens.
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

    def _build_visibility(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str],
    ) -> Tuple[torch.Tensor, List[int], List[List[List[int]]]]:
        """Returns:
        attn_bias: additive visibility mask over metadata-token and cell nodes
        metadata_ids: [CLS], caption-token, and header-token ids; caption
            tokens are globally-visible nodes, while header tokens are
            column-local nodes
        cell_token_ids[i][j]: word-piece ids for cell (i, j), to be pooled
            into ONE cell node *before* the Transformer
        """
        n_rows, n_cols = len(rows), len(headers)

        # Batch-tokenize every text (caption + headers + every cell) in ONE
        # tokenizer call up front instead of one self._tokenize_cell (i.e.
        # one Python-level .encode() call) per text -- same bug/fix as
        # hytrel.py's _mean_pool_texts and bert_baseline.py's
        # _build_sequence, missed here during this session's earlier
        # vectorization audit (that pass only checked GPU-tensor loops,
        # not tokenizer calls). Assembly/truncation logic below is
        # otherwise unchanged -- this only replaces WHERE each text gets
        # tokenized, not the sequence-building order.
        all_texts: List[str] = []
        if caption:
            all_texts.append(clean_cell(caption))
        all_texts.extend(clean_cell(h) for h in headers)
        for i in range(n_rows):
            for j in range(n_cols):
                all_texts.append(clean_cell(rows[i][j]))

        batch_ids = self.tokenizer(all_texts, add_special_tokens=False)["input_ids"] if all_texts else []
        tok_lists = [
            ids[: self.cell_max_tokens] if ids else [self.tokenizer.unk_token_id]
            for ids in batch_ids
        ]
        ptr = 0

        metadata_ids: List[int] = [self.tokenizer.cls_token_id]
        # -1 is globally visible ([CLS] / caption). Headers get unique
        # negative rows below so same_row never makes headers from different
        # columns visible to one another.
        token_row = [-1]
        token_col = [-1]

        if caption:
            ids = tok_lists[ptr]
            ptr += 1
            metadata_ids += ids
            token_row += [-1] * len(ids)
            token_col += [-1] * len(ids)

        for j, h in enumerate(headers):
            ids = tok_lists[ptr]
            ptr += 1
            metadata_ids += ids
            token_row += [-(j + 2)] * len(ids)
            token_col += [j] * len(ids)

        # Each cell becomes exactly ONE node in the structure-aware encoder.
        # Its word pieces are pooled into a mention embedding in forward(),
        # before any row/column contextualization.  This mirrors TURL's
        # entity-cell input representation (minus unavailable entity IDs).
        cell_token_ids: List[List[List[int]]] = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_cols):
                ids = tok_lists[ptr]
                ptr += 1
                cell_token_ids[i][j] = ids
                token_row.append(i)
                token_col.append(j)

        seq_len = len(token_row)
        device = self.token_embed.weight.device
        row_t = torch.tensor(token_row, device=device)
        col_t = torch.tensor(token_col, device=device)

        # visibility[a, b] = True if token b is visible from token a
        same_row = row_t.unsqueeze(0) == row_t.unsqueeze(1)
        same_col = col_t.unsqueeze(0) == col_t.unsqueeze(1)
        globally_visible = (row_t == -1).unsqueeze(0) | (row_t == -1).unsqueeze(1)
        self_visible = torch.eye(seq_len, dtype=torch.bool, device=device)
        visible = same_row | same_col | globally_visible | self_visible

        attn_bias = torch.zeros(seq_len, seq_len, device=device)
        attn_bias.masked_fill_(~visible, float("-inf"))

        return attn_bias, metadata_ids, cell_token_ids

    def forward(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str] = None,
    ) -> TableEncoding:
        validate_table(headers, rows)
        n_rows, n_cols = len(rows), len(headers)

        attn_bias, metadata_ids, cell_token_ids = self._build_visibility(headers, rows, caption)

        # Metadata is token-level in TURL, but table cells are entity-level.
        # Pool each cell mention's word embeddings NOW, before the masked
        # Transformer, rather than contextualizing its word pieces and
        # pooling afterward (the previous, non-TURL-faithful behavior).
        device = self.token_embed.weight.device
        metadata_x = self.token_embed(
            torch.tensor(metadata_ids, device=device)
        )  # [n_metadata_tokens, D]
        flat_cell_ids = [token_id for row in cell_token_ids for cell in row for token_id in cell]
        cell_lengths = [len(cell) for row in cell_token_ids for cell in row]
        cell_word_x = self.token_embed(
            torch.tensor(flat_cell_ids, device=device)
        )  # [sum(cell_lengths), D]
        cell_lengths_t = torch.tensor(cell_lengths, device=device)
        zero = torch.zeros(1, self.hidden_size, device=device, dtype=cell_word_x.dtype)
        cell_csum = torch.cat([zero, cell_word_x.cumsum(dim=0)], dim=0)
        cell_ends = cell_lengths_t.cumsum(dim=0)
        cell_starts = cell_ends - cell_lengths_t
        cell_x = (cell_csum[cell_ends] - cell_csum[cell_starts]) / cell_lengths_t.to(
            dtype=cell_word_x.dtype
        ).unsqueeze(-1)

        n_metadata = metadata_x.shape[0]
        x = torch.cat([metadata_x, cell_x], dim=0).unsqueeze(0)  # [1, metadata + cells, D]

        for layer in self.layers:
            x = layer(x, attn_bias)
        hidden = x.squeeze(0)  # [seq, D]

        # Cell nodes are already pooled before contextualization, so their
        # contextualized vectors can be reshaped directly back to the table.
        cell_emb = hidden[n_metadata:].reshape(n_rows, n_cols, self.hidden_size)

        row_emb = cell_emb.mean(dim=1)
        col_emb = cell_emb.mean(dim=0)
        table_emb = hidden[0]  # CLS, globally visible to (and from) every cell

        return TableEncoding(cell_emb, row_emb, col_emb, table_emb)
