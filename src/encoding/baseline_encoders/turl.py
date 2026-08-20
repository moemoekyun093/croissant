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
        # Attention weights are never consumed by TURL. Leaving
        # need_weights=True (PyTorch's default) materializes the dense
        # attention-weight tensor solely to discard it, and also prevents
        # MultiheadAttention from taking its optimized SDPA path.
        attn_out, _ = self.self_attn(
            x, x, x, attn_mask=attn_bias, need_weights=False
        )
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
        max_attention_elements: int = 2_000_000,
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
        self.num_heads = num_heads
        if max_attention_elements <= 0:
            raise ValueError("max_attention_elements must be positive")
        # Dynamic batching budget measured as B * S_max^2, where B is a
        # TURL microbatch's table count and S_max its padded node length.
        # This is architecture-independent and directly tracks the dense
        # visibility-attention term. A table whose own S^2 exceeds the
        # budget is still valid; it simply runs alone.
        self.max_attention_elements = max_attention_elements

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

        batch_ids = self.tokenizer(
            all_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.cell_max_tokens,
        )["input_ids"] if all_texts else []
        tok_lists = [
            ids if ids else [self.tokenizer.unk_token_id]
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

    def _prepare_table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """Build one table's pre-contextualization node sequence.

        Returns ``(x, attn_bias, n_metadata, n_rows, n_cols)``. ``x`` is
        [S, D], with cell mentions already pooled to one node each; the
        trainable masked Transformer is deliberately not run here so many
        independently masked tables can subsequently share one GPU call.
        """
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
        x = torch.cat([metadata_x, cell_x], dim=0)  # [metadata + cells, D]

        return x, attn_bias, n_metadata, n_rows, n_cols

    @staticmethod
    def _encoding_from_hidden(
        hidden: torch.Tensor,
        n_metadata: int,
        n_rows: int,
        n_cols: int,
    ) -> TableEncoding:
        """Restore one real (unpadded) TURL sequence to table tensors."""
        cell_emb = hidden[n_metadata:].reshape(n_rows, n_cols, hidden.shape[-1])
        row_emb = cell_emb.mean(dim=1)
        col_emb = cell_emb.mean(dim=0)
        table_emb = hidden[0]
        return TableEncoding(cell_emb, row_emb, col_emb, table_emb)

    def _forward_microbatch(
        self,
        tables: Sequence[Tuple[Sequence[str], Sequence[Sequence[object]], Optional[str]]],
    ) -> List[TableEncoding]:
        """Run a padded group of tables with independent visibility masks."""
        prepared = [self._prepare_table(headers, rows, caption) for headers, rows, caption in tables]
        batch_size = len(prepared)
        max_seq = max(item[0].shape[0] for item in prepared)
        device = self.token_embed.weight.device
        dtype = prepared[0][0].dtype

        # Each table is an independent batch element. Real queries can see
        # exactly their original visibility set; padded keys remain blocked.
        # A padded query is allowed to see only itself to avoid an all-masked
        # softmax row (NaN), and all padded outputs are discarded afterward.
        x = torch.zeros(batch_size, max_seq, self.hidden_size, device=device, dtype=dtype)
        blocked = torch.ones(batch_size, max_seq, max_seq, device=device, dtype=torch.bool)
        real_lengths: List[int] = []
        for b, (table_x, table_bias, _n_metadata, _n_rows, _n_cols) in enumerate(prepared):
            seq_len = table_x.shape[0]
            real_lengths.append(seq_len)
            x[b, :seq_len] = table_x
            blocked[b, :seq_len, :seq_len] = torch.isneginf(table_bias)
            if seq_len < max_seq:
                pad = torch.arange(seq_len, max_seq, device=device)
                blocked[b, pad, pad] = False

        # MultiheadAttention accepts a distinct mask per sample only in
        # [B * num_heads, S, S] form. For a singleton, the cheaper 2-D mask
        # is equivalent and avoids an unnecessary per-head copy.
        if batch_size == 1:
            batched_mask = blocked[0]
        else:
            batched_mask = blocked.repeat_interleave(self.num_heads, dim=0)

        for layer in self.layers:
            x = layer(x, batched_mask)

        results: List[TableEncoding] = []
        for b, (_table_x, _bias, n_metadata, n_rows, n_cols) in enumerate(prepared):
            hidden = x[b, : real_lengths[b]]
            results.append(self._encoding_from_hidden(hidden, n_metadata, n_rows, n_cols))
        return results

    def forward(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str] = None,
    ) -> TableEncoding:
        x, attn_bias, n_metadata, n_rows, n_cols = self._prepare_table(headers, rows, caption)

        for layer in self.layers:
            x = layer(x.unsqueeze(0), attn_bias).squeeze(0)
        return self._encoding_from_hidden(x, n_metadata, n_rows, n_cols)

    def forward_batch(
        self,
        tables: Sequence[Tuple[Sequence[str], Sequence[Sequence[object]], Optional[str]]],
    ) -> List[TableEncoding]:
        """Encode a candidate pool in order-preserving dynamic microbatches.

        Candidate selection, hard negatives, deduplication, and the
        query-by-candidate positive mask have already been resolved before
        this method is called. We may therefore reorder tables solely for
        efficient encoding, then scatter their encodings back to the exact
        input order before scoring.

        Tables are sorted by an upper bound on their TURL node count and
        greedily grouped while ``B * S_max^2`` stays below
        ``max_attention_elements``. Different row/column shapes are safe:
        padding plus a per-table visibility mask prevents any cross-table or
        padded-node interaction. Outlier tables automatically run alone.
        """
        if not tables:
            return []

        estimates: List[int] = []
        for headers, rows, caption in tables:
            validate_table(headers, rows)
            metadata_texts = len(headers) + (1 if caption else 0)
            estimates.append(
                1 + len(rows) * len(headers) + self.cell_max_tokens * metadata_texts
            )

        order = sorted(range(len(tables)), key=estimates.__getitem__)
        groups: List[List[int]] = []
        current: List[int] = []
        current_max = 0
        for idx in order:
            proposed_max = max(current_max, estimates[idx])
            proposed_cost = (len(current) + 1) * proposed_max * proposed_max
            if current and proposed_cost > self.max_attention_elements:
                groups.append(current)
                current = []
                current_max = 0
            current.append(idx)
            current_max = max(current_max, estimates[idx])
        if current:
            groups.append(current)

        import time

        is_cuda = self.token_embed.weight.device.type == "cuda"
        profile_timings = getattr(self, "_profile_timings", False)
        if is_cuda and profile_timings:
            torch.cuda.synchronize()
        started = time.perf_counter()

        results: List[Optional[TableEncoding]] = [None] * len(tables)
        for group in groups:
            group_results = self._forward_microbatch([tables[i] for i in group])
            for original_idx, encoding in zip(group, group_results):
                results[original_idx] = encoding

        if is_cuda and profile_timings:
            torch.cuda.synchronize()
        # TURL's frozen lookup/tokenization is tiny compared with its
        # trainable masked stack. Attribute the combined batched call to the
        # network bucket rather than repeating the adapter's old and highly
        # misleading "97% frozen backbone / 0% network" report.
        self._last_frozen_s = 0.0
        self._last_network_s = time.perf_counter() - started

        assert all(result is not None for result in results)
        return [result for result in results if result is not None]
