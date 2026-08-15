"""
Baseline #3 — StruBERT (Trabelsi et al., 2022, "StruBERT: Structure-aware
BERT for Table Search and Matching", WWW'22).

Reference implementation: https://github.com/medtray/StruBERT, which reuses
StruBERT's own TaBERT-derived `input_formatter.py`. Header handling there
is explicit and exact, not "header : value": `get_cell_input()` builds each
cell's tokens from a configurable template, defaulting to
`['column', 'type', 'value']` -- i.e. every cell is serialized as
`[header_name] [type] [value]`, where `type` is a single token describing
the column's inferred data type (StruBERT/TaBERT use `text` / `real` for
free text vs. numeric columns; see `Column.type` in that repo). This
matches the paper's eq. 6 definition of a cell as
`[header_name type cell_content]` exactly, and is what we reproduce below
via `_infer_type()` + `_serialize_cell()`.

StruBERT builds *two independent sequence views* of the table:
  - a column-based sequence per column j: [CLS] [header_j type] [SEP] [v_1j type] [SEP] ...
  - a row-based sequence per row i:       [CLS] [header_1 type v_i1] [SEP] [header_2 type v_i2] [SEP] ...
Each is run through BERT, then cell-wise average pooling collapses
sub-word tokens back to one vector per cell, giving:
  - C = {c_1, ..., c_n_cols}: column embeddings (one per column-sequence)
  - R = {r_1, ..., r_n_rows}: row embeddings (one per row-sequence)
as well as coarse [CLS] vectors per column-/row- sequence.

StruBERT then applies:
  - *vertical* self-attention over C to refine column embeddings
  - *horizontal* self-attention over R to refine row embeddings
producing two fine-grained feature sets (contextualized R, C) and two
coarse-grained feature sets (pooled column-CLS, pooled row-CLS). The four
are concatenated and linearly projected to a single table representation
(this is the "late interaction" framing the user described: row-view and
column-view are encoded almost independently and only fused at the end,
unlike TABBIE which fuses every layer).

Note: this module reuses a single shared BERT for both views (the original
paper initializes the row/column encoders from TaBERT checkpoints, which
are hard to source generically; a shared bert-base-uncased backbone is a
faithful, checkpoint-agnostic stand-in for baseline comparison purposes).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .common import BaseTableEncoder, TableEncoding, clean_cell, validate_table


def _infer_column_type(values: Sequence[object]) -> str:
    """Cheap real/text type inference, standing in for TaBERT/StruBERT's
    `Column.type` (they infer this from a type-annotated corpus at
    preprocessing time; a lightweight heuristic here keeps this baseline
    self-contained). Swap this out for your own `cell_type.py` if you want
    corpus-consistent typing across your other encoders."""
    non_empty = [v for v in values if clean_cell(v) != "[EMPTY]"]
    if not non_empty:
        return "text"
    numeric = 0
    for v in non_empty:
        try:
            float(str(v).replace(",", ""))
            numeric += 1
        except ValueError:
            pass
    return "real" if numeric / len(non_empty) >= 0.8 else "text"


def _serialize_cell(header: str, cell_type: str, value: object) -> str:
    """`[header_name] [type] [value]` -- StruBERT/TaBERT's `get_cell_input()`
    default template `['column', 'type', 'value']` (see module docstring)."""
    return f"{clean_cell(header)} {cell_type} {clean_cell(value)}"


class StruBertTableEncoder(BaseTableEncoder):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_attn_layers: int = 3,
        num_heads: int = 8,
        seq_max_length: int = 256,
        device: Optional[str] = None,
    ):
        super().__init__(model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name).to(self.device)
        self.hidden_size = self.backbone.config.hidden_size
        self.seq_max_length = seq_max_length

        # Frozen feature extractor -- see bert_baseline.py's comment for
        # the full rationale. This backbone gets called TWICE per table
        # (once for the row-view sequences, once for the column-view
        # sequences via _cellwise_pool_sequence), both fully trainable
        # graphs retained until backward previously -- this is exactly
        # what caused the real OOM on cuda:3 (94.96/94.97 GiB used).
        # Freezing removes both graphs entirely; only vertical_attn/
        # horizontal_attn/fuse_proj below stay trainable.
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        def make_stack():
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=num_heads,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=num_attn_layers).to(self.device)

        self.vertical_attn = make_stack()  # refines column embeddings C
        self.horizontal_attn = make_stack()  # refines row embeddings R
        self.fuse_proj = nn.Linear(self.hidden_size * 4, self.hidden_size).to(self.device)

    def train(self, mode: bool = True):
        """Keep the frozen backbone permanently in eval mode -- see
        bert_baseline.py's train() override for the full rationale."""
        super().train(mode)
        self.backbone.eval()
        return self

    def _cellwise_pool_sequence(self, seq_texts: List[str]) -> torch.Tensor:
        """Given a list of one "row-sequence" or "column-sequence" string per
        row/column (already `[SEP]`-joined cell serializations), run BERT and
        return the [CLS] vector for each — this acts as the pooled
        coarse-grained representation for that row/column view.

        We also return, per sequence, the mean of all non-special tokens as
        the fine-grained "cell-wise pooled" embedding approximation.
        """
        enc = self.tokenizer(
            seq_texts,
            padding=True,
            truncation=True,
            max_length=self.seq_max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.backbone(**enc)
        hidden = out.last_hidden_state  # [N, L, D]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        # zero out CLS/SEP contribution to the fine-grained mean by using a
        # copy of the mask with position 0 (CLS) excluded
        fine_mask = mask.clone()
        fine_mask[:, 0, :] = 0.0
        fine = (hidden * fine_mask).sum(dim=1) / fine_mask.sum(dim=1).clamp(min=1.0)
        coarse = hidden[:, 0, :]
        return fine, coarse

    def _build_seqs(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str],
    ) -> Tuple[List[str], List[str]]:
        """Builds this table's own col_seqs/row_seqs (see forward()'s
        original comment) -- pure string assembly, no tokenizer/backbone
        call. Shared by forward() and forward_batch()."""
        n_rows, n_cols = len(rows), len(headers)
        cap = clean_cell(caption) if caption else ""

        # one inferred type per column, reused for every cell in that column
        col_types = [_infer_column_type([rows[i][j] for i in range(n_rows)]) for j in range(n_cols)]

        # column-based sequences: one per column; every cell (incl. the header
        # cell itself) is serialized as "[header] [type] [value]"
        col_seqs = []
        for j in range(n_cols):
            parts = [cap] if cap else []
            parts.append(_serialize_cell(headers[j], col_types[j], headers[j]))
            parts += [_serialize_cell(headers[j], col_types[j], rows[i][j]) for i in range(n_rows)]
            col_seqs.append(f" {self.tokenizer.sep_token} ".join(parts))

        # row-based sequences: one per row; same per-cell "[header] [type] [value]" format
        row_seqs = []
        for i in range(n_rows):
            parts = [cap] if cap else []
            parts += [_serialize_cell(headers[j], col_types[j], rows[i][j]) for j in range(n_cols)]
            row_seqs.append(f" {self.tokenizer.sep_token} ".join(parts))

        return col_seqs, row_seqs

    def _forward_from_pooled(
        self,
        col_fine: torch.Tensor,
        col_coarse: torch.Tensor,
        row_fine: torch.Tensor,
        row_coarse: torch.Tensor,
    ) -> TableEncoding:
        """Everything AFTER the (expensive, frozen-BERT) row/column-view
        pooling: vertical/horizontal self-attention + final fusion. Shared
        by forward() and forward_batch() -- identical computation either
        way, this stack only ever sees one table's own pooled vectors at
        a time (it's small/cheap relative to the frozen backbone, so it
        stays a per-table call even in forward_batch)."""
        # vertical self-attention over column embeddings C -> refined columns
        col_refined = self.vertical_attn(col_fine.unsqueeze(0)).squeeze(0)  # [n_cols, D]
        # horizontal self-attention over row embeddings R -> refined rows
        row_refined = self.horizontal_attn(row_fine.unsqueeze(0)).squeeze(0)  # [n_rows, D]

        # cell embedding = outer combination of its refined row and column vector
        cell_emb = (row_refined.unsqueeze(1) + col_refined.unsqueeze(0)) / 2.0  # [R, C, D]

        # four features -> concat -> project to final table embedding
        four_features = torch.cat(
            [row_refined.mean(0), col_refined.mean(0), row_coarse.mean(0), col_coarse.mean(0)],
            dim=-1,
        )
        table_emb = self.fuse_proj(four_features)

        return TableEncoding(cell_emb, row_refined, col_refined, table_emb)

    def forward(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str] = None,
    ) -> TableEncoding:
        validate_table(headers, rows)

        col_seqs, row_seqs = self._build_seqs(headers, rows, caption)

        col_fine, col_coarse = self._cellwise_pool_sequence(col_seqs)  # [n_cols, D] each
        row_fine, row_coarse = self._cellwise_pool_sequence(row_seqs)  # [n_rows, D] each

        return self._forward_from_pooled(col_fine, col_coarse, row_fine, row_coarse)

    def forward_batch(
        self,
        tables: Sequence[Tuple[Sequence[str], Sequence[Sequence[object]], Optional[str]]],
    ) -> List[TableEncoding]:
        """Batched version of forward(): runs the frozen BERT backbone
        TWICE TOTAL (once for every table's column-view sequences
        combined, once for every table's row-view sequences combined)
        instead of TWICE PER TABLE -- StruBERT is actually the worst
        offender of this "batch-of-1 backbone call" class of issue among
        the baselines, since forward() already called
        _cellwise_pool_sequence (i.e. self.backbone(**enc)) twice per
        table on its own. Same fix/rationale as bert_baseline.py's
        forward_batch (see its docstring) -- see adapter.py's
        forward_batch_cellwise for how this gets picked up automatically.

        tables: list of (headers, rows, caption) tuples, one per table.
        returns: list of TableEncoding, one per table, SAME ORDER as the
                 input list.
        """
        if not tables:
            return []

        all_col_seqs: List[str] = []
        all_row_seqs: List[str] = []
        col_offsets: List[Tuple[int, int]] = []
        row_offsets: List[Tuple[int, int]] = []

        col_off = 0
        row_off = 0
        for headers, rows, caption in tables:
            validate_table(headers, rows)
            col_seqs, row_seqs = self._build_seqs(headers, rows, caption)
            all_col_seqs.extend(col_seqs)
            all_row_seqs.extend(row_seqs)
            col_offsets.append((col_off, col_off + len(col_seqs)))
            row_offsets.append((row_off, row_off + len(row_seqs)))
            col_off += len(col_seqs)
            row_off += len(row_seqs)

        # ONE backbone call for every table's column-view sequences
        # combined, ONE for every table's row-view sequences combined --
        # _cellwise_pool_sequence already batches whatever list it's
        # given via one tokenizer call + one backbone call, so passing it
        # ALL tables' sequences at once (instead of one table's worth at
        # a time, looped) is the entire fix.
        all_col_fine, all_col_coarse = self._cellwise_pool_sequence(all_col_seqs)
        all_row_fine, all_row_coarse = self._cellwise_pool_sequence(all_row_seqs)

        results: List[TableEncoding] = []
        for (c_start, c_end), (r_start, r_end) in zip(col_offsets, row_offsets):
            results.append(
                self._forward_from_pooled(
                    all_col_fine[c_start:c_end],
                    all_col_coarse[c_start:c_end],
                    all_row_fine[r_start:r_end],
                    all_row_coarse[r_start:r_end],
                )
            )
        return results
