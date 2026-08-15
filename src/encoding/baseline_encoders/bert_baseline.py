"""
Baseline #1 — Plain BERT.

Flattens the whole table into one token sequence and runs vanilla BERT
self-attention over it: every token attends to every other token, with no
row/column structural bias beyond BERT's own absolute position ids (which
here just encode "flattening order", not table geometry).

This is the weakest structural baseline in the suite and exists mainly as a
lower bound: any of the structure-aware encoders (TABBIE, StruBERT, TAPAS,
TURL, HyTrel) should outperform this if table structure is actually useful
for your task.

Reference: standard `transformers.BertModel`, no external repo needed --
plain BERT has no notion of a "table" at all, so there is no paper-defined
convention for folding headers into cell text here. We use the same
"header : value" per-cell serialization the other baselines' papers use
in spirit (see e.g. StruBERT's "[header] [type] [value]"), but it's an
arbitrary choice made for this baseline specifically, not something
inherited from a reference implementation.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from .common import BaseTableEncoder, TableEncoding, clean_cell, mean_pool_span, validate_table


def _serialize_cell(header: str, value: object) -> str:
    """Arbitrary "header : value" flattening -- see module docstring."""
    return f"{clean_cell(header)} : {clean_cell(value)}"


class BertTableEncoder(BaseTableEncoder):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        super().__init__(model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length
        self.hidden_size = self.backbone.config.hidden_size

        # Frozen feature extractor, per-professor's guidance: use BERT's
        # last-layer output as a fixed input feature, never backprop
        # through BERT itself. Matches CellEncoder's own default
        # (text_trainable=False) for a fair, consistent comparison, and
        # avoids retaining BERT's activation graph for backward -- the
        # single biggest memory cost in these baselines (see strubert.py's
        # OOM). Note: plain BERT has no separate on-top layer of its own
        # (see adapter.py::_NUM_LAYERS_KWARG's comment -- "the pretrained
        # backbone IS the whole model" for this baseline), so freezing it
        # leaves this encoder itself with zero trainable parameters; the
        # trainable parts of pretraining/finetuning for this baseline are
        # entirely the shared DiscriminatorHead / QueryEncoder / MultiScorer
        # that sit on top of it, not anything inside this file.
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        """Keep the frozen backbone permanently in eval mode (no dropout)
        even though Trainer.fit() calls model.train() every epoch, which
        would otherwise recursively flip it back to train mode -- the
        no_grad() wrapper around its forward call already prevents any
        gradient flow regardless of mode, but eval mode also keeps its
        output features deterministic/stable across repeated calls on the
        same input, which matters for caching/consistency."""
        super().train(mode)
        self.backbone.eval()
        return self

    def _build_sequence(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str],
    ) -> Tuple[List[int], List[List[List[Tuple[int, int]]]]]:
        """Flatten table row-major into a single token id sequence, tracking
        the [start, end) token span of every cell for later mean-pooling.
        Truncates whole cells (never splits a cell) once max_length is hit.

        Tokenizes every cell (and the caption) in ONE batched tokenizer
        call up front, instead of one Python-level self.tokenizer.encode()
        call per cell -- with max_rows=50/max_columns=20 that was up to
        1,000 individual (unbatched) tokenizer calls per table, which adds
        up fast at real batch sizes (e.g. 192 tables/step with
        n_hard_negatives=2 at batch_size=64). A single batched call lets
        the underlying fast (Rust) tokenizer parallelize internally;
        the truncation/assembly logic below is unchanged, just consuming
        already-tokenized id lists instead of tokenizing as it goes.
        """
        n_rows, n_cols = len(rows), len(headers)
        texts: List[str] = []
        if caption:
            texts.append(caption)  # raw, matching the original (unbatched) code's behavior exactly
        for i in range(n_rows):
            for j in range(n_cols):
                texts.append(_serialize_cell(headers[j], rows[i][j]))

        all_ids = self.tokenizer(texts, add_special_tokens=False)["input_ids"] if texts else []

        input_ids: List[int] = [self.tokenizer.cls_token_id]
        cell_spans: List[List[Tuple[int, int]]] = [
            [(0, 0) for _ in headers] for _ in rows
        ]

        idx = 0
        if caption:
            input_ids += all_ids[idx]
            idx += 1
            input_ids.append(self.tokenizer.sep_token_id)

        truncated = False
        for i in range(n_rows):
            if truncated:
                break
            for j in range(n_cols):
                ids = all_ids[idx]
                idx += 1
                if len(input_ids) + len(ids) + 1 >= self.max_length:
                    truncated = True
                    break
                start = len(input_ids)
                input_ids += ids
                end = len(input_ids)
                cell_spans[i][j] = (start, end)
                input_ids.append(self.tokenizer.sep_token_id)

        if input_ids[-1] != self.tokenizer.sep_token_id:
            input_ids.append(self.tokenizer.sep_token_id)
        return input_ids, cell_spans

    def forward(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str] = None,
    ) -> TableEncoding:
        validate_table(headers, rows)
        n_rows, n_cols = len(rows), len(headers)

        input_ids, cell_spans = self._build_sequence(headers, rows, caption)
        ids_t = torch.tensor([input_ids], device=self.device)
        attn_mask = torch.ones_like(ids_t)

        with torch.no_grad():
            out = self.backbone(input_ids=ids_t, attention_mask=attn_mask)
        hidden = out.last_hidden_state[0]  # [seq_len, dim]
        cls = hidden[0]

        cell_emb = torch.zeros(n_rows, n_cols, self.hidden_size, device=self.device)
        for i in range(n_rows):
            for j in range(n_cols):
                start, end = cell_spans[i][j]
                cell_emb[i, j] = mean_pool_span(hidden, start, end) if end > start else cls

        row_emb = cell_emb.mean(dim=1)
        col_emb = cell_emb.mean(dim=0)
        table_emb = cls

        return TableEncoding(cell_emb, row_emb, col_emb, table_emb)
