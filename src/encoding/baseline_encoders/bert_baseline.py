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

from .common import BaseTableEncoder, TableEncoding, clean_cell, validate_table


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

    def _pool_cells(
        self,
        hidden: torch.Tensor,
        cell_spans: List[List[Tuple[int, int]]],
        n_rows: int,
        n_cols: int,
    ) -> torch.Tensor:
        """Vectorized mean-pool over every cell's token span at once via a
        prefix-sum trick, instead of calling mean_pool_span (one small
        GPU op each) in a Python double loop -- with up to 1,000 cells
        per table and up to ~192 tables/step (batch_size=64 plus
        n_hard_negatives=2), that loop was up to ~192,000 individual
        tiny GPU calls per training step, each paying its own
        kernel-launch overhead. csum[b] = sum of hidden[0:b], so a
        span's sum is csum[end] - csum[start] for every cell in one
        shot; truncated/empty spans (end <= start) fall back to cls,
        same as the original per-cell logic.

        Shared by forward() (single table, hidden = that table's own
        [seq_len, dim]) and forward_batch() (hidden = one table's own
        SLICE out of a batched [B, maxlen, dim] backbone output, real
        tokens only -- see forward_batch's docstring).
        """
        hidden = hidden.to(hidden.dtype)
        cls = hidden[0]

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
        lengths = (ends - starts).clamp(min=1).unsqueeze(-1).to(hidden.dtype)  # [n_rows, n_cols, 1]
        span_sums = csum[ends] - csum[starts]  # [n_rows, n_cols, dim]
        means = span_sums / lengths

        empty_mask = (ends <= starts).unsqueeze(-1)  # [n_rows, n_cols, 1]
        cls_broadcast = cls.view(1, 1, -1).expand(n_rows, n_cols, -1)
        return torch.where(empty_mask, cls_broadcast, means)

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

        cell_emb = self._pool_cells(hidden, cell_spans, n_rows, n_cols)
        row_emb = cell_emb.mean(dim=1)
        col_emb = cell_emb.mean(dim=0)
        table_emb = hidden[0]

        return TableEncoding(cell_emb, row_emb, col_emb, table_emb)

    def forward_batch(
        self,
        tables: List[Tuple[Sequence[str], Sequence[Sequence[object]], Optional[str]]],
    ) -> List[TableEncoding]:
        """Batched version of forward(): encodes MULTIPLE tables in ONE
        BertModel forward pass instead of one pass per table.

        This is the real reason 'ours' (src/encoding/cell_encoder.py)
        was faster than this baseline even after every earlier
        vectorization/caching fix this session: CellEncoder always
        batches every cell across the WHOLE batch of tables into a
        single BERT call. This baseline's adapter
        (BaselineCellwiseAdapter.forward_batch_cellwise) used to call
        forward() once PER TABLE -- e.g. 192 separate batch-of-1
        BertModel forward passes for a 192-table training step, instead
        of one batch-of-192 pass. GPUs run one large batched matmul far
        more efficiently than many tiny sequential ones (each paying its
        own kernel-launch overhead, leaving most of the GPU idle).
        BaselineCellwiseAdapter checks for this method (hasattr) and
        uses it instead of looping forward() calls when present -- see
        adapter.py's forward_batch_cellwise.

        tables: list of (headers, rows, caption) tuples, one per table
                (caption is always None on the current adapter call
                path, same as forward()'s default -- kept as a real
                parameter here for API parity with forward() regardless).
        returns: list of TableEncoding, one per table, SAME ORDER as
                 the input list.
        """
        if not tables:
            return []

        per_table_ids: List[List[int]] = []
        per_table_spans: List[List[List[Tuple[int, int]]]] = []
        per_table_shape: List[Tuple[int, int]] = []

        for headers, rows, caption in tables:
            validate_table(headers, rows)
            input_ids, cell_spans = self._build_sequence(headers, rows, caption)
            per_table_ids.append(input_ids)
            per_table_spans.append(cell_spans)
            per_table_shape.append((len(rows), len(headers)))

        B = len(tables)
        maxlen = max(len(ids) for ids in per_table_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:  # some tokenizers have no pad token configured; 0 is BERT's own [PAD] id
            pad_id = 0

        ids_t = torch.full((B, maxlen), pad_id, dtype=torch.long, device=self.device)
        attn_mask = torch.zeros((B, maxlen), dtype=torch.long, device=self.device)
        for i, ids in enumerate(per_table_ids):
            ids_t[i, : len(ids)] = torch.tensor(ids, device=self.device)
            attn_mask[i, : len(ids)] = 1

        # Defensive checks BEFORE the backbone call. An out-of-range
        # token id, or a sequence longer than this backbone's
        # max_position_embeddings, would otherwise surface as an opaque
        # CUDA "index out of bounds" device-side assert deep inside an
        # embedding lookup -- and CRITICALLY, get reported ASYNCHRONOUSLY
        # at some unrelated LATER call (observed in practice: the crash
        # this was added for showed up several lines later, inside
        # trainer.py's _corpus_scores, at an unrelated .to(device) sync
        # point, with a traceback pointing at the wrong line entirely).
        # Catching it here, on values we just built ourselves, gives an
        # immediate, correctly-attributed Python error instead of an
        # unattributable device-side assert. Both checks are vectorized
        # (two whole-tensor comparisons) -- only the list comprehension
        # building the error message itself is a Python loop, and that
        # only runs on the failure path.
        vocab_size = getattr(self.backbone.config, "vocab_size", None)
        if vocab_size is not None and bool((ids_t.ge(vocab_size) | ids_t.lt(0)).any()):
            bad = [
                (i, tid)
                for i, ids in enumerate(per_table_ids)
                for tid in ids
                if tid < 0 or tid >= vocab_size
            ]
            raise ValueError(
                f"forward_batch: token id(s) outside this backbone's vocab_size "
                f"({vocab_size}) -- would have crashed as an unattributed CUDA "
                f"device-side assert inside the embedding lookup instead. "
                f"offending (table_index_in_this_call, token_id): {bad[:5]}"
            )

        max_pos = getattr(self.backbone.config, "max_position_embeddings", None)
        if max_pos is not None and maxlen > max_pos:
            offending = [(i, len(ids)) for i, ids in enumerate(per_table_ids) if len(ids) > max_pos]
            raise ValueError(
                f"forward_batch: batch's max sequence length ({maxlen}) exceeds "
                f"this backbone's max_position_embeddings ({max_pos}) -- "
                f"self.max_length={self.max_length} should prevent this via "
                f"_build_sequence's truncation, so either self.max_length is "
                f"configured above max_pos, or there's an edge case letting a "
                f"sequence through un-truncated. offending "
                f"(table_index_in_this_call, seq_len): {offending[:5]}"
            )

        with torch.no_grad():
            out = self.backbone(input_ids=ids_t, attention_mask=attn_mask)
        hidden_batch = out.last_hidden_state  # [B, maxlen, dim]

        # Per-table pooling loop below is O(B) (tens to low hundreds of
        # tables), not O(cells) -- the expensive part (the BertModel
        # call itself) already happened once, batched, above. Each
        # table's own real (non-padding) token span is sliced out before
        # pooling so padding positions never contribute to its cell
        # embeddings, even though they were attention-masked out of the
        # backbone call already too (belt and suspenders -- padding
        # tokens' hidden states are typically near-zero/meaningless
        # under a correct attention mask, but never actually indexed
        # into cell_spans regardless, since every span was built from
        # that table's own un-padded input_ids length).
        results: List[TableEncoding] = []
        for i in range(B):
            seq_len = len(per_table_ids[i])
            hidden = hidden_batch[i, :seq_len]  # [seq_len_i, dim] -- real tokens only
            n_rows, n_cols = per_table_shape[i]
            cell_emb = self._pool_cells(hidden, per_table_spans[i], n_rows, n_cols)
            row_emb = cell_emb.mean(dim=1)
            col_emb = cell_emb.mean(dim=0)
            table_emb = hidden[0]
            results.append(TableEncoding(cell_emb, row_emb, col_emb, table_emb))

        return results
