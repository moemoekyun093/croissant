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

        # col/row-sequence text -> (fine, coarse) frozen-BERT pooled
        # vectors (CPU) -- same pattern as tabbie.py's _cell_cache /
        # cell_encoder.py's TextEmbedder. backbone is frozen, so a given
        # sequence string always produces the same (fine, coarse) pair;
        # the SAME column/row sequence text recurs heavily (a table's
        # own columns/rows re-seen every epoch, as a hard negative for
        # many queries, etc). Only the frozen backbone call is cached --
        # vertical_attn/horizontal_attn/fuse_proj on top always run
        # fresh, since their output changes every training step.
        self._seq_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

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

        Cache-aware, same pattern as tabbie.py's _encode_cells_isolated:
        backbone is frozen, so a given sequence string always produces the
        same (fine, coarse) pair -- only strings never seen before in this
        process actually get tokenized and run through the backbone.
        """
        uncached_unique = list({t for t in seq_texts if t not in self._seq_cache})

        if uncached_unique:
            enc = self.tokenizer(
                uncached_unique,
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
            new_fine = (hidden * fine_mask).sum(dim=1) / fine_mask.sum(dim=1).clamp(min=1.0)
            new_coarse = hidden[:, 0, :]
            # Move both [N, D] blocks GPU->CPU in ONE transfer each, then
            # index -- instead of `for ...: f.detach().cpu(), c.detach().cpu()`,
            # which fired TWO tiny (synchronizing) device-to-host copies
            # PER sequence string. .clone() copies each row out so entries
            # stand alone and the blocks can be freed (cheap CPU-side).
            new_fine_cpu = new_fine.detach().cpu()  # [N, D], one D2H copy
            new_coarse_cpu = new_coarse.detach().cpu()  # [N, D], one D2H copy
            for i, t in enumerate(uncached_unique):
                self._seq_cache[t] = (new_fine_cpu[i].clone(), new_coarse_cpu[i].clone())

        fine = torch.stack([self._seq_cache[t][0] for t in seq_texts], dim=0).to(self.device)
        coarse = torch.stack([self._seq_cache[t][1] for t in seq_texts], dim=0).to(self.device)
        return fine, coarse

    def save_frozen_cache(self, path: str) -> None:
        """Persists self._seq_cache (col/row-sequence text -> frozen BERT
        (fine, coarse) pooled vectors) to disk -- same torch.save pattern
        as tabbie.py's save_frozen_cache / adapter.py's
        save_table_cache. Only the frozen backbone sub-step is cached --
        vertical_attn/horizontal_attn/fuse_proj always run fresh, since
        their output changes every training step and can't be cached the
        way bert/tapas's FULL table embedding can."""
        torch.save(self._seq_cache, path)

    def load_frozen_cache(self, path: str, merge: bool = True) -> None:
        """Loads a previously-saved col/row-sequence cache. merge=True
        keeps existing in-memory entries on a key collision (see
        adapter.py's load_table_cache for why plain dict.update() would
        get this backwards); merge=False replaces the cache entirely."""
        loaded = torch.load(path, map_location="cpu")
        if merge:
            for k, v in loaded.items():
                self._seq_cache.setdefault(k, v)
        else:
            self._seq_cache = loaded

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

    def _forward_from_pooled_batch(
        self,
        col_fine_l: List[torch.Tensor],
        col_coarse_l: List[torch.Tensor],
        row_fine_l: List[torch.Tensor],
        row_coarse_l: List[torch.Tensor],
    ) -> List[TableEncoding]:
        """Batched equivalent of calling _forward_from_pooled once per table.

        The trainable vertical/horizontal self-attention stacks used to run
        once PER TABLE ([1, n_cols, D] and [1, n_rows, D] each) -- with
        diverse table shapes that's ~2 tiny, GPU-underutilized attention
        launches per table per step (the `network on top` profiler cost).
        Here every table's column embeddings are padded to the batch-max
        n_cols and run through vertical_attn ONCE for all tables (with a
        src_key_padding_mask so each table attends only over its own real
        columns), and likewise all rows through horizontal_attn.

        Numerically identical to _forward_from_pooled on every real
        position: these are plain (positionless) self-attention stacks, so
        masked padded keys get exactly-zero attention weight and never
        affect a real column's/row's refined vector; the mean-pools below
        are taken over each table's real n_cols/n_rows only. coarse vectors
        never enter attention (only their mean is used), so they stay
        per-table.
        """
        T = len(col_fine_l)
        D = self.hidden_size
        dev = self.device
        n_cols_l = [t.shape[0] for t in col_fine_l]
        n_rows_l = [t.shape[0] for t in row_fine_l]
        maxC = max(n_cols_l)
        maxR = max(n_rows_l)

        col_pad = torch.zeros(T, maxC, D, device=dev)
        row_pad = torch.zeros(T, maxR, D, device=dev)
        col_key_pad = torch.ones(T, maxC, dtype=torch.bool, device=dev)  # True == ignore
        row_key_pad = torch.ones(T, maxR, dtype=torch.bool, device=dev)
        for i in range(T):
            nc, nr = n_cols_l[i], n_rows_l[i]
            col_pad[i, :nc] = col_fine_l[i]
            row_pad[i, :nr] = row_fine_l[i]
            col_key_pad[i, :nc] = False
            row_key_pad[i, :nr] = False

        col_refined_all = self.vertical_attn(col_pad, src_key_padding_mask=col_key_pad)  # [T, maxC, D]
        row_refined_all = self.horizontal_attn(row_pad, src_key_padding_mask=row_key_pad)  # [T, maxR, D]

        results: List[TableEncoding] = []
        for i in range(T):
            nc, nr = n_cols_l[i], n_rows_l[i]
            col_refined = col_refined_all[i, :nc]  # [n_cols, D]
            row_refined = row_refined_all[i, :nr]  # [n_rows, D]
            cell_emb = (row_refined.unsqueeze(1) + col_refined.unsqueeze(0)) / 2.0  # [R, C, D]
            four_features = torch.cat(
                [
                    row_refined.mean(0),
                    col_refined.mean(0),
                    row_coarse_l[i].mean(0),
                    col_coarse_l[i].mean(0),
                ],
                dim=-1,
            )
            table_emb = self.fuse_proj(four_features)
            results.append(TableEncoding(cell_emb, row_refined, col_refined, table_emb))
        return results

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
        # Timing split for profiling (see tabbie.py's forward_batch for
        # the full rationale -- same pattern): _last_frozen_s is the
        # frozen BERT backbone pass (cacheable), _last_network_s is
        # vertical_attn/horizontal_attn/fuse_proj on top (never
        # cacheable -- trainable).
        import time
        is_cuda = self.device.type == "cuda" if hasattr(self.device, "type") else False
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        all_col_fine, all_col_coarse = self._cellwise_pool_sequence(all_col_seqs)
        all_row_fine, all_row_coarse = self._cellwise_pool_sequence(all_row_seqs)

        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Trainable stack (vertical/horizontal attn + fuse), batched across
        # ALL tables in one call per attention stack instead of the old
        # per-table loop -- see _forward_from_pooled_batch.
        results = self._forward_from_pooled_batch(
            [all_col_fine[c0:c1] for (c0, c1) in col_offsets],
            [all_col_coarse[c0:c1] for (c0, c1) in col_offsets],
            [all_row_fine[r0:r1] for (r0, r1) in row_offsets],
            [all_row_coarse[r0:r1] for (r0, r1) in row_offsets],
        )

        if is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        self._last_frozen_s = t1 - t0
        self._last_network_s = t2 - t1

        return results
