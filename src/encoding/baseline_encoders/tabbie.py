"""
Baseline #2 — TABBIE (Iida et al., 2021, "TABBIE: Pretrained Representations
of Tabular Data").

Reference implementation: https://github.com/SFIG611/tabbie (old
`allennlp` + `pytorch-pretrained-bert` codebase). Re-implemented here against
modern `transformers`, keeping the architecture described in the paper.

Header handling (confirmed against the reference repo, not guessed): TABBIE
does **not** fold the header into each cell's text. `preprocess.py` reads
`header, cells = table[0], table[1:]` and indexes the header row completely
separately from the data cells (`indexed_headers` vs `indexed_cells`).
`finetune_table.py::get_tabemb` then concatenates the header embeddings back
in as **row 0** of the table before running the row/column transformer
(`prob_tables_cls[:, 1, 1:, :]` is literally "the header row", `[:, 2:, 1:, :]`
is "the data rows", both indexed the same way after a shared CLS-row/
CLS-col prepend). So the header row gets its own row-CLS embedding and
participates in every column's column-attention exactly like a data row --
it is never mixed into a data cell's own tokens.

We reproduce that here: build an (n_rows + 1) x n_cols grid where row 0 is
the header, run the identical row/column transformer over it, then split
the header row's contextualized row-CLS embedding back out at the end.

Architecture (Sec 3.1-3.2 of the paper):
  1. Each cell (including header cells) is encoded *independently* with
     BERT; the [CLS] output is the initial, uncontextualized cell
     embedding x_ij.
  2. Learned row/column positional embeddings p_i^(r), p_j^(c) are added.
  3. For `num_layers` rounds:
       - a *row* Transformer contextualizes cells within each row
         (a CLS_ROW token is prepended to obtain the row embedding),
       - a *column* Transformer contextualizes cells within each column
         (a CLS_COL token is prepended to obtain the column embedding),
       - the new cell embedding is the average of the row- and
         column-contextualized versions, fed to the next round.
  4. Table embedding = mean of all final row/column CLS embeddings
     (header row's CLS included, since it's a real row here too).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .common import BaseTableEncoder, TableEncoding, clean_cell, validate_table
from src.encoding.cache_utils import downcast_cache, upcast_cache


class TabbieTableEncoder(BaseTableEncoder):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_layers: int = 2,
        num_heads: int = 8,
        max_rows: int = 129,  # +1 to leave room for the header row
        max_cols: int = 64,
        cell_max_length: int = 32,
        device: Optional[str] = None,
    ):
        super().__init__(model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cell_encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.hidden_size = self.cell_encoder.config.hidden_size
        self.cell_max_length = cell_max_length

        # Frozen feature extractor -- see bert_baseline.py's comment for
        # the full rationale. Every cell in the table goes through this
        # backbone independently as one batched call (n_rows*n_cols
        # sequences per table), so this is the single most memory-hungry
        # backbone call in the whole baseline suite; freezing it removes
        # that entire activation graph from backward, leaving only the
        # row/column transformer layers below (self.row_layers/col_layers)
        # trainable, same "train only the layers on top" pattern.
        self.cell_encoder.eval()
        for p in self.cell_encoder.parameters():
            p.requires_grad = False

        # cell text -> raw [CLS] vector (CPU), same pattern as
        # TextEmbedder in cell_encoder.py -- cell_encoder is frozen, so a
        # given cell string always produces the same output, and cells
        # repeat heavily across a real corpus (dates, ids, categorical
        # values, and the same table's cells re-seen every epoch or as a
        # hard negative for many queries). See _encode_cells_isolated.
        self._cell_cache: dict[str, torch.Tensor] = {}

        self.row_pos_embed = nn.Embedding(max_rows, self.hidden_size).to(self.device)
        self.col_pos_embed = nn.Embedding(max_cols, self.hidden_size).to(self.device)
        self.cls_row = nn.Parameter(torch.randn(self.hidden_size) * 0.02).to(self.device)
        self.cls_col = nn.Parameter(torch.randn(self.hidden_size) * 0.02).to(self.device)

        def make_layer():
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=num_heads,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True,
            )
            return enc_layer.to(self.device)

        self.row_layers = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.col_layers = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.num_layers = num_layers

    def train(self, mode: bool = True):
        """Keep the frozen cell_encoder permanently in eval mode -- see
        bert_baseline.py's train() override for the full rationale."""
        super().train(mode)
        self.cell_encoder.eval()
        return self

    def _encode_cells_isolated(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode a flat list of cell texts independently with BERT (batched)
        and return the [CLS] vector as x for each -- matches the reference
        repo's `bert_embedder`, which never lets one cell's tokens attend to
        another cell's tokens at this stage.

        Cache-aware: only strings never seen before in this process
        actually get tokenized and run through cell_encoder -- see
        self._cell_cache's docstring in __init__. Duplicates (including
        repeats within this same call, e.g. a common header text
        appearing once per table) are gathered from the single computed
        result, same as TextEmbedder.forward's approach.
        """
        uncached_unique = list({t for t in texts if t not in self._cell_cache})

        if uncached_unique:
            enc = self.tokenizer(
                uncached_unique,
                padding=True,
                truncation=True,
                max_length=self.cell_max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.cell_encoder(**enc)
            # Move the WHOLE [N, D] block of new [CLS] vectors GPU->CPU in
            # ONE transfer, then file each row into the cache from CPU --
            # instead of `for ...: v.detach().cpu()`, which issued one
            # separate (tiny, synchronizing) device-to-host copy PER cell
            # (N of them every time new cells are seen, right on the hot
            # cache-warming path). new_cls[i] is a view into the moved
            # block; .clone() copies out just that one vector so each cache
            # entry stands alone and the big block can be freed (cheap
            # CPU-side copy, no transfer).
            new_cls = out.last_hidden_state[:, 0, :].detach().cpu()  # [N, D], one D2H copy
            for i, t in enumerate(uncached_unique):
                self._cell_cache[t] = new_cls[i].clone()

        return torch.stack([self._cell_cache[t] for t in texts], dim=0).to(self.device)

    def save_frozen_cache(self, path: str) -> None:
        """Persists self._cell_cache (cell/header text -> frozen BERT
        [CLS] vector) to disk -- same torch.save pattern as
        TextEmbedder.save_cache_to_disk (cell_encoder.py) and
        BaselineCellwiseAdapter.save_table_cache (adapter.py). Unlike
        bert/tapas's per-table cache, this is per-STRING and only covers
        the frozen sub-step (_encode_cells_isolated) -- the trainable
        row/column transformer stack on top (self.row_layers/col_layers,
        _forward_from_cls) always runs fresh regardless of a cache hit
        here, since ITS output changes every training step and can't be
        cached. See adapter.py's cacheable docstring for why tabbie's
        FULL table embedding (unlike bert/tapas's) can't be persisted
        the same way."""
        torch.save(downcast_cache(self._cell_cache), path)

    def load_frozen_cache(self, path: str, merge: bool = True) -> None:
        """Loads a previously-saved cell/header text cache. merge=True
        keeps existing in-memory entries on a key collision (see
        adapter.py's load_table_cache for why plain dict.update() would
        get this backwards); merge=False replaces the cache entirely."""
        loaded = upcast_cache(torch.load(path, map_location="cpu"))
        if merge:
            for k, v in loaded.items():
                self._cell_cache.setdefault(k, v)
        else:
            self._cell_cache = loaded

    def forward(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str] = None,
    ) -> TableEncoding:
        validate_table(headers, rows)
        n_rows, n_cols = len(rows), len(headers)

        # row 0 = header cells (their own text, NOT "header : value"); rows 1.. = data cells
        flat_texts = [clean_cell(h) for h in headers]
        for i in range(n_rows):
            flat_texts += [clean_cell(rows[i][j]) for j in range(n_cols)]

        cls = self._encode_cells_isolated(flat_texts)  # [(1+n_rows)*n_cols, D]
        return self._forward_from_cls(cls, n_rows, n_cols)

    def forward_batch(
        self,
        tables: Sequence[Tuple[Sequence[str], Sequence[Sequence[object]], Optional[str]]],
    ) -> List[TableEncoding]:
        """Batched version of forward(): runs ONE frozen-BERT cell-encoder
        pass for MULTIPLE tables' cells/headers combined, instead of one
        pass per table -- same fix, same rationale as bert_baseline.py's
        forward_batch (see its docstring for the full "why ours was
        faster" explanation). See adapter.py's forward_batch_cellwise for
        how this gets picked up automatically (hasattr check).

        Unlike bert_baseline.py/tapas_encoder.py, this doesn't need any
        new padding/stacking logic: _encode_cells_isolated already
        accepts an arbitrary flat list of texts and batches internally
        (with its own cross-string cache) -- the only change is
        concatenating every table's flat_texts together BEFORE calling
        it once, instead of calling it once per table the way forward()
        (looped by the adapter) used to. The row/column transformer
        stack below (self.row_layers/col_layers) is small and already
        cheap relative to the frozen BERT cell encoder, so it stays a
        per-table loop -- this only batches the expensive part.

        tables: list of (headers, rows, caption) tuples, one per table
                (caption unused -- TABBIE has no caption handling on the
                current adapter call path, same as forward()'s own
                caption parameter).
        returns: list of TableEncoding, one per table, SAME ORDER as the
                 input list.
        """
        if not tables:
            return []

        per_table_offsets: List[Tuple[int, int]] = []
        per_table_shape: List[Tuple[int, int]] = []
        all_texts: List[str] = []

        offset = 0
        for headers, rows, _caption in tables:
            validate_table(headers, rows)
            n_rows, n_cols = len(rows), len(headers)
            flat_texts = [clean_cell(h) for h in headers]
            for i in range(n_rows):
                flat_texts += [clean_cell(rows[i][j]) for j in range(n_cols)]
            all_texts.extend(flat_texts)
            per_table_offsets.append((offset, offset + len(flat_texts)))
            per_table_shape.append((n_rows, n_cols))
            offset += len(flat_texts)

        # Timing split for profiling (see adapter.py/trainer.py's
        # _score_batch): _last_frozen_s is the frozen BERT cell-encoding
        # pass (cacheable -- 0 on a full cache hit run), _last_network_s
        # is the row/col transformer stack on top (never cacheable, it's
        # what's actually being trained). torch.cuda.synchronize() at the
        # boundary is needed for an accurate split -- otherwise the async
        # GPU queue lets the frozen pass's real cost bleed into the
        # network measurement.
        import time
        is_cuda = self.device.type == "cuda" if hasattr(self.device, "type") else False
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        all_cls = self._encode_cells_isolated(all_texts)  # [total_cells_across_all_tables, D]

        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Trainable row/column transformer stack, batched across the whole
        # table batch instead of looped one table at a time. This used to
        # be `for ...: results.append(self._forward_from_cls(cls, ...))` --
        # one row-transformer + one col-transformer forward PER TABLE, per
        # layer, i.e. ~2 * num_layers * len(tables) separate batch-of-a-few
        # TransformerEncoderLayer launches every step (the "network on top"
        # line in trainer.py's profiler -- ~45% of step time and almost
        # entirely GPU underutilization: hundreds of tiny kernels with the
        # GPU idle between them, not real compute). _forward_from_cls_batch
        # groups same-shape tables and runs each group as ONE real batch,
        # collapsing those hundreds of tiny launches into a handful of full
        # ones. Numerically identical to the old per-table loop (a
        # TransformerEncoderLayer applies independently per sequence, so
        # stacking tables in the batch dim changes nothing but throughput).
        cls_list = [all_cls[start:end] for (start, end) in per_table_offsets]
        results: List[TableEncoding] = self._forward_from_cls_batch(cls_list, per_table_shape)

        if is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        self._last_frozen_s = t1 - t0
        self._last_network_s = t2 - t1

        return results

    def _forward_from_cls(self, cls: torch.Tensor, n_rows: int, n_cols: int) -> TableEncoding:
        """Everything AFTER cell/header encoding: row/col positional
        embeddings + the row/column transformer stack. Shared by
        forward() (cls from a single table's own _encode_cells_isolated
        call) and forward_batch() (cls sliced out of a combined
        multi-table call) -- identical computation either way, this
        stack only ever sees one table's own cls vectors at a time."""
        n_rows_total = n_rows + 1  # +1 for the header row
        x = cls.view(n_rows_total, n_cols, self.hidden_size)

        row_ids = torch.arange(min(n_rows_total, self.row_pos_embed.num_embeddings), device=self.device)
        col_ids = torch.arange(min(n_cols, self.col_pos_embed.num_embeddings), device=self.device)
        row_pos = self.row_pos_embed(row_ids.clamp(max=self.row_pos_embed.num_embeddings - 1))
        col_pos = self.col_pos_embed(col_ids.clamp(max=self.col_pos_embed.num_embeddings - 1))
        x = x + row_pos.unsqueeze(1) + col_pos.unsqueeze(0)

        row_cls_final = None
        col_cls_final = None
        for layer_idx in range(self.num_layers):
            # --- row transformer: batch = rows (incl. header row), seq = [CLS_ROW, cell_0..cell_{C-1}]
            row_cls_tok = self.cls_row.view(1, 1, -1).expand(n_rows_total, 1, -1)
            row_input = torch.cat([row_cls_tok, x], dim=1)
            row_out = self.row_layers[layer_idx](row_input)
            row_cls_final = row_out[:, 0, :]  # [1+n_rows, D]
            row_ctx_cells = row_out[:, 1:, :]

            # --- column transformer: batch = cols, seq = [CLS_COL, cell_0..cell_{R-1}] (incl. header cell)
            x_by_col = x.transpose(0, 1)  # [C, 1+R, D]
            col_cls_tok = self.cls_col.view(1, 1, -1).expand(n_cols, 1, -1)
            col_input = torch.cat([col_cls_tok, x_by_col], dim=1)
            col_out = self.col_layers[layer_idx](col_input)
            col_cls_final = col_out[:, 0, :]  # [C, D]
            col_ctx_cells = col_out[:, 1:, :].transpose(0, 1)

            x = (row_ctx_cells + col_ctx_cells) / 2.0

        # split header row (index 0) back out from the data rows (indices 1..)
        cell_emb = x[1:]  # [n_rows, n_cols, D] -- matches TableEncoding contract
        row_emb = row_cls_final[1:]  # [n_rows, D], header row excluded
        col_emb = col_cls_final  # [n_cols, D] -- already reflects header influence via col attention
        table_emb = torch.cat([row_cls_final, col_cls_final], dim=0).mean(dim=0)

        return TableEncoding(cell_emb, row_emb, col_emb, table_emb)

    def _forward_from_cls_batch(
        self, cls_list: List[torch.Tensor], shapes: List[Tuple[int, int]]
    ) -> List[TableEncoding]:
        """Batched equivalent of calling _forward_from_cls once per table.

        cls_list[i] is table i's flat cell/header [CLS] stack
        ([(1+n_rows_i)*n_cols_i, D]); shapes[i] is its (n_rows_i,
        n_cols_i). Returns one TableEncoding per input, in the SAME order.

        FULLY batched across the whole table batch: every table is padded
        to the batch-max (n_rows_total, n_cols) grid and the row/column
        transformers run ONCE per layer for ALL tables together, with a
        src_key_padding_mask so each table attends only over its own real
        rows/cols. This replaces both the old per-table Python loop AND the
        earlier group-by-exact-shape version -- with diverse table shapes,
        exact-shape grouping degrades to ~one tiny transformer call per
        table (hundreds of underutilized GPU launches per step, the
        `network on top` profiler cost). Padding to a common shape lets all
        of them ride one big, well-utilized call instead.

        Numerically identical to _forward_from_cls on every real position:
        a TransformerEncoderLayer applies independently per sequence, and
        masked keys get exactly-zero attention weight, so a padded column
        or row never contributes to any real cell's / CLS's output;
        LayerNorm and the FFN are per-position, so padded positions
        (discarded at the end) can't perturb real ones. Positional-embedding
        lookup/clamp, CLS-token prepend, row/col averaging, and the final
        row/col-CLS mean pooling are the same expressions as
        _forward_from_cls; see it for the architecture rationale.
        """
        D = self.hidden_size
        dev = self.device
        T = len(cls_list)

        n_rows_total = [r + 1 for (r, _c) in shapes]  # +1 header row, matches _forward_from_cls
        n_cols_l = [c for (_r, c) in shapes]
        maxR = max(n_rows_total)
        maxC = max(n_cols_l)

        # Padded cell grid [T, maxR, maxC, D]; each table placed at [i, :nrt, :nc].
        x = torch.zeros(T, maxR, maxC, D, device=dev)
        for i in range(T):
            nrt, nc = n_rows_total[i], n_cols_l[i]
            x[i, :nrt, :nc] = cls_list[i].view(nrt, nc, D)

        # Positional embeddings added to every position (padded ones are
        # masked as keys and discarded as outputs, so they don't matter).
        # Same clamp as _forward_from_cls -- a no-op while shapes stay
        # within max_rows/max_cols, kept for parity.
        row_ids = torch.arange(maxR, device=dev).clamp(max=self.row_pos_embed.num_embeddings - 1)
        col_ids = torch.arange(maxC, device=dev).clamp(max=self.col_pos_embed.num_embeddings - 1)
        row_pos = self.row_pos_embed(row_ids)  # [maxR, D]
        col_pos = self.col_pos_embed(col_ids)  # [maxC, D]
        x = x + row_pos.view(1, maxR, 1, D) + col_pos.view(1, 1, maxC, D)

        # Per-table validity, and a [T, maxR, maxC, 1] gate to keep padded
        # cells at exactly zero between layers (defensive: masked keys with
        # a non-finite value could otherwise poison a real output via
        # 0*inf; zeroing guarantees they stay finite).
        rows_valid = torch.zeros(T, maxR, dtype=torch.bool, device=dev)
        cols_valid = torch.zeros(T, maxC, dtype=torch.bool, device=dev)
        for i in range(T):
            rows_valid[i, : n_rows_total[i]] = True
            cols_valid[i, : n_cols_l[i]] = True
        cell_gate = (rows_valid.unsqueeze(2) & cols_valid.unsqueeze(1)).unsqueeze(-1).float()
        x = x * cell_gate

        # key_padding_mask (True == ignore). CLS token (position 0) is never
        # masked. Row transformer seq = [CLS_ROW, col_0..col_{maxC-1}] ->
        # mask padded columns; same for every row of a table. Column
        # transformer seq = [CLS_COL, row_0..row_{maxR-1}] -> mask padded rows.
        row_key_pad = torch.zeros(T, 1 + maxC, dtype=torch.bool, device=dev)
        row_key_pad[:, 1:] = ~cols_valid
        row_key_pad = row_key_pad.unsqueeze(1).expand(T, maxR, 1 + maxC).reshape(T * maxR, 1 + maxC)

        col_key_pad = torch.zeros(T, 1 + maxR, dtype=torch.bool, device=dev)
        col_key_pad[:, 1:] = ~rows_valid
        col_key_pad = col_key_pad.unsqueeze(1).expand(T, maxC, 1 + maxR).reshape(T * maxC, 1 + maxR)

        row_cls_final = None
        col_cls_final = None
        for layer_idx in range(self.num_layers):
            # --- row transformer: batch = (tables x rows), seq = [CLS_ROW, cells across cols]
            row_input = x.reshape(T * maxR, maxC, D)
            row_cls_tok = self.cls_row.view(1, 1, -1).expand(T * maxR, 1, -1)
            row_input = torch.cat([row_cls_tok, row_input], dim=1)  # [T*maxR, 1+maxC, D]
            row_out = self.row_layers[layer_idx](row_input, src_key_padding_mask=row_key_pad)
            row_cls_final = row_out[:, 0, :].reshape(T, maxR, D)
            row_ctx_cells = row_out[:, 1:, :].reshape(T, maxR, maxC, D)

            # --- column transformer: batch = (tables x cols), seq = [CLS_COL, cells across rows]
            x_by_col = x.transpose(1, 2)  # [T, maxC, maxR, D]
            col_input = x_by_col.reshape(T * maxC, maxR, D)
            col_cls_tok = self.cls_col.view(1, 1, -1).expand(T * maxC, 1, -1)
            col_input = torch.cat([col_cls_tok, col_input], dim=1)  # [T*maxC, 1+maxR, D]
            col_out = self.col_layers[layer_idx](col_input, src_key_padding_mask=col_key_pad)
            col_cls_final = col_out[:, 0, :].reshape(T, maxC, D)
            col_ctx_cells = col_out[:, 1:, :].reshape(T, maxC, maxR, D).transpose(1, 2)  # [T, maxR, maxC, D]

            x = (row_ctx_cells + col_ctx_cells) / 2.0
            x = x * cell_gate  # keep padded cells at zero for the next layer

        results: List[Optional[TableEncoding]] = [None] * T
        for i in range(T):
            nrt, nc = n_rows_total[i], n_cols_l[i]
            rcf = row_cls_final[i, :nrt]  # [1+n_rows, D]
            ccf = col_cls_final[i, :nc]  # [n_cols, D]
            cell_emb = x[i, 1:nrt, :nc]  # [n_rows, n_cols, D]
            row_emb = rcf[1:]  # [n_rows, D], header row excluded
            col_emb = ccf  # [n_cols, D]
            table_emb = torch.cat([rcf, ccf], dim=0).mean(dim=0)
            results[i] = TableEncoding(cell_emb, row_emb, col_emb, table_emb)

        return results  # type: ignore[return-value]
