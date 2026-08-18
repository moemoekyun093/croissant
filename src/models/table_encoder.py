"""
Column/table architecture: turns per-column cell embeddings into
row-resolved cell embeddings [B, N, M, k] (B tables, N=max columns,
M=max rows), consumed directly by the ELECTRA discriminator head
(pretraining) and MultiScorer (query-table finetuning) -- see
forward_batch_cellwise() below, the only forward path in the current
pipeline.

Every module operates on that batch-with-table dimension, using masks
(col_mask: [B,N], row_mask: [B,M]) rather than Python loops over columns
or tables. A single table is just B=1.

Pipeline order:
    (header injection -- done ONCE, before the layer stack, matching how
     token+positional embeddings are added once in a standard transformer,
     not re-added at every layer)
    TableLayer x num_layers  -- each layer:
        ColumnAggregator      -- produces Q, K, V. Each is: a learned
                                channel-mixing matrix (pointwise per row)
                                followed by the original per-channel
                                sigma(gate * x_c x_c^T) x_c formula,
                                unchanged, applied to the mixed
                                representation. Three independent
                                branches -- Q, K, V each get their own
                                channel-mix and their own gates.
        CrossColumnAttention   -- consumes Q, K, V directly (no
                                projection of its own) and compares
                                columns: row-matched, summed jointly over
                                rows and all k channels.
        ChannelMix             -- across the k channels.
    Repeatable, stacked num_layers times, each with its own weights.

Profiling: measured empirically (not assumed) that cell encoding
(BERT + numeric embedder) dominates wall-clock cost by roughly 99%+ of a
forward pass -- the custom table-level layers are a rounding error in
comparison. This means stacking more layers (or widening embed_dim) costs
almost nothing in wall-clock terms; BERT is the actual bottleneck if
speed ever needs addressing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.table import Table

_NONLINEARITIES = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "relu": torch.relu,
}


# ==========================================================
# COLUMN AGGREGATOR (produces Q, K, V for cross-column attention)
# ==========================================================
# Three separate branches (Q, K, V). Each branch: a learned channel-mix
# matrix (pointwise per row -- never touches the row axis), followed by
# your ORIGINAL per-channel formula, unchanged:
#     gate_c = scale_c * mean(x_c) + bias_c
#     A_c    = sigma(gate_c * (x_c x_c^T))
#     out_c  = A_c @ x_c
# applied independently per (now-mixed) channel -- no heads, no further
# joint reduction inside this step. The channel-mixing is what gives each
# branch awareness of all k original channels; the row-attention formula
# itself is exactly as originally specified.
#
# This replaces the earlier design where ColumnAggregator produced a
# single updated X, and CrossColumnAttention separately re-projected that
# X into its own Q/K/V. Now: constructing Q/K/V IS this row-attention
# mechanism, run three times with three independent channel-mixings.
#
# Row-equivariance verified numerically for the full combined mechanism
# (channel-mix -> per-channel row-attention x3 -> cross-column
# comparison): permuting input rows produces an output permuted exactly
# the same way, to floating-point precision. No positional encoding is
# used on the row axis anywhere.
#
# No header injection here -- that happens once, in TableEncoder, before
# the layer stack.

class ColumnAggregator(nn.Module):
    def __init__(self, input_dim: int, nonlinearity: str = "sigmoid", num_heads: int = 8):
        super().__init__()

        self.input_dim = input_dim

        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        if nonlinearity not in _NONLINEARITIES:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

        self.sigma = _NONLINEARITIES[nonlinearity]

        self.input_norm = nn.LayerNorm(input_dim)

        # three independent channel-mixing matrices -- pointwise per row,
        # bias=False so zero-padded rows map to zero and stay zero
        self.W_Q_mix = nn.Linear(input_dim, input_dim, bias=False)
        self.W_K_mix = nn.Linear(input_dim, input_dim, bias=False)
        self.W_V_mix = nn.Linear(input_dim, input_dim, bias=False)

        # three independent sets of per-channel gates -- W(mean(.)) from
        # the original spec, one full set per branch
        self.mean_scale_Q = nn.Parameter(torch.ones(input_dim))
        self.mean_bias_Q = nn.Parameter(torch.zeros(input_dim))
        self.mean_scale_K = nn.Parameter(torch.ones(input_dim))
        self.mean_bias_K = nn.Parameter(torch.zeros(input_dim))
        # value branch: plain channel-mixed projection, NO row-attention
        # (gated-Gram) -- removed. Rows must stay atomic on the value path
        # so cross-column attention combines row-matched cells, not
        # row-blends; also drops a third of the [B,N,M,M,k] Gram cost.

    def _row_attention(
        self,
        Xp: torch.Tensor,
        row_mask: torch.Tensor,
        mean_scale: torch.Tensor,
        mean_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-head row aggregation. The original per-channel formula is
        the H = k special case (head_dim = 1): each head groups d = k/H
        channels and builds ONE M x M gated Gram Z Zᵀ over the head's d
        channels (a d-dimensional inner product between rows), instead of
        one scalar-Gram per channel. That single M x M matrix is shared
        across the head's d channels in the matvec back. This shrinks the
        Gram from [B,N,M,M,k] to [B,N,M,M,H] -- the term that otherwise
        blocks high embed_dim.

        Xp: [B, N, M, k]
        row_mask: [B, M]
        returns: [B, N, M, k]
        """

        B, N, M, k = Xp.shape
        H, d = self.num_heads, self.head_dim

        denom = row_mask.sum(dim=-1).clamp(min=1.0)  # [B]
        channel_means = (
            Xp * row_mask.view(B, 1, M, 1)
        ).sum(dim=2) / denom.view(B, 1, 1)  # [B, N, k]

        # per-channel learnable gate, then one scalar per head (mean over
        # the head's d channels) so it multiplies the head's M x M Gram.
        gate = mean_scale * channel_means + mean_bias  # [B, N, k]
        gate_h = gate.view(B, N, H, d).mean(dim=-1)  # [B, N, H]

        Xph = Xp.reshape(B, N, M, H, d)

        # Z Zᵀ per head: row-row similarity is now a d-dim inner product
        A = torch.einsum("bnihd,bnjhd->bnijh", Xph, Xph)  # [B, N, M, M, H]
        A = self.sigma(A * gate_h.view(B, N, 1, 1, H))

        row_mask_i = row_mask.view(B, 1, M, 1)
        row_mask_j = row_mask.view(B, 1, 1, M)
        pair_mask = row_mask_i * row_mask_j  # [B, 1, M, M]
        A = A * pair_mask.unsqueeze(-1)  # [B,1,M,M,1] broadcast over heads

        out = torch.einsum("bnijh,bnjhd->bnihd", A, Xph)  # [B, N, M, H, d]
        return out.reshape(B, N, M, k)

    def forward(
        self, X: torch.Tensor, row_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        X:        [B, N, M, k]
        row_mask: [B, M]

        returns: (Q, K, V), each [B, N, M, k]
        """

        Xn = self.input_norm(X)

        X_Q = self.W_Q_mix(Xn)
        X_K = self.W_K_mix(Xn)
        X_V = self.W_V_mix(Xn)

        Q = X_Q + self._row_attention(X_Q, row_mask, self.mean_scale_Q, self.mean_bias_Q)
        K = X_K + self._row_attention(X_K, row_mask, self.mean_scale_K, self.mean_bias_K)
        V = X_V  # plain projection -- value net (gated-Gram) removed

        return Q, K, V


# ==========================================================
# CROSS COLUMN ATTENTION (consumes Q, K, V from ColumnAggregator)
# ==========================================================
# No projection of its own anymore -- Q, K, V arrive already constructed
# (channel-mixed + row-attended) from ColumnAggregator. This module's
# only job is the column-to-column comparison: row-matched (row m of
# column i vs row m of column j), summed jointly over rows and all k
# channels -- genuine joint-channel comparison, since Q/K already carry
# mixed-channel information from ColumnAggregator.

class CrossColumnAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 8):
        super().__init__()

        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.norm_q = nn.LayerNorm(input_dim)
        self.norm_k = nn.LayerNorm(input_dim)
        self.norm_v = nn.LayerNorm(input_dim)
        self.W_O = nn.Linear(input_dim, input_dim, bias=False)

    def attend(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Q, K, V: [B, N, M, k]
        row_mask: [B, M] -- used for the per-table scale (real row count)
        col_mask: [B, N] -- prevents attending to padding columns

        returns: [B, N, M, k]
        """

        B, N, M, k = Q.shape
        H, d = self.num_heads, self.head_dim

        real_m = row_mask.sum(dim=-1).clamp(min=1.0)  # [B]
        # per-head scale: each head sums over rows AND its own d channels
        scale = (real_m.view(B, 1, 1, 1) * d) ** 0.5  # [B,1,1,1] -> (i,j,h)

        Qh = Q.reshape(B, N, M, H, d)
        Kh = K.reshape(B, N, M, H, d)
        Vh = V.reshape(B, N, M, H, d)

        # per-head row-matched score: Frobenius inner product of the two
        # [M, d] blocks (row m of column i vs row m of column j), summed
        # jointly over rows AND the head's d channels -- one N x N map per
        # head instead of a single map collapsing all k channels.
        scores = torch.einsum("bimhd,bjmhd->bijh", Qh, Kh) / scale  # [B, N, N, H]

        doc_mask = col_mask.view(B, 1, N, 1)  # mask key columns j, all heads
        scores = scores.masked_fill(doc_mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=2)  # softmax over key columns j, per head

        out = torch.einsum("bijh,bjmhd->bimhd", attn, Vh)  # [B, N, M, H, d]
        out = out.reshape(B, N, M, k)

        return self.W_O(out)  # W_O mixes the heads (kept for uniformity)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attend(
            self.norm_q(Q), self.norm_k(K), self.norm_v(V), row_mask, col_mask
        )
        return V + attn_out


# ==========================================================
# CHANNEL MIX (shared MLP, across the k channels)
# ==========================================================

class ChannelMix(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int | None = None):
        super().__init__()

        hidden_dim = hidden_dim or input_dim * 2

        self.pre_norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.post_norm = nn.LayerNorm(input_dim)

    def forward(self, columns: torch.Tensor) -> torch.Tensor:
        """columns: [B, N, M, k] (or any shape ending in k)"""
        mixed = self.mlp(self.pre_norm(columns))
        return self.post_norm(columns + mixed)


# ==========================================================
# TABLE LAYER (one repeatable block: the unit that gets stacked)
# ==========================================================

class TableLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nonlinearity: str = "sigmoid",
        channel_mix_hidden_dim: int | None = None,
        num_heads: int = 8,
    ):
        super().__init__()

        self.column_aggregator = ColumnAggregator(input_dim, nonlinearity=nonlinearity, num_heads=num_heads)
        self.cross_column_attention = CrossColumnAttention(input_dim, num_heads=num_heads)
        self.channel_mix = ChannelMix(input_dim, channel_mix_hidden_dim)

    def forward(
        self,
        X: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        Q, K, V = self.column_aggregator(X, row_mask)
        X = self.cross_column_attention(Q, K, V, row_mask, col_mask)
        X = self.channel_mix(X)
        return X


# ==========================================================
# TABLE ENCODER (orchestrator)
# ==========================================================

class TableEncoder(nn.Module):
    def __init__(
        self,
        cell_encoder: nn.Module,
        embed_dim: int,
        num_layers: int = 1,
        nonlinearity: str = "sigmoid",
        channel_mix_hidden_dim: int | None = None,
        num_heads: int = 8,
        table_microbatch_cell_budget: int | None = None,
        table_microbatch_max_tables: int | None = None,
    ):
        super().__init__()

        self.cell_encoder = cell_encoder
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        if table_microbatch_cell_budget is not None and table_microbatch_cell_budget <= 0:
            raise ValueError("table_microbatch_cell_budget must be positive or None")
        if table_microbatch_max_tables is not None and table_microbatch_max_tables <= 0:
            raise ValueError("table_microbatch_max_tables must be positive or None")
        # Maximum padded cell slots (B * N_max * M_max) contextualized in
        # one table microbatch. This changes execution grouping only; all
        # candidate embeddings are restored to their original order before
        # the global query-table loss is formed.
        self.table_microbatch_cell_budget = table_microbatch_cell_budget
        self.table_microbatch_max_tables = table_microbatch_max_tables

        self.layers = nn.ModuleList(
            [
                TableLayer(embed_dim, nonlinearity, channel_mix_hidden_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )

    def save_text_cache(self, path: str) -> None:
        """Pass-through to CellEncoder.save_text_cache -- see that
        method's docstring. Only meaningful when self.cell_encoder is a
        real CellEncoder (the "ours" path); calling this on a model
        wrapping some other cell_encoder that doesn't implement it will
        raise AttributeError."""
        self.cell_encoder.save_text_cache(path)

    def load_text_cache(self, path: str, merge: bool = True) -> None:
        """Pass-through to CellEncoder.load_text_cache -- see that
        method's docstring."""
        self.cell_encoder.load_text_cache(path, merge=merge)

    def _encode_cellwise(
        self, tables: list[Table], ablation: str | None = None, profile: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs CellEncoder + the full TableLayer stack -- row-resolved,
        [B, N, M, k]. This is the same tensor shape
        src/scoring/multi_score.py's MultiScorer expects, and what the
        ELECTRA-style per-cell discriminator head needs.

        returns: (X, col_mask, row_mask, cell_mask) -- X: [B,N,M,k]
        """

        # Timing split ALWAYS recorded (not just when profile=True) as
        # self._last_frozen_s/_last_network_s -- same two attribute names
        # adapter.py's BaselineCellwiseAdapter exposes, so trainer.py's
        # _score_batch can read either model type uniformly without
        # branching on which one it is. profile=True additionally prints
        # a one-off line here for standalone/manual use.
        import time
        device = next(self.parameters()).device
        is_cuda = device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # H (per-column header embeddings) is no longer added onto X here --
        # CellEncoder already fuses each cell's own column header into that
        # cell's embedding directly (concat(cell, header), no projection),
        # so X arrives with header information already baked in. Adding H
        # again here would double-count it. H is still returned by
        # encode_tables_batched (e.g. for inspection/debugging) but is
        # otherwise unused on this path now.
        X, H, col_mask, row_mask, cell_mask, _n_list = self.cell_encoder.encode_tables_batched(tables)

        if ablation in ("headers_only", "content_only"):
            # No longer separable at this level: CellEncoder fuses header
            # and cell content together (concat, no projection) before this
            # point, so there's no clean "content without header" or
            # "header without content" tensor to zero out here anymore --
            # zeroing X for either mode would just produce the same
            # (meaningless) all-zero result for both. Revisit inside
            # CellEncoder if this ablation is still needed (e.g. expose
            # separate fused/content-only/header-only tensors from
            # encode_tables_batched).
            raise NotImplementedError(
                f"ablation={ablation!r} is not currently supported: header and "
                "cell content are fused inside CellEncoder before TableEncoder "
                "sees them, so they can no longer be isolated at this level."
            )
        elif ablation is not None:
            raise ValueError(f"Unknown ablation mode: {ablation}")

        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        for layer in self.layers:
            X = layer(X, row_mask, col_mask)

        if is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        self._last_frozen_s = t1 - t0
        self._last_network_s = t2 - t1

        if profile:
            total = self._last_frozen_s + self._last_network_s
            print(
                f"[profile] cell encoding (BERT+numeric): {self._last_frozen_s:.3f}s "
                f"({100*self._last_frozen_s/total:.1f}%)  |  "
                f"table-level layers ({self.num_layers}x): {self._last_network_s:.3f}s "
                f"({100*self._last_network_s/total:.1f}%)"
            )

        return X, col_mask, row_mask, cell_mask

    def forward_batch_cellwise(
        self, tables: list[Table], ablation: str | None = None, profile: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns per-CELL embeddings, for anything that needs cell-level
        resolution: the ELECTRA discriminator head (pretraining) and
        MultiScorer-based query-table scoring (finetuning). This is the
        only forward path in the current pipeline -- there is no
        table-level (one-vector-per-column) output anymore; that used to
        be produced by a RowCollapse module feeding table-vs-table MaxSim
        (src/scoring/maxsim.py), both removed since nothing in
        PretrainTrainer/FinetuneTrainer ever exercised them (see git
        history if that table-table retrieval path is ever needed again).

        When table_microbatch_cell_budget is set, candidates are grouped by
        power-of-two row/column shape bins, contextualized group-by-group,
        and padded/restored to the original candidate order here. Attention,
        normalization, and the pointwise MLP never mix different tables in
        the batch dimension, so this is mathematically the same model and
        leaves the subsequent full-query-pool InfoNCE loss unchanged.

        returns: (X, col_mask, row_mask, cell_mask) -- X: [B, N, M, k]
        """
        budget = self.table_microbatch_cell_budget
        max_tables_per_group = self.table_microbatch_max_tables
        if (budget is None and max_tables_per_group is None) or len(tables) <= 1:
            self._last_table_microbatches = 1
            return self._encode_cellwise(tables, ablation=ablation, profile=profile)

        # Bucketing both axes independently avoids combining a tall/narrow
        # table with a short/wide one and paying padding for the large empty
        # rectangle their unrelated maxima would create. Within a bin, each
        # axis is at most 2x its smallest member.
        def ceil_power_of_two(value: int) -> int:
            return 1 << (max(1, value) - 1).bit_length()

        bins: dict[tuple[int, int], list[int]] = {}
        for i, table in enumerate(tables):
            shape_bin = (
                ceil_power_of_two(table.num_columns),
                ceil_power_of_two(table.num_rows),
            )
            bins.setdefault(shape_bin, []).append(i)

        groups: list[list[int]] = []
        for (col_bin, row_bin), indices in sorted(
            bins.items(), key=lambda item: (item[0][0] * item[0][1], item[0])
        ):
            tables_per_group = len(indices)
            if budget is not None:
                tables_per_group = min(
                    tables_per_group, max(1, budget // (col_bin * row_bin))
                )
            if max_tables_per_group is not None:
                tables_per_group = min(tables_per_group, max_tables_per_group)
            groups.extend(
                indices[start : start + tables_per_group]
                for start in range(0, len(indices), tables_per_group)
            )

        # If all candidates already form one safe group in input order,
        # retain the original fast path and avoid padding/scatter overhead.
        if len(groups) == 1 and groups[0] == list(range(len(tables))):
            self._last_table_microbatches = 1
            return self._encode_cellwise(tables, ablation=ablation, profile=profile)

        per_table_x: list[torch.Tensor | None] = [None] * len(tables)
        per_table_col: list[torch.Tensor | None] = [None] * len(tables)
        per_table_row: list[torch.Tensor | None] = [None] * len(tables)
        per_table_cell: list[torch.Tensor | None] = [None] * len(tables)
        frozen_s = 0.0
        network_s = 0.0

        for group in groups:
            group_tables = [tables[i] for i in group]
            group_x, group_col, group_row, group_cell = self._encode_cellwise(
                group_tables, ablation=ablation, profile=False
            )
            frozen_s += self._last_frozen_s
            network_s += self._last_network_s

            for local_i, original_i in enumerate(group):
                n_cols = tables[original_i].num_columns
                n_rows = tables[original_i].num_rows
                per_table_x[original_i] = group_x[local_i, :n_cols, :n_rows]
                per_table_col[original_i] = group_col[local_i, :n_cols]
                per_table_row[original_i] = group_row[local_i, :n_rows]
                per_table_cell[original_i] = group_cell[local_i, :n_cols, :n_rows]

        if any(value is None for value in per_table_x):
            raise RuntimeError("internal error: table microbatching failed to restore every candidate")

        max_cols = max(table.num_columns for table in tables)
        max_rows = max(table.num_rows for table in tables)

        # F.pad + stack preserves each group's autograd graph, unlike
        # detaching/caching contextualized outputs, while presenting the
        # scorer with the exact same globally padded tensors as before.
        X = torch.stack(
            [
                F.pad(value, (0, 0, 0, max_rows - value.shape[1], 0, max_cols - value.shape[0]))
                for value in per_table_x
                if value is not None
            ],
            dim=0,
        )
        col_mask = torch.stack(
            [F.pad(value, (0, max_cols - value.shape[0])) for value in per_table_col if value is not None],
            dim=0,
        )
        row_mask = torch.stack(
            [F.pad(value, (0, max_rows - value.shape[0])) for value in per_table_row if value is not None],
            dim=0,
        )
        cell_mask = torch.stack(
            [
                F.pad(value, (0, max_rows - value.shape[1], 0, max_cols - value.shape[0]))
                for value in per_table_cell
                if value is not None
            ],
            dim=0,
        )

        self._last_frozen_s = frozen_s
        self._last_network_s = network_s
        self._last_table_microbatches = len(groups)

        if profile:
            total = frozen_s + network_s
            print(
                f"[profile] {len(tables)} tables -> {len(groups)} shape-aware microbatches | "
                f"cell encoding: {frozen_s:.3f}s ({100*frozen_s/max(total, 1e-12):.1f}%) | "
                f"table layers: {network_s:.3f}s ({100*network_s/max(total, 1e-12):.1f}%)"
            )

        return X, col_mask, row_mask, cell_mask


# ==========================================================
# DISCRIMINATOR HEAD (ELECTRA-style pretraining)
# ==========================================================
# Consumes forward_batch_cellwise()'s row-resolved output directly --
# one real/corrupted logit per cell, matching src/data/electra_corruption.py's
# per-cell label grids (via pad_labels()). Deliberately a SEPARATE module
# from TableEncoder (not a submodule of it) so it's trivial to keep out
# of a finetuning checkpoint -- see load_pretrained_encoder() below.

class DiscriminatorHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, N, M, k] -- row-resolved cell embeddings (TableEncoder.
           forward_batch_cellwise's output)
        returns: [B, N, M] -- one real/corrupted logit per cell (apply
                 sigmoid, or use with BCEWithLogitsLoss directly)
        """
        return self.mlp(X).squeeze(-1)


# ==========================================================
# CHECKPOINT TRANSFER (pretrain -> finetune)
# ==========================================================

def load_pretrained_encoder(
    model: TableEncoder,
    checkpoint_path: str,
    device: str = "cpu",
) -> None:
    """
    Loads a TableEncoder's weights from a pretraining checkpoint (saved
    by Trainer/PretrainTrainer.save_checkpoint(), which stores the WHOLE
    training state including any DiscriminatorHead parameters if it was
    registered as part of the saved state dict) -- but discards anything
    that isn't actually part of the TableEncoder itself.

    In practice this only matters if a DiscriminatorHead's parameters
    ever ended up prefixed under the saved state dict as if they were
    part of the model (e.g. "discriminator.*") -- since DiscriminatorHead
    is deliberately kept as a separate module (not a TableEncoder
    submodule) and pretraining code should save it under its own key
    rather than folding it into model_state_dict, this filter is a
    defensive no-op in the common case, and a real fix if that
    convention is ever violated.

    Loads in place (mutates `model`); returns nothing.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}

    dropped = set(state_dict.keys()) - set(filtered.keys())
    if dropped:
        print(
            f"[load_pretrained_encoder] discarded {len(dropped)} checkpoint "
            f"key(s) not part of this TableEncoder (e.g. a discriminator "
            f"head): {sorted(dropped)[:5]}{'...' if len(dropped) > 5 else ''}"
        )

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(
            f"[load_pretrained_encoder] {len(missing)} model parameter(s) had "
            f"no matching checkpoint entry (randomly initialized): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
