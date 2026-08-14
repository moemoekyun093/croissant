"""
RCPE (Row-Collapsing Permutation-Equivariant) attention block.

Replaces ColumnAggregator + CrossColumnAttention as a single mechanism.
Derived on paper together, step by step -- see conversation history for
the full derivation and every masking/scaling/residual decision. Summary
of the final design:

For a single channel c, column j's row-resolved slice z = X[:, j, :, c]
in R^M (M = padded row count):

    z_hat   = z / ||z||_2                (real rows only; unit-norm,
                                            keeps the Gram matrix bounded
                                            regardless of upstream scale)
    G       = z_hat @ z_hat^T            in R^{MxM}, masked to zero at
                                            any (i,t) touching a padded row
                                            BEFORE the nonlinearity
    gate    = scale * mean(z) + bias     (mean over real rows of the
                                            UNNORMALIZED z, so magnitude
                                            information survives; gate is
                                            permutation-invariant, so this
                                            stays entrywise-equivariant)
    NetOut  = sigmoid(gate * G)          re-masked to zero at padded
                                            positions AFTER the sigmoid,
                                            since sigmoid(0) = 0.5 != 0
    out     = NetOut @ z                 in R^M -- still row-resolved,
                                            not yet collapsed

This same three-step (normalize -> gate -> gated Gram nonlinearity ->
matvec) mechanism is applied independently for Q, K, and V (three
separate `scale`/`bias` pairs, i.e. three separate GramGateNet
instances), weight-tied ACROSS channels (depthwise: the same GramGateNet
instance processes every one of the d_l channel slices the same way,
not one set of weights per channel).

Rows only actually collapse at the attention score: q_j^(c) . k_t^(c) is
a dot product over the M row axis, producing one scalar per (channel,
column-pair). Attention is computed INDEPENDENTLY per channel (each
channel gets its own N x N score matrix / softmax) -- more expressive
than a shared-across-channels score, at a real compute cost (documented
in TableEncoder's cost caveats).

No residual connection: confirmed explicitly that the block's output is
purely the attention-weighted sum over v_t (self-inclusion via t=j in
the softmax is the only "retention" mechanism -- no separate learned
coefficient, no added raw z_j).

RCPE keeps the input/output shape contract of the block it replaces:
[B, N, M, k] in, [B, N, M, k] out -- so TableLayer, ChannelMix,
TableEncoder.forward_batch, header injection, and RowCollapse are all
UNCHANGED and do not need to know this block was swapped in.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GramGateNet(nn.Module):
    """One of the three (Q / K / V) branches. Weight-tied across all d_l
    channels -- a single (scale, bias) pair, applied identically (via
    broadcasting) to every channel slice, not per-channel parameters.
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, X: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
        """
        X:        [B, N, M, C] -- one branch's channel-mixed input (all
                   C channels at once; the (scale, bias) gate is shared
                   across channels, but each channel is still processed
                   independently -- there is no mixing BETWEEN channels
                   in this step, only within each channel's own M rows).
        row_mask: [B, M] -- 1 for real rows, 0 for padding.

        returns: [B, N, M, C] -- still row-resolved (rows collapse later,
                 at the attention score in RCPEAttention, not here).
        """
        B, N, M, C = X.shape
        rm = row_mask.view(B, 1, M, 1)  # broadcast over N, C
        z = X * rm  # defensive; should already be zero from bias-free channel-mix

        denom = row_mask.sum(dim=-1).clamp(min=1.0)  # [B]
        norm = z.norm(dim=2, keepdim=True).clamp(min=1e-6)  # [B, N, 1, C], over real rows only (padding is already 0)
        z_hat = z / norm  # unit-norm per (table, column, channel)

        # Gram matrix per (table, column, channel): G[b, n, i, t, c]
        G = torch.einsum("bnic,bntc->bnitc", z_hat, z_hat)  # [B, N, M, M, C]

        pair_mask = (row_mask.view(B, 1, M, 1) * row_mask.view(B, 1, 1, M)).unsqueeze(-1)  # [B,1,M,M,1]
        G = G * pair_mask  # zero padded entries BEFORE the nonlinearity

        channel_mean = (z * rm).sum(dim=2) / denom.view(B, 1, 1)  # [B, N, C], unnormalized z's mean
        gate = self.scale * channel_mean + self.bias  # [B, N, C], permutation-invariant

        net_out = torch.sigmoid(gate.view(B, N, 1, 1, C) * G)  # [B, N, M, M, C]
        net_out = net_out * pair_mask  # re-mask AFTER sigmoid (sigmoid(0) = 0.5, not 0)

        out = torch.einsum("bnitc,bntc->bnic", net_out, z)  # matvec back onto (unnormalized) z
        return out  # [B, N, M, C]


class RCPEAttention(nn.Module):
    """Full RCPE block: three GramGateNet branches (Q, K, V) + per-channel
    cross-column attention. Drop-in replacement for
    ColumnAggregator + CrossColumnAttention -- same [B,N,M,k] -> [B,N,M,k]
    contract, consumed by TableLayer exactly like the block it replaces.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.input_norm = nn.LayerNorm(input_dim)

        # bias-free: padded (zero) rows must stay exactly zero after mixing
        self.W_Q_mix = nn.Linear(input_dim, input_dim, bias=False)
        self.W_K_mix = nn.Linear(input_dim, input_dim, bias=False)
        self.W_V_mix = nn.Linear(input_dim, input_dim, bias=False)

        self.query_net = GramGateNet()
        self.key_net = GramGateNet()
        self.value_net = GramGateNet()

    def forward(
        self,
        X: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        X:        [B, N, M, k]
        row_mask: [B, M]
        col_mask: [B, N]

        returns: [B, N, M, k]
        """
        B, N, M, k = X.shape

        Xn = self.input_norm(X)
        X_Q = self.W_Q_mix(Xn)
        X_K = self.W_K_mix(Xn)
        X_V = self.W_V_mix(Xn)

        q = self.query_net(X_Q, row_mask)  # [B, N, M, k]
        kk = self.key_net(X_K, row_mask)   # [B, N, M, k]
        v = self.value_net(X_V, row_mask)  # [B, N, M, k]

        # rows collapse HERE: dot product over the M row axis, per channel c,
        # per (query-column j, key-column t) pair
        real_m = row_mask.sum(dim=-1).clamp(min=1.0)  # [B]
        scale = real_m.view(B, 1, 1, 1).sqrt()  # [B,1,1,1], broadcasts over (j,t,c)

        scores = torch.einsum("bjmc,btmc->bjtc", q, kk) / scale  # [B, N, N, C] -- per-channel independent

        doc_mask = col_mask.view(B, 1, N, 1)  # mask key/doc side (t)
        scores = scores.masked_fill(doc_mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=2)  # softmax over t, independently per channel c

        out = torch.einsum("bjtc,btmc->bjmc", attn, v)  # [B, N, M, k], row-resolved again

        # zero out padded query columns' output for cleanliness downstream
        out = out * col_mask.view(B, N, 1, 1)
        return out


class ColumnCollapse(nn.Module):
    """Dual of RowCollapse: attention-pools over the COLUMN axis (N)
    instead of the row axis (M), producing one vector per (table, row)
    instead of per (table, column). This is the X' construction from the
    diagram's top-right sketch -- RowCollapse itself is UNCHANGED and not
    touched by this module; ColumnCollapse is purely additive, used only
    where a row-indexed summary (X') is needed, e.g. scoring option 6.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.seed = nn.Parameter(torch.randn(1, input_dim))
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward_table(
        self,
        columns: torch.Tensor,
        col_mask: torch.Tensor,
        cell_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        columns:   [B, N, M, k]
        col_mask:  [B, N] -- 1 for real (non-padding) columns
        cell_mask: [B, N, M] -- 1 for non-null cells specifically (same
                   mask RowCollapse uses, just pooled over the other axis
                   here). Same null-column-collapse-attractor guard as
                   RowCollapse: if a row has zero non-null cells across
                   all its real columns, fall back to col_mask alone for
                   that row rather than producing an all -inf softmax.

        returns: [B, M, k] -- one pooled vector per (table, row)
        """
        B, N, M, k = columns.shape

        has_any_nonnull = cell_mask.sum(dim=1, keepdim=True) > 0  # [B, 1, M] -- any real column non-null, per row
        has_any_nonnull = has_any_nonnull.transpose(1, 2)  # [B, M, 1]
        col_mask_expanded = col_mask.view(B, N, 1).expand(-1, -1, M).transpose(1, 2)  # [B, M, N]
        cell_mask_t = cell_mask.transpose(1, 2)  # [B, M, N]
        effective_mask = torch.where(has_any_nonnull, cell_mask_t, col_mask_expanded)  # [B, M, N]

        cols_t = columns.transpose(1, 2)  # [B, M, N, k]
        K = self.k_proj(cols_t)
        V = self.v_proj(cols_t)

        seed = self.seed.view(1, 1, 1, k).expand(B, M, 1, k)
        scores = torch.einsum("bmqk,bmnk->bmqn", seed, K) / (self.input_dim ** 0.5)

        mask = effective_mask.view(B, M, 1, N)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # [B, M, 1, N]
        pooled = torch.einsum("bmqn,bmnk->bmqk", attn, V).squeeze(2)  # [B, M, k]

        return self.out_proj(pooled)