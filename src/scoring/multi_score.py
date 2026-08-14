"""
Six candidate query-table scoring functions over row-resolved cell
embeddings X[i, j, :] (TableEncoder.forward_batch_cellwise's output),
as sketched on paper.

Convention: Q is a set of L query vectors (however produced upstream --
this module is agnostic to that). X is one document table's cell
embeddings, row-resolved, [n_cols, n_rows, k] for a single table (batched
versions below operate on padded [B, N, M, k] with row_mask/col_mask,
matching the rest of the codebase's masking convention).

    1. global:        max_{j,i} q^T X[i,j,:]
                       best-matching cell anywhere in the table -- no
                       row/column structure at all.
    2. row_match:      sum_j max_i q^T X[i,j,:]
                       best row per column, summed over columns.
    3. column_match:   sum_i max_j q^T X[i,j,:]
                       best column per row, summed over rows.
    4. col_deepset:    DeepSet-pool_j( max_i q^T X[i,j,:] )
                       same inner term as (2), learned pooling over
                       columns instead of a plain sum.
    5. row_deepset:    DeepSet-pool_i( max_j q^T X[i,j,:] )
                       same inner term as (3), learned pooling over rows.
    6. mixture:        lambda * sum_j max_i q^T X[i,j,:]
                       + (1-lambda) * sum_i q^T X'[i,:]
                       (2)'s formula on X, blended with a plain dot
                       product against X' (ColumnCollapse's row-indexed
                       dual -- columns already pooled away there, so no
                       max is needed/possible for that term), lambda
                       learned via a sigmoid-constrained parameter.

Final aggregation across the L query vectors in Q: summed, matching the
ColBERT/Starmie convention.

This module was written and smoke-tested standalone against dummy
tensors of the documented shapes (see conversation history for the test
script) -- it has NOT been run against the real TableEncoder/CellEncoder
pipeline, since those weren't available to check against. Verify
shapes/gradients against your actual encoder output before relying on
this in training.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DeepSetPool1D(nn.Module):
    """rho(sum_j phi(x_j)) over a 1-D sequence of SCALARS (the per-column
    or per-row max-sim values from options 2/3), replacing a plain sum
    with a learned, permutation-invariant pooling. Masked sum so padded
    positions contribute exactly zero regardless of phi's bias.
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.rho = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [..., S] -- S scalar values to pool (S = n_cols or n_rows)
        mask: [..., S] -- 1 for real positions, 0 for padding
        returns: [...] -- pooled scalar
        """
        phi_x = self.phi(x.unsqueeze(-1))  # [..., S, H]
        phi_x = phi_x * mask.unsqueeze(-1)  # zero out padding regardless of phi's bias
        pooled = phi_x.sum(dim=-2)  # [..., H]
        return self.rho(pooled).squeeze(-1)  # [...]


def _masked_max(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """max over `dim`, ignoring masked-out positions (mask==0 -> -inf
    before the max so padding never wins)."""
    scores = scores.masked_fill(mask == 0, float("-inf"))
    out = scores.max(dim=dim).values
    # a fully-masked slice (e.g. a padding column with no real rows) would
    # produce -inf; clamp back to 0 so it doesn't poison a later sum
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


class MultiScorer(nn.Module):
    """Computes any of the six scoring variants for a batch of (query-set,
    document-table) pairs. Options 4/5 own their DeepSetPool1D weights;
    option 6 owns its learned lambda. Options 1/2/3 are parameter-free.
    """

    def __init__(self, deepset_hidden_dim: int = 16):
        super().__init__()
        self.col_pool = DeepSetPool1D(deepset_hidden_dim)  # for option 4
        self.row_pool = DeepSetPool1D(deepset_hidden_dim)  # for option 5
        self._lambda_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5 init, for option 6

    @property
    def mixture_lambda(self) -> torch.Tensor:
        return torch.sigmoid(self._lambda_logit)

    def _cell_scores(self, Q: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Q: [B, L, k] -- L query vectors per table-pair in the batch
        X: [B, N, M, k] -- row-resolved cell embeddings (col-major: N=cols, M=rows)
        returns: [B, L, N, M] -- q^T X[i,j,:] for every (query, col, row)
        """
        return torch.einsum("blk,bnmk->blnm", Q, X)

    def score(
        self,
        mode: str,
        Q: torch.Tensor,
        X: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
        X_prime: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Q:        [B, L, k]
        X:        [B, N, M, k] -- row-resolved, col-major
        row_mask: [B, M]
        col_mask: [B, N]
        X_prime:  [B, M, k] -- required for mode="mixture" only; the
                  ColumnCollapse-derived, row-indexed dual representation.
                  Columns are already pooled away by ColumnCollapse, so
                  term_b is a plain per-row dot product (no max over a
                  nonexistent column axis).

        returns: [B] -- one score per table pair, summed over the L
                 query vectors (ColBERT/Starmie convention).
        """
        B, L, k = Q.shape
        cell_mask = col_mask.view(B, 1, -1, 1) * row_mask.view(B, 1, 1, -1)  # [B,1,N,M], broadcasts vs [B,L,N,M]

        sims = self._cell_scores(Q, X)  # [B, L, N, M]

        if mode == "global":
            per_query = _masked_max(sims.flatten(-2), cell_mask.flatten(-2), dim=-1)  # [B, L]
            return per_query.sum(dim=1)

        if mode == "row_match":
            # max_i q^T X[i,j,:] per column j, then sum_j
            best_per_col = _masked_max(sims, row_mask.view(B, 1, 1, -1), dim=-1)  # [B, L, N]
            best_per_col = best_per_col * col_mask.view(B, 1, -1)
            per_query = best_per_col.sum(dim=-1)  # [B, L]
            return per_query.sum(dim=1)

        if mode == "column_match":
            # max_j q^T X[i,j,:] per row i, then sum_i
            best_per_row = _masked_max(sims.transpose(-1, -2), col_mask.view(B, 1, 1, -1), dim=-1)  # [B, L, M]
            best_per_row = best_per_row * row_mask.view(B, 1, -1)
            per_query = best_per_row.sum(dim=-1)
            return per_query.sum(dim=1)

        if mode == "col_deepset":
            best_per_col = _masked_max(sims, row_mask.view(B, 1, 1, -1), dim=-1)  # [B, L, N]
            m = col_mask.view(B, 1, -1).expand(-1, L, -1)
            per_query = self.col_pool(best_per_col, m)  # [B, L]
            return per_query.sum(dim=1)

        if mode == "row_deepset":
            best_per_row = _masked_max(sims.transpose(-1, -2), col_mask.view(B, 1, 1, -1), dim=-1)  # [B, L, M]
            m = row_mask.view(B, 1, -1).expand(-1, L, -1)
            per_query = self.row_pool(best_per_row, m)  # [B, L]
            return per_query.sum(dim=1)

        if mode == "mixture":
            if X_prime is None:
                raise ValueError(
                    "mode='mixture' requires X_prime: [B, M, k], ColumnCollapse's "
                    "row-indexed dual (columns already pooled away -- no column "
                    "axis left, so term_b is a plain dot product, no max)."
                )
            term_a = self.score("row_match", Q, X, row_mask, col_mask)  # [B]
            sims_prime = torch.einsum("blk,bmk->blm", Q, X_prime)  # [B, L, M]
            sims_prime = sims_prime * row_mask.view(B, 1, -1)
            term_b = sims_prime.sum(dim=-1).sum(dim=1)  # [B]
            lam = self.mixture_lambda
            return lam * term_a + (1 - lam) * term_b

        raise ValueError(f"Unknown scoring mode: {mode}")