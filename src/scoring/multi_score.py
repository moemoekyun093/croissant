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
    2. row_match:      mean_j max_i q^T X[i,j,:]
                       best row per column, AVERAGED over the table's
                       own real column count (not summed) -- so a table
                       with more columns doesn't win purely by having
                       more terms to accumulate; every table is scored
                       on the same per-column-average scale regardless
                       of its own width. (Was a plain sum; changed
                       deliberately, per-instruction, for table-size
                       fairness -- see row_match's inline comment. Note
                       this is NOT rank-neutral: it can and will change
                       which of two differently-sized tables ranks
                       higher for the same query, by design.)
    3. column_match:   mean_i max_j q^T X[i,j,:]
                       best column per row, averaged over the table's
                       own real row count -- same fairness rationale as
                       (2), transposed.
    4. col_deepset:    DeepSet-pool_j( max_i q^T X[i,j,:] )
                       same inner term as (2)'s pre-averaging step,
                       learned pooling over columns instead of a mean --
                       NOT changed to match (2)/(3)'s averaging (the
                       learned rho/phi pooling is a separate mechanism);
                       still has the same table-size-bias risk (2)/(3)
                       had before this change, since DeepSetPool1D's
                       internal sum isn't count-normalized either.
    5. row_deepset:    DeepSet-pool_i( max_j q^T X[i,j,:] )
                       same inner term as (3)'s pre-averaging step,
                       learned pooling over rows -- same caveat as (4).
    6. mixture:        lambda * mean_j max_i q^T X[i,j,:]
                       + (1-lambda) * mean_i q^T X'[i,:]
                       (2)'s formula on X (already row-count-averaged,
                       via calling row_match), blended with X'
                       (ColumnCollapse's row-indexed dual -- columns
                       already pooled away there, so no max is
                       needed/possible for that term) -- ALSO averaged
                       over real row count now, same fairness rationale,
                       lambda learned via a sigmoid-constrained parameter.

Every q^T X[i,j,:] term above is actually a COSINE similarity, not a raw
dot product: both Q and X (and X_prime, for mode 6) are L2-normalized
along the embedding dim right before each einsum (see _l2_normalize).
This bounds every per-cell term to [-1, 1] regardless of table size or
embedding norm, which keeps the aggregated (summed/max'd) scores in a
sane, comparable range across tables of very different shapes -- the
raw, unnormalized version let scores scale with table size and blew up
query_table_info_nce_loss's cross_scores/temperature into the
hundreds/thousands.

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
import torch.nn.functional as F


def _l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    L2-normalize along `dim`. Bounds every dot product between two
    normalized vectors to [-1, 1] (cosine similarity) instead of an
    unbounded raw dot product -- without this, summing/max-ing many
    unbounded per-cell scores across a whole table produces scores whose
    scale depends on table size and embedding norm, not just semantic
    match, which is what was making query_table_info_nce_loss's
    cross_scores/temperature blow up to the hundreds/thousands and
    produce erratic, uninformative loss values.

    F.normalize adds a small eps to the denominator (default 1e-12), so
    an all-zero vector (e.g. a padded/null cell's embedding) stays the
    zero vector instead of dividing by zero -- masking still zeroes out
    padded positions downstream anyway, this just avoids a NaN before
    that masking happens.
    """
    return F.normalize(x, p=2, dim=dim)


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
        returns: [B, L, N, M] -- cosine sim (L2-normalized dot product,
                 see _l2_normalize) for every (query, col, row)
        """
        Q = _l2_normalize(Q)
        X = _l2_normalize(X)
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
            # max_i q^T X[i,j,:] per column j, then MEAN over j (not sum)
            # -- averaging by each table's own real column count so a
            # table with more columns doesn't win purely by having more
            # terms to accumulate; every table gets scored on the same
            # [-1, 1]-ish scale regardless of its own width. See
            # module docstring.
            best_per_col = _masked_max(sims, row_mask.view(B, 1, 1, -1), dim=-1)  # [B, L, N]
            best_per_col = best_per_col * col_mask.view(B, 1, -1)
            n_real = col_mask.sum(dim=-1).clamp(min=1.0)  # [B]
            per_query = best_per_col.sum(dim=-1) / n_real.view(B, 1)  # [B, L]
            return per_query.sum(dim=1)

        if mode == "column_match":
            # max_j q^T X[i,j,:] per row i, then MEAN over i (not sum) --
            # same per-table fairness rationale as row_match above, this
            # time normalizing by the table's real row count.
            best_per_row = _masked_max(sims.transpose(-1, -2), col_mask.view(B, 1, 1, -1), dim=-1)  # [B, L, M]
            best_per_row = best_per_row * row_mask.view(B, 1, -1)
            m_real = row_mask.sum(dim=-1).clamp(min=1.0)  # [B]
            per_query = best_per_row.sum(dim=-1) / m_real.view(B, 1)
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
            term_a = self.score("row_match", Q, X, row_mask, col_mask)  # [B] -- already row-count-averaged internally
            sims_prime = torch.einsum("blk,bmk->blm", _l2_normalize(Q), _l2_normalize(X_prime))  # [B, L, M]
            sims_prime = sims_prime * row_mask.view(B, 1, -1)
            m_real = row_mask.sum(dim=-1).clamp(min=1.0)  # [B] -- same per-table fairness as row_match/column_match
            term_b = (sims_prime.sum(dim=-1) / m_real.view(B, 1)).sum(dim=1)  # [B]
            lam = self.mixture_lambda
            return lam * term_a + (1 - lam) * term_b

        raise ValueError(f"Unknown scoring mode: {mode}")

    def _cell_scores_cross(self, Q: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Q: [Bq, L, k], X: [Bt, N, M, k] -- DIFFERENT batch sizes on each
        side (score() above requires them equal/paired).
        returns: [Bq, Bt, L, N, M] -- cosine sim, see _l2_normalize.
        """
        Q = _l2_normalize(Q)
        X = _l2_normalize(X)
        return torch.einsum("qlk,tnmk->qtlnm", Q, X)

    def score_cross(
        self,
        mode: str,
        Q: torch.Tensor,
        X: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
        X_prime: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Every query in Q scored against every table in X, in ONE batched
        tensor op -- no Python-level loop over queries or tables (unlike
        looping score() once per query, which is what
        src/training/losses.py::cross_score_queries_tables used to do).
        That loop doesn't scale to a large candidate corpus (tens or
        hundreds of thousands of tables) -- this does the same math as a
        single vectorized computation instead.

        Only the corpus (X) side still needs to be processed in
        CHUNKS by the caller (see FinetuneTrainer.evaluate_map) -- that
        chunking is a genuine memory constraint (a 5-dimensional
        [Bq, Bt, L, N, M] similarity tensor scales with Bt), not a
        removable Python loop; this method itself has no loop at all
        within one (Q, X) chunk pair.

        Q:        [Bq, L, k]
        X:        [Bt, N, M, k] -- row-resolved, col-major
        row_mask: [Bt, M]
        col_mask: [Bt, N]
        X_prime:  [Bt, M, k] -- required for mode="mixture" only, see
                  score()'s docstring.

        returns: [Bq, Bt]
        """
        Bq, L, k = Q.shape
        Bt = X.shape[0]
        # [1, Bt, 1, N, M] -- broadcasts against sims' [Bq, Bt, L, N, M]
        cell_mask = col_mask.view(1, Bt, 1, -1, 1) * row_mask.view(1, Bt, 1, 1, -1)

        sims = self._cell_scores_cross(Q, X)  # [Bq, Bt, L, N, M]

        if mode == "global":
            flat_sims = sims.flatten(-2)  # [Bq, Bt, L, N*M]
            flat_mask = cell_mask.flatten(-2)  # [1, Bt, 1, N*M]
            per_query = _masked_max(flat_sims, flat_mask, dim=-1)  # [Bq, Bt, L]
            return per_query.sum(dim=-1)  # [Bq, Bt]

        if mode == "row_match":
            # MEAN over columns (per-table real column count), not sum --
            # see score()'s row_match for the fairness rationale.
            best_per_col = _masked_max(sims, row_mask.view(1, Bt, 1, 1, -1), dim=-1)  # [Bq,Bt,L,N]
            best_per_col = best_per_col * col_mask.view(1, Bt, 1, -1)
            n_real = col_mask.sum(dim=-1).clamp(min=1.0)  # [Bt]
            per_query = best_per_col.sum(dim=-1) / n_real.view(1, Bt, 1)  # [Bq, Bt, L]
            return per_query.sum(dim=-1)  # [Bq, Bt]

        if mode == "column_match":
            # MEAN over rows (per-table real row count), not sum.
            best_per_row = _masked_max(
                sims.transpose(-1, -2), col_mask.view(1, Bt, 1, 1, -1), dim=-1
            )  # [Bq, Bt, L, M]
            best_per_row = best_per_row * row_mask.view(1, Bt, 1, -1)
            m_real = row_mask.sum(dim=-1).clamp(min=1.0)  # [Bt]
            per_query = best_per_row.sum(dim=-1) / m_real.view(1, Bt, 1)
            return per_query.sum(dim=-1)

        if mode == "col_deepset":
            best_per_col = _masked_max(sims, row_mask.view(1, Bt, 1, 1, -1), dim=-1)  # [Bq,Bt,L,N]
            m = col_mask.view(1, Bt, 1, -1).expand(Bq, -1, L, -1)
            per_query = self.col_pool(best_per_col, m)  # [Bq, Bt, L]
            return per_query.sum(dim=-1)

        if mode == "row_deepset":
            best_per_row = _masked_max(
                sims.transpose(-1, -2), col_mask.view(1, Bt, 1, 1, -1), dim=-1
            )  # [Bq, Bt, L, M]
            m = row_mask.view(1, Bt, 1, -1).expand(Bq, -1, L, -1)
            per_query = self.row_pool(best_per_row, m)
            return per_query.sum(dim=-1)

        if mode == "mixture":
            if X_prime is None:
                raise ValueError(
                    "mode='mixture' requires X_prime: [Bt, M, k], see score()'s docstring."
                )
            term_a = self.score_cross("row_match", Q, X, row_mask, col_mask)  # [Bq, Bt] -- already row-count-averaged
            sims_prime = torch.einsum("qlk,tmk->qtlm", _l2_normalize(Q), _l2_normalize(X_prime))  # [Bq, Bt, L, M]
            sims_prime = sims_prime * row_mask.view(1, Bt, 1, -1)
            m_real = row_mask.sum(dim=-1).clamp(min=1.0)  # [Bt]
            term_b = (sims_prime.sum(dim=-1) / m_real.view(1, Bt, 1)).sum(dim=-1)  # [Bq, Bt]
            lam = self.mixture_lambda
            return lam * term_a + (1 - lam) * term_b

        raise ValueError(f"Unknown scoring mode: {mode}")