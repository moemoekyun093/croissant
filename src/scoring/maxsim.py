"""
MaxSim: table-vs-table similarity over sets of column vectors.

batched_maxsim_matrix_padded is the primary entry point now -- it takes
an already-padded+masked batch (exactly what TableEncoder.forward_batch
produces) and computes the full N x N similarity matrix in one
vectorized GPU pass, no padding step and no Python loop over pairs.

batched_maxsim_matrix (list-based) and maxsim (single-pair) are kept as
convenience wrappers for ad-hoc use (e.g. sanity checks comparing two
individual tables) -- not used on the main training path anymore, since
TableEncoder no longer produces ragged per-table lists.
"""

import torch
import torch.nn.functional as F


def maxsim(query_table: torch.Tensor, doc_table: torch.Tensor) -> torch.Tensor:
    """
    query_table: [n_q, k]
    doc_table:   [n_d, k]

    ColBERT-style late interaction: for each of the query table's column
    vectors, take its max cosine similarity against any of the doc
    table's column vectors, then sum over the query table's columns.

    returns: a 0-dim tensor (the similarity score)
    """

    q = F.normalize(query_table, dim=-1)
    d = F.normalize(doc_table, dim=-1)

    sims = q @ d.transpose(0, 1)  # [n_q, n_d]

    return sims.max(dim=-1).values.sum()


def batched_maxsim_matrix_padded(
    X: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    X:    [N, max_n, k] -- already padded (e.g. TableEncoder.forward_batch's output)
    mask: [N, max_n]    -- 1 for real columns, 0 for padding

    returns: [N, N] similarity matrix, sims[i, j] = maxsim(X[i], X[j])
    (restricted to each table's real columns via mask)
    """

    X = F.normalize(X, dim=-1)
    X = X * mask.unsqueeze(-1)

    N, max_n, _k = X.shape

    # raw[i, qi, j, di] = X[i, qi] . X[j, di]
    raw = torch.einsum("iqk,jdk->iqjd", X, X)

    doc_mask = mask.view(1, 1, N, max_n).expand_as(raw)
    raw = raw.masked_fill(doc_mask == 0, float("-inf"))

    max_over_doc = raw.max(dim=-1).values  # [N, max_n, N]

    query_mask = mask.view(N, max_n, 1).expand_as(max_over_doc)
    max_over_doc = max_over_doc.masked_fill(query_mask == 0, 0.0)

    return max_over_doc.sum(dim=1)  # [N, N]


def _pad_and_stack(
    reprs: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience padding for ad-hoc list-based use (not the main
    training path). Involves a small Python loop -- fine for occasional
    use, not used per training step anymore."""

    N = len(reprs)
    k = reprs[0].shape[-1]
    max_n = max(r.shape[0] for r in reprs)
    device = reprs[0].device

    stacked = torch.zeros(N, max_n, k, device=device)
    mask = torch.zeros(N, max_n, device=device)

    for i, r in enumerate(reprs):
        n = r.shape[0]
        stacked[i, :n] = r
        mask[i, :n] = 1.0

    return stacked, mask


def batched_maxsim_matrix(reprs: list[torch.Tensor]) -> torch.Tensor:
    """List-based convenience wrapper around batched_maxsim_matrix_padded."""
    X, mask = _pad_and_stack(reprs)
    return batched_maxsim_matrix_padded(X, mask)