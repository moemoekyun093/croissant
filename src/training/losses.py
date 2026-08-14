"""
Losses for table representation learning.

Contains two distinct losses for two distinct training stages:
    electra_discriminator_loss -- ELECTRA-style cell-corruption
                                   pretraining (PretrainTrainer)
    query_table_info_nce_loss  -- real query->table contrastive
                                   finetuning, using MultiScorer instead
                                   of table-table MaxSim (FinetuneTrainer)
"""

import torch
import torch.nn.functional as F


def electra_discriminator_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    cell_mask: torch.Tensor,
) -> torch.Tensor:
    """
    ELECTRA-style replaced-cell-detection loss: binary cross-entropy per
    cell (real vs. corrupted), masked to only real, non-null cells --
    padding and originally-empty cells contribute nothing (there's
    nothing meaningful to detect there; see
    src/data/electra_corruption.py's corrupt_tables(), which never
    corrupts an already-empty cell either).

    logits:    [B, N, M] -- DiscriminatorHead's raw output (pre-sigmoid)
    labels:    [B, N, M] -- 1.0 = corrupted, 0.0 = original (from
               src/data/electra_corruption.py::pad_labels)
    cell_mask: [B, N, M] -- 1 for real, non-null cells (same mask
               TableEncoder.forward_batch_cellwise returns -- note this
               is CellEncoder's null-cell mask, not corruption-specific;
               combine with a col/row mask upstream if padding columns
               past a table's real width could otherwise sneak in a
               non-null-looking all-zero cell -- in practice cell_mask
               already implies real+non-null since padding cells are
               always empty strings, hence always excluded).

    returns: scalar loss -- mean BCE over every real, non-null cell in
    the batch.
    """
    per_cell = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    masked = per_cell * cell_mask

    denom = cell_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def cross_score_queries_tables(
    scorer,
    mode: str,
    Q: torch.Tensor,
    X: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Expands src/scoring/multi_score.py's MultiScorer.score() (which
    scores PAIRED (query_i, table_i) -- one score per batch position)
    into a full [Bq, Bt] cross matrix, every query against every table
    in the batch -- needed for in-batch-negative contrastive finetuning
    (query_table_info_nce_loss below).

    Loops over the query batch (Bq calls to scorer.score(), each scoring
    that one query against ALL Bt tables at once via broadcasting) --
    not a full O(Bq*Bt) Python loop, just a lighter Bq-only one,
    acceptable at typical batch sizes. Revisit with a fully vectorized
    einsum across both batch dims if this shows up as a bottleneck.

    Q:        [Bq, L, k] -- caller must already have zeroed out any
              padding-token rows (e.g. Q * query_mask.unsqueeze(-1)),
              since MultiScorer.score() has no query_mask argument of
              its own and treats every row of Q as a real query vector.
    X:        [Bt, N, M, k]  -- row-resolved cell embeddings
    row_mask: [Bt, M]
    col_mask: [Bt, N]

    returns: [Bq, Bt] score matrix
    """
    Bq = Q.shape[0]
    Bt = X.shape[0]

    rows = []
    for i in range(Bq):
        q_i = Q[i : i + 1].expand(Bt, -1, -1)  # [Bt, L, k]
        # MultiScorer.score() doesn't take a query_mask argument today --
        # it assumes every row of Q is a valid query vector. If a query
        # has padding tokens (shorter than the batch's max length), zero
        # those positions out of Q BEFORE calling this function (see
        # FinetuneTrainer.train_step) so they contribute a zero dot
        # product rather than a spurious signal. query_mask is accepted
        # here for that same masking, applied by the caller in advance.
        scores_i = scorer.score(mode, q_i, X, row_mask, col_mask)  # [Bt]
        rows.append(scores_i)

    return torch.stack(rows, dim=0)  # [Bq, Bt]


def query_table_info_nce_loss(
    cross_scores: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Real query -> positive-table contrastive loss (finetuning), given a
    [Bq, Bt] cross-score matrix (cross_score_queries_tables' output).
    Assumes the batch is constructed with Bq == Bt and query i's
    positive at table i (SynSQLQueryDataset-based batches, one positive
    table sampled per query) -- other tables in the batch are in-batch
    negatives.

    One-directional (query -> table) softmax cross-entropy, NOT
    symmetric like info_nce_loss's table-table version: retrieval here
    is inherently asymmetric (we score/rank tables given a query, never
    the reverse), so there's no natural "table -> query" direction to
    average in, unlike table-table's two genuinely symmetric augmented
    views.

    returns: scalar loss
    """
    assert cross_scores.shape[0] == cross_scores.shape[1], (
        "query_table_info_nce_loss assumes one positive table per query, "
        "i.e. a square [B, B] cross-score matrix with the positive on "
        "the diagonal"
    )

    B = cross_scores.shape[0]
    targets = torch.arange(B, device=cross_scores.device)

    return F.cross_entropy(cross_scores / temperature, targets)