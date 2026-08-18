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
    Every query in Q scored against every table in X, via
    src/scoring/multi_score.py's MultiScorer.score_cross() -- ONE
    batched tensor op, no Python-level loop over queries or tables.
    Needed for in-batch-negative contrastive finetuning
    (query_table_info_nce_loss below) and for full-corpus MAP ranking
    (FinetuneTrainer.evaluate_map), where Bt can be tens or hundreds of
    thousands of candidate tables -- a per-query Python loop doesn't
    scale to that (this function used to loop Bq times calling
    scorer.score() per query; replaced with score_cross(), which does
    the same math as a single vectorized computation).

    Q:        [Bq, L, k] -- caller must already have zeroed out any
              padding-token rows (e.g. Q * query_mask.unsqueeze(-1)),
              since MultiScorer.score_cross() has no query_mask argument
              of its own and treats every row of Q as a real query
              vector.
    X:        [Bt, N, M, k]  -- row-resolved cell embeddings
    row_mask: [Bt, M]
    col_mask: [Bt, N]

    returns: [Bq, Bt] score matrix
    """
    return scorer.score_cross(mode, Q, X, row_mask, col_mask)


def query_table_info_nce_loss(
    cross_scores: torch.Tensor,
    positive_indices: torch.Tensor | None = None,
    positive_mask: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Real query -> positive-table contrastive loss (finetuning), given a
    [Bq, Bt] cross-score matrix (cross_score_queries_tables' output).
    With ``positive_mask`` omitted, every non-target column is a negative
    and this is ordinary one-positive cross-entropy.  With a mask, every
    ``True`` entry in row i is a valid answer for query i and the loss is
    multi-positive InfoNCE.  This prevents a table that is valid for two
    queries in one batch from becoming a false negative for either one.

    positive_indices: [Bq] int64, the column index of query i's true
        positive table. Defaults to arange(Bq) -- positive for query i
        at column i -- matching resolve_train_batches' convention: the
        batch's Bq positive tables are always placed first (columns
        0..Bq-1, one per query, in order), any hard negatives appended
        afterward. Pass this explicitly only if a caller ever needs a
        different positive layout.
    positive_mask: optional bool [Bq, Bt] matrix whose true entries mark
        every candidate table valid for each query.  Mutually exclusive
        with ``positive_indices``.

    One-directional (query -> table) softmax cross-entropy, NOT
    symmetric like info_nce_loss's table-table version: retrieval here
    is inherently asymmetric (we score/rank tables given a query, never
    the reverse), so there's no natural "table -> query" direction to
    average in, unlike table-table's two genuinely symmetric augmented
    views.

    returns: scalar loss
    """
    Bq, Bt = cross_scores.shape

    if positive_mask is not None:
        if positive_indices is not None:
            raise ValueError("pass either positive_indices or positive_mask, not both")
        if positive_mask.shape != cross_scores.shape:
            raise ValueError(
                "positive_mask must have the same shape as cross_scores: "
                f"got {tuple(positive_mask.shape)} vs {tuple(cross_scores.shape)}"
            )
        positive_mask = positive_mask.to(device=cross_scores.device, dtype=torch.bool)
        if not torch.all(positive_mask.any(dim=1)):
            missing = (~positive_mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"multi-positive loss received queries with no candidate positive: {missing[:10]}")

        scaled = cross_scores / temperature
        positive_logsumexp = torch.logsumexp(
            scaled.masked_fill(~positive_mask, float("-inf")), dim=1
        )
        return (torch.logsumexp(scaled, dim=1) - positive_logsumexp).mean()

    if positive_indices is None:
        assert Bq <= Bt, (
            f"query_table_info_nce_loss got Bq={Bq} queries but only "
            f"Bt={Bt} tables -- every query needs at least its own "
            f"positive table present as a column."
        )
        positive_indices = torch.arange(Bq, device=cross_scores.device)

    return F.cross_entropy(cross_scores / temperature, positive_indices)
