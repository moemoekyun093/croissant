"""
Retrieval evaluation metrics -- Mean Average Precision (MAP) over a
FIXED table corpus.

Per-instruction: only the QUERIES are split into train/val/test
(scripts/build_query_splits.py); the table corpus itself is never
partitioned, so every split ranks against the exact same, full set of
candidate tables. MAP is the metric used for early stopping
(src/training/trainer.py::FinetuneTrainer) -- val loss is no longer the
early-stopping signal, only best validation MAP is.
"""

from __future__ import annotations


def average_precision(ranked_relevant: list) -> float:
    """
    ranked_relevant: for ONE query, a list of 0/1 (or bool) values in
    RANKED ORDER -- highest-scored candidate first -- 1 if that
    candidate is a true positive for this query.

    Standard (uninterpolated) average precision: at every rank where a
    relevant item appears, compute precision@that_rank, then average
    those precision values. A query with zero relevant items anywhere
    in the ranking returns 0.0 -- callers should generally exclude such
    queries from a MAP average entirely (see compute_map), since AP is
    undefined rather than legitimately 0 when there's nothing to find
    in the corpus at all.
    """
    n_relevant_seen = 0
    precisions = []
    for rank, is_relevant in enumerate(ranked_relevant, start=1):
        if is_relevant:
            n_relevant_seen += 1
            precisions.append(n_relevant_seen / rank)
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def compute_map(query_scores, positive_mask) -> float:
    """
    Mean Average Precision across queries, ranking the FULL corpus per
    query by query_scores and scoring against positive_mask.

    query_scores:  [n_queries, n_corpus] -- score of each query against
                   every corpus table (higher = more relevant). Accepts
                   a torch.Tensor or anything array-like.
    positive_mask: [n_queries, n_corpus] -- 1/True where that corpus
                   table is a true positive for that query, 0/False
                   otherwise.

    Queries with zero positives in the corpus (positive_mask row sums
    to 0 -- e.g. a query whose only positive table got excluded from
    this particular corpus slice) are EXCLUDED from the mean, not
    counted as AP=0 -- that would penalize the model for a
    corpus/data-coverage issue outside its control, not a ranking
    failure.

    Implemented as a plain per-query argsort loop rather than a fully
    batched torch op -- readability over speed here, and fine at pilot
    scale (hundreds of val queries x however many corpus tables the
    pilot loaded). Revisit with a batched implementation if this shows
    up as a bottleneck at larger corpus/query-set sizes.
    """
    import torch

    if not isinstance(query_scores, torch.Tensor):
        query_scores = torch.tensor(query_scores)
    if not isinstance(positive_mask, torch.Tensor):
        positive_mask = torch.tensor(positive_mask)

    n_queries = query_scores.shape[0]
    ap_values = []

    for q in range(n_queries):
        pos_row = positive_mask[q]
        if pos_row.sum().item() == 0:
            continue
        order = torch.argsort(query_scores[q], descending=True)
        ranked_relevant = pos_row[order].tolist()
        ap_values.append(average_precision(ranked_relevant))

    if not ap_values:
        return 0.0
    return sum(ap_values) / len(ap_values)
