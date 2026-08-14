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


def reciprocal_rank(ranked_relevant: list) -> float:
    """
    ranked_relevant: same convention as average_precision's argument --
    a list of 0/1 in ranked order, highest-scored candidate first.

    1 / rank of the FIRST relevant item (rank is 1-indexed). 0.0 if
    nothing relevant appears anywhere in the ranking. Unlike AP, MRR
    only cares about how quickly you find ONE correct answer, not how
    well-ranked every correct answer is -- a useful complement to MAP
    when "did the top result work" matters as much as "are all the
    right tables near the top."
    """
    for rank, is_relevant in enumerate(ranked_relevant, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


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

    This is a thin wrapper around compute_ranking_metrics() -- if you
    also need MRR from the same scores, call that directly instead of
    computing MAP and MRR separately (each does its own argsort pass
    over the same [n_queries, n_corpus] scores otherwise).
    """
    return compute_ranking_metrics(query_scores, positive_mask)["map"]


def compute_mrr(query_scores, positive_mask) -> float:
    """Mean Reciprocal Rank -- see reciprocal_rank's docstring. Same
    exclusion rule as compute_map: queries with zero positives in this
    corpus are excluded from the mean, not scored as 0. Thin wrapper
    around compute_ranking_metrics() -- see compute_map's docstring."""
    return compute_ranking_metrics(query_scores, positive_mask)["mrr"]


def compute_ranking_metrics(query_scores, positive_mask) -> dict:
    """
    Computes MAP and MRR together in ONE pass over the ranking (one
    argsort per query, shared by both metrics) -- use this instead of
    calling compute_map()/compute_mrr() separately when you want both,
    to avoid re-ranking the same [n_queries, n_corpus] scores twice.

    returns: {"map": float, "mrr": float}
    """
    import torch

    if not isinstance(query_scores, torch.Tensor):
        query_scores = torch.tensor(query_scores)
    if not isinstance(positive_mask, torch.Tensor):
        positive_mask = torch.tensor(positive_mask)

    n_queries = query_scores.shape[0]
    ap_values = []
    rr_values = []

    for q in range(n_queries):
        pos_row = positive_mask[q]
        if pos_row.sum().item() == 0:
            continue
        order = torch.argsort(query_scores[q], descending=True)
        ranked_relevant = pos_row[order].tolist()
        ap_values.append(average_precision(ranked_relevant))
        rr_values.append(reciprocal_rank(ranked_relevant))

    return {
        "map": sum(ap_values) / len(ap_values) if ap_values else 0.0,
        "mrr": sum(rr_values) / len(rr_values) if rr_values else 0.0,
    }
