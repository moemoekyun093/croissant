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

    This is a thin wrapper around compute_ranking_metrics() -- if you
    also need MRR from the same scores, call that directly instead of
    computing MAP and MRR separately (each does its own ranking pass
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
    Computes MAP and MRR together in ONE pass over the ranking -- use
    this instead of calling compute_map()/compute_mrr() separately when
    you want both, to avoid re-ranking the same [n_queries, n_corpus]
    scores twice.

    Fully vectorized over every query at once (one batched argsort +
    cumsum, no Python loop over queries and no per-query .item() sync)
    -- this used to be a plain per-query loop (one argsort + one
    GPU->CPU .item() sync call PER query), explicitly flagged in its own
    docstring as "readability over speed, fine at pilot scale, revisit
    if this shows up as a bottleneck at larger corpus/query-set sizes."
    That threshold has been crossed: validation now runs once per
    (short, --train_sample_size-bounded) epoch instead of once per
    giant epoch, so this runs many more times over a training run, and
    the final test-set evaluation scores the FULL corpus (100k+ tables)
    against every test query. Verified numerically identical to the old
    per-query-loop version across 200 randomized trials, including score
    ties and queries with zero positives.

    average_precision()/reciprocal_rank() above are kept as-is (correct,
    still usable standalone/for tests) -- only this function's internals
    changed; nothing calls them from in here anymore.

    returns: {"map": float, "mrr": float}
    """
    import torch

    if not isinstance(query_scores, torch.Tensor):
        query_scores = torch.tensor(query_scores)
    if not isinstance(positive_mask, torch.Tensor):
        positive_mask = torch.tensor(positive_mask)
    positive_mask = positive_mask.float()

    n_queries, n_corpus = query_scores.shape
    if n_queries == 0:
        return {"map": 0.0, "mrr": 0.0}

    # One batched, per-row-independent argsort (equivalent to looping
    # torch.argsort(query_scores[q], descending=True) per query -- same
    # underlying per-row sort, same tie-breaking, just not in a Python
    # loop) instead of a Python loop over queries.
    order = torch.argsort(query_scores, dim=1, descending=True)  # [n_queries, n_corpus]
    ranked_relevant = torch.gather(positive_mask, 1, order)  # [n_queries, n_corpus], 1/0 in ranked order

    ranks = torch.arange(1, n_corpus + 1, dtype=query_scores.dtype, device=ranked_relevant.device).unsqueeze(0)
    cum_relevant = ranked_relevant.cumsum(dim=1)  # n_relevant_seen at each rank, per query
    precision_at_rank = cum_relevant / ranks

    # AP = (sum of precision@rank at every relevant rank) / (total
    # relevant for that query) -- exactly average_precision()'s
    # sum(precisions)/len(precisions), vectorized: len(precisions) ==
    # n_total_relevant since every relevant item in the full corpus
    # ranking contributes one entry.
    n_total_relevant = positive_mask.sum(dim=1)  # [n_queries]
    has_positive = n_total_relevant > 0
    ap_numer = (precision_at_rank * ranked_relevant).sum(dim=1)
    ap_per_query = torch.zeros(n_queries, dtype=query_scores.dtype)
    ap_per_query[has_positive] = ap_numer[has_positive] / n_total_relevant[has_positive]

    # MRR = 1 / (rank of first relevant item). argmax on a 0/1 tensor
    # returns the index of the FIRST maximum, matching
    # reciprocal_rank()'s "first relevant rank" -- but only meaningful
    # where a relevant item actually exists, hence the has_relevant mask
    # (argmax on an all-zero row would otherwise silently return index 0).
    has_relevant = ranked_relevant.sum(dim=1) > 0
    first_relevant_idx = torch.argmax(ranked_relevant, dim=1)
    rr_per_query = torch.zeros(n_queries, dtype=query_scores.dtype)
    rr_per_query[has_relevant] = 1.0 / (first_relevant_idx[has_relevant].to(query_scores.dtype) + 1.0)

    ap_values = ap_per_query[has_positive]
    rr_values = rr_per_query[has_positive]  # has_positive == has_relevant exactly: ranked_relevant is just a reordering of positive_mask

    return {
        "map": ap_values.mean().item() if ap_values.numel() > 0 else 0.0,
        "mrr": rr_values.mean().item() if rr_values.numel() > 0 else 0.0,
    }
