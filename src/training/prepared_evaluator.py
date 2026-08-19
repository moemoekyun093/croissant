"""MAP/MRR evaluation for lookup-free prepared query/corpus shards."""

from __future__ import annotations

import glob
import os
from collections.abc import Callable

import torch

from src.data.prepared_eval import PreparedEvalQueries, PreparedEvalTables, read_eval_shard
from src.training.losses import cross_score_queries_tables


def _ranking_metrics(scores: torch.Tensor, gold_ids, table_ids: list[str]) -> tuple[float, float]:
    table_index = {table_id: index for index, table_id in enumerate(table_ids)}
    relevance = torch.zeros(scores.shape, dtype=torch.bool)
    for query_index, labels in enumerate(gold_ids):
        for table_id in labels:
            candidate_index = table_index.get(table_id)
            if candidate_index is not None:
                relevance[query_index, candidate_index] = True
    if not torch.all(relevance.any(dim=1)):
        raise ValueError("prepared evaluation corpus omits a query's positive table")

    order = scores.argsort(dim=1, descending=True)
    ranked_relevance = relevance.gather(1, order)
    ranks = torch.arange(1, scores.shape[1] + 1, dtype=torch.float32).unsqueeze(0)
    precision = ranked_relevance.cumsum(dim=1) / ranks
    average_precision = (
        (precision * ranked_relevance).sum(dim=1)
        / ranked_relevance.sum(dim=1).clamp(min=1)
    )
    first_rank = ranked_relevance.float().argmax(dim=1) + 1
    reciprocal_rank = first_rank.float().reciprocal()
    return average_precision.sum().item(), reciprocal_rank.sum().item()


@torch.inference_mode()
def evaluate_prepared(
    prepared_dir: str,
    table_model,
    query_model,
    scorer,
    device: str | torch.device,
    training_metadata: dict,
    query_batch_size: int = 32,
    progress: Callable[[str], None] | None = print,
) -> dict[str, float]:
    """Evaluate one fixed prepared split without any text/table lookup."""
    device = torch.device(device)
    query_paths = sorted(glob.glob(os.path.join(prepared_dir, "queries_*.pkl")))
    table_paths = sorted(glob.glob(os.path.join(prepared_dir, "tables_*.pkl")))
    if not query_paths or not table_paths:
        raise ValueError(f"prepared evaluation set is incomplete: {prepared_dir}")
    if not os.path.exists(os.path.join(prepared_dir, "PREPARATION_COMPLETE")):
        raise ValueError(f"prepared evaluation set has no completion marker: {prepared_dir}")

    table_model.eval()
    query_model.eval()
    scorer.eval()

    # Contextualize each corpus table exactly once for this model state.
    contextualized = []
    table_ids: list[str] = []
    reference_metadata = None
    table_progress_every = max(1, len(table_paths) // 20)
    for table_shard_index, path in enumerate(table_paths, start=1):
        metadata, payload = read_eval_shard(path)
        if not isinstance(payload, PreparedEvalTables):
            raise TypeError(f"expected table payload in {path}")
        if reference_metadata is None:
            reference_metadata = metadata
            for key in (
                "projection_dim", "projection_seed", "model_name", "max_rows",
                "max_columns", "split_sha256", "questions_sha256",
            ):
                if metadata.get(key) != training_metadata.get(key):
                    raise ValueError(f"evaluation/training metadata mismatch for {key!r}")
        cells, headers, row_mask, col_mask, _cell_mask = payload.materialize(device)
        encoded = table_model(cells, headers, row_mask, col_mask)
        contextualized.append(
            (
                encoded.cpu(),
                row_mask.cpu(),
                col_mask.cpu(),
            )
        )
        table_ids.extend(payload.table_ids)
        if progress is not None and (
            table_shard_index % table_progress_every == 0
            or table_shard_index == len(table_paths)
        ):
            progress(
                f"[prepared-eval] contextualized table shard "
                f"{table_shard_index}/{len(table_paths)} "
                f"({len(table_ids)} tables)"
            )

    ap_sum = 0.0
    rr_sum = 0.0
    n_queries = 0
    query_progress_every = max(1, len(query_paths) // 20)
    for query_shard_index, path in enumerate(query_paths, start=1):
        metadata, payload = read_eval_shard(path)
        if not isinstance(payload, PreparedEvalQueries):
            raise TypeError(f"expected query payload in {path}")
        for key in (
            "projection_dim", "projection_seed", "model_name", "max_rows",
            "max_columns", "split_sha256", "questions_sha256",
        ):
            if metadata.get(key) != reference_metadata.get(key):
                raise ValueError(f"mixed prepared evaluation metadata for {key!r}")

        for query_start in range(0, payload.features.shape[0], query_batch_size):
            query_end = min(payload.features.shape[0], query_start + query_batch_size)
            features = payload.features[query_start:query_end].to(
                device=device, dtype=torch.float32
            )
            mask = payload.mask[query_start:query_end].to(device)
            q = query_model(features, mask)
            score_parts = []
            for encoded_cpu, row_cpu, col_cpu in contextualized:
                encoded = encoded_cpu.to(device)
                row_mask = row_cpu.to(device)
                col_mask = col_cpu.to(device)
                score_parts.append(
                    cross_score_queries_tables(
                        scorer, "row_match", q, encoded, row_mask.float(), col_mask.float()
                    ).cpu()
                )
                del encoded, row_mask, col_mask
            scores = torch.cat(score_parts, dim=1)
            gold = payload.gold_table_ids[query_start:query_end]
            ap, rr = _ranking_metrics(scores, gold, table_ids)
            ap_sum += ap
            rr_sum += rr
            n_queries += scores.shape[0]
        if progress is not None and (
            query_shard_index % query_progress_every == 0
            or query_shard_index == len(query_paths)
        ):
            progress(
                f"[prepared-eval] scored query shard "
                f"{query_shard_index}/{len(query_paths)} "
                f"({n_queries} queries)"
            )

    if n_queries == 0:
        raise ValueError("prepared evaluation set contains no queries")
    return {
        "map": ap_sum / n_queries,
        "mrr": rr_sum / n_queries,
        "n_queries": n_queries,
        "n_tables": len(table_ids),
    }
