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
    """Evaluate one fixed prepared split without any text/table lookup.

    Queries are encoded once and retained on the GPU.  Evaluation is then
    table-major: each fixed-feature table shard is loaded, contextualized,
    scored against every query batch, and discarded.  This is the same
    complete score matrix as the previous query-major implementation, but
    avoids moving every contextualized table shard CPU->GPU once per query
    batch (roughly 94 redundant transfers for a 3,000-query validation set
    with query_batch_size=32).
    """
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

    # Encode every query exactly once and keep the compact result resident
    # on GPU.  For the usual 3,000 x 32 x 128 setup this is only ~47 MiB in
    # FP32, much smaller than repeatedly transferring the table corpus.
    query_embeddings = []
    gold_ids = []
    reference_metadata = None
    encoded_queries = 0
    query_progress_every = max(1, len(query_paths) // 20)
    for query_shard_index, path in enumerate(query_paths, start=1):
        metadata, payload = read_eval_shard(path)
        if not isinstance(payload, PreparedEvalQueries):
            raise TypeError(f"expected query payload in {path}")
        if reference_metadata is None:
            reference_metadata = metadata
            for key in (
                "projection_dim", "projection_seed", "model_name", "max_rows",
                "max_columns", "split_sha256", "questions_sha256",
            ):
                if metadata.get(key) != training_metadata.get(key):
                    raise ValueError(
                        f"evaluation/training metadata mismatch for {key!r}"
                    )
        else:
            for key in (
                "projection_dim", "projection_seed", "model_name", "max_rows",
                "max_columns", "split_sha256", "questions_sha256",
            ):
                if metadata.get(key) != reference_metadata.get(key):
                    raise ValueError(
                        f"mixed prepared evaluation metadata for {key!r}"
                    )

        for query_start in range(0, payload.features.shape[0], query_batch_size):
            query_end = min(payload.features.shape[0], query_start + query_batch_size)
            features = payload.features[query_start:query_end].to(
                device=device, dtype=torch.float32
            )
            mask = payload.mask[query_start:query_end].to(device)
            query_embeddings.append(query_model(features, mask))
            encoded_queries += query_end - query_start
        gold_ids.extend(payload.gold_table_ids)
        if progress is not None and (
            query_shard_index % query_progress_every == 0
            or query_shard_index == len(query_paths)
        ):
            progress(
                f"[prepared-eval] encoded query shard "
                f"{query_shard_index}/{len(query_paths)} "
                f"({encoded_queries} queries)"
            )

    if not query_embeddings:
        raise ValueError("prepared evaluation set contains no queries")
    queries = torch.cat(query_embeddings, dim=0)
    del query_embeddings

    # Table-major scoring: a table shard crosses the PCIe/NAS boundary only
    # once.  Its model-dependent contextualization remains on GPU while all
    # query batches are scored.  The small [Nq, shard_tables] score block is
    # then moved to CPU; concatenating blocks reconstructs the exact
    # [Nq, full_corpus] matrix in table-file order.
    table_ids: list[str] = []
    table_score_blocks = []
    table_progress_every = max(1, len(table_paths) // 20)
    for table_shard_index, path in enumerate(table_paths, start=1):
        metadata, payload = read_eval_shard(path)
        if not isinstance(payload, PreparedEvalTables):
            raise TypeError(f"expected table payload in {path}")
        for key in (
            "projection_dim", "projection_seed", "model_name", "max_rows",
            "max_columns", "split_sha256", "questions_sha256",
        ):
            if metadata.get(key) != reference_metadata.get(key):
                raise ValueError(f"mixed prepared evaluation metadata for {key!r}")

        cells, headers, row_mask, col_mask, _cell_mask = payload.materialize(device)
        encoded_tables = table_model(cells, headers, row_mask, col_mask)
        row_mask_float = row_mask.float()
        col_mask_float = col_mask.float()
        query_score_blocks = []
        for query_start in range(0, queries.shape[0], query_batch_size):
            query_end = min(queries.shape[0], query_start + query_batch_size)
            query_score_blocks.append(
                cross_score_queries_tables(
                    scorer,
                    "row_match",
                    queries[query_start:query_end],
                    encoded_tables,
                    row_mask_float,
                    col_mask_float,
                ).cpu()
            )
        table_score_blocks.append(torch.cat(query_score_blocks, dim=0))
        table_ids.extend(payload.table_ids)
        del (
            cells,
            headers,
            row_mask,
            col_mask,
            row_mask_float,
            col_mask_float,
            encoded_tables,
            query_score_blocks,
        )
        if progress is not None and (
            table_shard_index % table_progress_every == 0
            or table_shard_index == len(table_paths)
        ):
            progress(
                f"[prepared-eval] scored table shard "
                f"{table_shard_index}/{len(table_paths)} "
                f"({len(table_ids)} tables against {queries.shape[0]} queries)"
            )

    scores = torch.cat(table_score_blocks, dim=1)
    if scores.shape != (len(gold_ids), len(table_ids)):
        raise RuntimeError(
            f"prepared evaluation score shape mismatch: scores={tuple(scores.shape)}, "
            f"queries={len(gold_ids)}, tables={len(table_ids)}"
        )
    ap_sum, rr_sum = _ranking_metrics(scores, gold_ids, table_ids)
    n_queries = scores.shape[0]
    return {
        "map": ap_sum / n_queries,
        "mrr": rr_sum / n_queries,
        "n_queries": n_queries,
        "n_tables": len(table_ids),
    }
