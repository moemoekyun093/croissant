"""MAP/MRR evaluation for lookup-free prepared query/corpus shards."""

from __future__ import annotations

import glob
import os
from collections.abc import Callable

import torch

from src.data.prepared_eval import (
    PreparedEvalQueries,
    PreparedEvalTables,
    PreparedStreamingEvalQueries,
    read_eval_shard,
)
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


def _ranking_metrics_from_mask(
    scores: torch.Tensor, positive_mask: torch.Tensor
) -> tuple[float, float]:
    """Return AP/RR sums for a fixed per-query candidate/relevance mask."""
    if scores.shape != positive_mask.shape:
        raise ValueError(
            f"score/positive shapes differ: {tuple(scores.shape)} vs "
            f"{tuple(positive_mask.shape)}"
        )
    if positive_mask.dtype != torch.bool:
        positive_mask = positive_mask.bool()
    if not torch.all(positive_mask.any(dim=1)):
        raise ValueError("streaming metric received a query without a positive")
    order = scores.argsort(dim=1, descending=True)
    ranked_relevance = positive_mask.gather(1, order)
    ranks = torch.arange(1, scores.shape[1] + 1, dtype=torch.float32).unsqueeze(0)
    precision = ranked_relevance.cumsum(dim=1) / ranks
    average_precision = (
        (precision * ranked_relevance).sum(dim=1)
        / ranked_relevance.sum(dim=1).clamp(min=1)
    )
    first_rank = ranked_relevance.float().argmax(dim=1) + 1
    return (
        average_precision.sum().item(),
        first_rank.float().reciprocal().sum().item(),
    )


def _check_eval_metadata(reference: dict, candidate: dict) -> None:
    for key in (
        "projection_dim", "projection_seed", "model_name", "max_rows",
        "max_columns", "split_sha256", "questions_sha256",
    ):
        if candidate.get(key) != reference.get(key):
            raise ValueError(f"prepared evaluation metadata mismatch for {key!r}")


class _PreparedTableStore:
    """Indexed, cached access to one globally prepared table corpus."""

    def __init__(
        self,
        paths: list[str],
        metadata: dict,
        device: torch.device,
    ) -> None:
        if not paths:
            raise ValueError("prepared streaming evaluation has no table shards")
        self.paths = paths
        self.metadata = metadata
        self.device = device
        self.table_shard_size = int(metadata["table_shard_size"])
        self.n_tables = int(metadata["n_tables"])
        self._cache: dict[int, PreparedEvalTables] = {}

    def _load(self, shard: int) -> PreparedEvalTables:
        payload = self._cache.get(shard)
        if payload is not None:
            return payload
        if shard < 0 or shard >= len(self.paths):
            raise IndexError(f"prepared table shard {shard} is out of range")
        metadata, loaded = read_eval_shard(self.paths[shard])
        _check_eval_metadata(self.metadata, metadata)
        if not isinstance(loaded, PreparedEvalTables):
            raise TypeError(f"expected table payload in {self.paths[shard]}")
        self._cache[shard] = loaded
        return loaded

    def materialize(self, global_indices: torch.Tensor):
        """Gather arbitrary corpus positions into one dense GPU table batch."""
        indices = [int(index) for index in global_indices.tolist()]
        if not indices:
            raise ValueError("cannot materialize an empty table batch")
        if min(indices) < 0 or max(indices) >= self.n_tables:
            raise IndexError("streaming candidate table index is out of range")

        records = []
        target_n = target_m = 0
        for global_index in indices:
            shard = global_index // self.table_shard_size
            local = global_index % self.table_shard_size
            payload = self._load(shard)
            if local >= len(payload.table_ids):
                raise IndexError(
                    f"global table {global_index} resolves past shard {shard}"
                )
            n = int(payload.col_mask[local].sum())
            m = int(payload.row_mask[local].sum())
            target_n = max(target_n, n)
            target_m = max(target_m, m)
            records.append((payload, local, n, m))

        # Re-pad only to this execution batch's true maxima. Padding is
        # masked in every encoder, so this is numerically equivalent while
        # avoiding the global 20x50 cost for batches of smaller tables.
        batch_size = len(records)
        row_mask = torch.zeros(batch_size, target_m, dtype=torch.bool)
        col_mask = torch.zeros(batch_size, target_n, dtype=torch.bool)
        cell_mask = torch.zeros(batch_size, target_n, target_m, dtype=torch.bool)
        cell_features = []
        header_features = []
        cell_scatter = []
        header_scatter = []
        table_ids = []

        for output_index, (payload, local, n, m) in enumerate(records):
            source_n = payload.col_mask.shape[1]
            source_m = payload.row_mask.shape[1]
            row_mask[output_index, :m] = payload.row_mask[local, :m]
            col_mask[output_index, :n] = payload.col_mask[local, :n]
            cell_mask[output_index, :n, :m] = payload.cell_mask[local, :n, :m]
            table_ids.append(payload.table_ids[local])

            cell_start = local * source_n * source_m
            cell_end = cell_start + source_n * source_m
            cell_lo = int(torch.searchsorted(payload.cell_scatter, cell_start))
            cell_hi = int(torch.searchsorted(payload.cell_scatter, cell_end))
            source_cell_scatter = payload.cell_scatter[cell_lo:cell_hi].to(torch.long)
            relative_cell = source_cell_scatter - cell_start
            columns = torch.div(relative_cell, source_m, rounding_mode="floor")
            rows = relative_cell.remainder(source_m)
            target_cell_scatter = (
                (output_index * target_n + columns) * target_m + rows
            )
            cell_features.append(payload.cell_features[cell_lo:cell_hi])
            cell_scatter.append(target_cell_scatter.to(torch.int32))

            header_start = local * source_n
            header_end = header_start + source_n
            header_lo = int(torch.searchsorted(payload.header_scatter, header_start))
            header_hi = int(torch.searchsorted(payload.header_scatter, header_end))
            source_header_scatter = payload.header_scatter[header_lo:header_hi].to(torch.long)
            relative_header = source_header_scatter - header_start
            target_header_scatter = output_index * target_n + relative_header
            header_features.append(payload.header_features[header_lo:header_hi])
            header_scatter.append(target_header_scatter.to(torch.int32))

        projection_dim = int(self.metadata["projection_dim"])
        packed_cells = (
            torch.cat(cell_features)
            if cell_features
            else torch.empty(0, projection_dim, dtype=torch.float16)
        )
        packed_headers = (
            torch.cat(header_features)
            if header_features
            else torch.empty(0, projection_dim, dtype=torch.float16)
        )
        gathered = PreparedEvalTables(
            table_ids=tuple(table_ids),
            cell_features=packed_cells,
            cell_scatter=torch.cat(cell_scatter),
            header_features=packed_headers,
            header_scatter=torch.cat(header_scatter),
            row_mask=row_mask,
            col_mask=col_mask,
            cell_mask=cell_mask,
        )
        gathered.validate(projection_dim)
        return gathered.materialize(self.device)


@torch.inference_mode()
def evaluate_prepared_streaming(
    prepared_dir: str,
    table_model,
    query_model,
    scorer,
    device: str | torch.device,
    training_metadata: dict,
    query_batch_size: int = 32,
    table_batch_size: int = 32,
    progress: Callable[[str], None] | None = print,
) -> dict[str, float]:
    """Evaluate every prepared query using fixed per-query candidate pools."""
    if query_batch_size <= 0 or table_batch_size <= 0:
        raise ValueError("streaming query/table batch sizes must be positive")
    if not os.path.exists(os.path.join(prepared_dir, "PREPARATION_COMPLETE")):
        raise ValueError(f"prepared evaluation set has no completion marker: {prepared_dir}")
    query_paths = sorted(glob.glob(os.path.join(prepared_dir, "query_chunks_*.pkl")))
    table_paths = sorted(glob.glob(os.path.join(prepared_dir, "tables_*.pkl")))
    if not query_paths or not table_paths:
        raise ValueError(f"prepared streaming evaluation set is incomplete: {prepared_dir}")

    first_metadata, first_payload = read_eval_shard(query_paths[0])
    if not isinstance(first_payload, PreparedStreamingEvalQueries):
        raise TypeError(f"expected streaming query payload in {query_paths[0]}")
    if first_metadata.get("evaluation_mode") != "streaming_per_query_candidates":
        raise ValueError("prepared directory is not a streaming candidate evaluation")
    _check_eval_metadata(training_metadata, first_metadata)

    device = torch.device(device)
    table_store = _PreparedTableStore(table_paths, first_metadata, device)
    table_model.eval()
    query_model.eval()
    scorer.eval()
    ap_sum = rr_sum = 0.0
    n_queries = 0
    candidate_pool_sum = 0

    for chunk_number, path in enumerate(query_paths, start=1):
        metadata, payload = read_eval_shard(path)
        _check_eval_metadata(first_metadata, metadata)
        if not isinstance(payload, PreparedStreamingEvalQueries):
            raise TypeError(f"expected streaming query payload in {path}")

        encoded_query_batches = []
        for start in range(0, payload.features.shape[0], query_batch_size):
            end = min(payload.features.shape[0], start + query_batch_size)
            features = payload.features[start:end].to(device=device, dtype=torch.float32)
            mask = payload.mask[start:end].to(device)
            encoded_query_batches.append(query_model(features, mask))
        queries = torch.cat(encoded_query_batches, dim=0)
        del encoded_query_batches

        pool_size = payload.candidate_table_indices.numel()
        scores = torch.empty(queries.shape[0], pool_size, dtype=torch.float32)
        for table_start in range(0, pool_size, table_batch_size):
            table_end = min(pool_size, table_start + table_batch_size)
            (
                cells,
                headers,
                row_mask,
                col_mask,
                _cell_mask,
            ) = table_store.materialize(
                payload.candidate_table_indices[table_start:table_end]
            )
            encoded_tables = table_model(cells, headers, row_mask, col_mask)
            row_mask_float = row_mask.float()
            col_mask_float = col_mask.float()
            for query_start in range(0, queries.shape[0], query_batch_size):
                query_end = min(queries.shape[0], query_start + query_batch_size)
                scores[query_start:query_end, table_start:table_end] = (
                    cross_score_queries_tables(
                        scorer,
                        "row_match",
                        queries[query_start:query_end],
                        encoded_tables,
                        row_mask_float,
                        col_mask_float,
                    ).cpu()
                )
            del (
                cells,
                headers,
                row_mask,
                col_mask,
                encoded_tables,
                row_mask_float,
                col_mask_float,
            )

        scores.masked_fill_(~payload.visible_mask, float("-inf"))
        chunk_ap, chunk_rr = _ranking_metrics_from_mask(
            scores, payload.positive_mask
        )
        ap_sum += chunk_ap
        rr_sum += chunk_rr
        n_queries += queries.shape[0]
        candidate_pool_sum += pool_size * queries.shape[0]
        if progress is not None:
            progress(
                f"[prepared-streaming-eval] [{n_queries}/"
                f"{int(first_metadata['n_queries'])} q | pool {pool_size}] "
                f"MAP {ap_sum / n_queries:.4f} MRR {rr_sum / n_queries:.4f} "
                f"({chunk_number}/{len(query_paths)} chunks; "
                f"cached table shards={len(table_store._cache)}/{len(table_paths)})"
            )
        del queries, scores

    return {
        "map": ap_sum / n_queries,
        "mrr": rr_sum / n_queries,
        "n_queries": n_queries,
        "n_tables": int(first_metadata["n_tables"]),
        "n_query_chunks": len(query_paths),
        "mean_candidate_pool": candidate_pool_sum / n_queries,
        "n_distractors_per_query": int(first_metadata["n_distractors"]),
    }


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
