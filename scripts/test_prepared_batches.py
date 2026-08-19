"""Small CPU check for prepared serialization, labels, gradients, and shapes."""

from __future__ import annotations

import os
import tempfile

import torch

from src.data.prepared_batches import (
    PreparedBatch,
    PreparedBatchWriter,
    iter_prepared_batches,
    read_prepared_metadata,
)
from src.data.prepared_eval import (
    PreparedEvalQueries,
    PreparedEvalTables,
    read_eval_shard,
    write_eval_shard,
)
from src.models.prepared_table_encoder import PreparedQueryEncoder, PreparedTableEncoder
from src.scoring.multi_score import MultiScorer
from src.training.losses import cross_score_queries_tables, query_table_info_nce_loss
from src.training.prepared_evaluator import _ranking_metrics


def main() -> None:
    torch.manual_seed(7)
    batch = PreparedBatch(
        query_texts=("q0", "q1", "q2"),
        candidate_table_ids=("t0", "t1", "t2", "t3", "t4"),
        query_features=torch.randn(3, 5, 8).half(),
        query_mask=torch.ones(3, 5, dtype=torch.bool),
        cell_features=torch.randn(5 * 4 * 6, 8).half(),
        cell_scatter=torch.arange(5 * 4 * 6, dtype=torch.int32),
        header_features=torch.randn(5 * 4, 8).half(),
        header_scatter=torch.arange(5 * 4, dtype=torch.int32),
        row_mask=torch.ones(5, 6, dtype=torch.bool),
        col_mask=torch.ones(5, 4, dtype=torch.bool),
        cell_mask=torch.ones(5, 4, 6, dtype=torch.bool),
        positive_mask=torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ],
            dtype=torch.bool,
        ),
    )
    batch.validate(8)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "epoch_000.pkl")
        with PreparedBatchWriter(path, {"projection_dim": 8, "seed": 7}) as writer:
            writer.write(batch)
        assert read_prepared_metadata(path)["projection_dim"] == 8
        restored = list(iter_prepared_batches(path))
        assert len(restored) == 1
        assert torch.equal(restored[0].positive_mask, batch.positive_mask)
        assert torch.equal(restored[0].query_features, batch.query_features)

        query_path = os.path.join(directory, "queries_00000.pkl")
        eval_queries = PreparedEvalQueries(
            query_texts=("q0", "q1"),
            gold_table_ids=(("t0", "t2"), ("t2",)),
            features=torch.randn(2, 5, 8).half(),
            mask=torch.ones(2, 5, dtype=torch.bool),
        )
        write_eval_shard(query_path, {"projection_dim": 8}, eval_queries)
        _metadata, restored_queries = read_eval_shard(query_path)
        assert torch.equal(restored_queries.features, eval_queries.features)

        table_path = os.path.join(directory, "tables_00000.pkl")
        eval_tables = PreparedEvalTables(
            table_ids=("t0", "t1", "t2"),
            cell_features=torch.randn(3 * 4 * 6, 8).half(),
            cell_scatter=torch.arange(3 * 4 * 6, dtype=torch.int32),
            header_features=torch.randn(3 * 4, 8).half(),
            header_scatter=torch.arange(3 * 4, dtype=torch.int32),
            row_mask=torch.ones(3, 6, dtype=torch.bool),
            col_mask=torch.ones(3, 4, dtype=torch.bool),
            cell_mask=torch.ones(3, 4, 6, dtype=torch.bool),
        )
        write_eval_shard(table_path, {"projection_dim": 8}, eval_tables)
        _metadata, restored_tables = read_eval_shard(table_path)
        assert restored_tables.table_ids == eval_tables.table_ids

    ap_sum, rr_sum = _ranking_metrics(
        torch.tensor([[0.9, 0.1, 0.8], [0.2, 0.3, 0.1]]),
        (("t0", "t2"), ("t2",)),
        ["t0", "t1", "t2"],
    )
    assert abs(ap_sum / 2 - 2 / 3) < 1e-6
    assert abs(rr_sum / 2 - 2 / 3) < 1e-6

    prepared = batch.materialize("cpu")
    query_model = PreparedQueryEncoder(8, 8)
    table_model = PreparedTableEncoder(
        embed_dim=8,
        num_layers=1,
        num_heads=2,
        channel_mix_hidden_dim=16,
        table_microbatch_cell_budget=48,
        table_microbatch_max_tables=2,
    )
    scorer = MultiScorer()
    q = query_model(prepared.query_features, prepared.query_mask)
    x = table_model(
        prepared.cell_features,
        prepared.header_features,
        prepared.row_mask,
        prepared.col_mask,
    )
    scores = cross_score_queries_tables(
        scorer,
        "row_match",
        q,
        x,
        prepared.row_mask.float(),
        prepared.col_mask.float(),
    )
    assert scores.shape == (3, 5)
    loss = query_table_info_nce_loss(
        scores, positive_mask=prepared.positive_mask, temperature=0.07
    )
    loss.backward()
    assert table_model.film_content.weight.grad is not None
    assert query_model.proj.weight.grad is not None
    print("prepared-batch serialization/gradient check passed")


if __name__ == "__main__":
    main()
