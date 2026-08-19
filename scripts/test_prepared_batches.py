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
from src.models.prepared_table_encoder import PreparedQueryEncoder, PreparedTableEncoder
from src.scoring.multi_score import MultiScorer
from src.training.losses import cross_score_queries_tables, query_table_info_nce_loss


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
