"""Small CPU check for prepared serialization, labels, gradients, and shapes."""

from __future__ import annotations

import os
import tempfile

import torch

from scripts.prepare_fixed_batches import build_train_epochs
from scripts.prepare_streaming_eval import _build_candidate_plan
from src.data.prepared_batches import (
    PreparedBatch,
    PreparedBatchWriter,
    iter_prepared_batches,
    prefetch_iterable,
    read_prepared_metadata,
)
from src.data.prepared_eval import (
    PreparedEvalQueries,
    PreparedStreamingEvalQueries,
    PreparedEvalTables,
    read_eval_shard,
    write_eval_shard,
)
from src.models.prepared_table_encoder import (
    PreparedQueryEncoder,
    PreparedTabbieEncoder,
    PreparedTableEncoder,
    PreparedTurlEncoder,
)
from src.scoring.multi_score import MultiScorer
from src.training.losses import cross_score_queries_tables, query_table_info_nce_loss
from src.training.prepared_evaluator import (
    _ranking_metrics,
    evaluate_prepared,
    evaluate_prepared_streaming,
)


def main() -> None:
    torch.manual_seed(7)
    table_ids = ["d0#sep#t0", "d0#sep#t1", "d1#sep#t0", "d2#sep#t0"]
    table_index = {table_id: index for index, table_id in enumerate(table_ids)}
    database_tables = {"d0": [0, 1], "d1": [2], "d2": [3]}
    candidate_args = (
        [("q0", "d0", ("d0#sep#t1",)), ("q1", "d1", ("d1#sep#t0",))],
        database_tables,
        table_ids,
        table_index,
        1,
        42,
        0,
    )
    plan_a = _build_candidate_plan(*candidate_args)
    plan_b = _build_candidate_plan(*candidate_args)
    assert all(torch.equal(left, right) for left, right in zip(plan_a, plan_b))
    candidate_indices, visible_mask, positive_mask = plan_a
    assert {0, 1, 2}.issubset(set(candidate_indices.tolist()))
    assert torch.all(positive_mask.any(dim=1))
    assert not torch.any(positive_mask & ~visible_mask)

    legacy_epochs = build_train_epochs(list(range(10)), 4, None, seed=7)
    assert [len(epoch) for epoch in legacy_epochs] == [4, 4, 2]
    assert sorted(index for epoch in legacy_epochs for index in epoch) == list(range(10))
    repeated_epochs = build_train_epochs(list(range(10)), 20, 3, seed=7)
    assert len(repeated_epochs) == 3
    assert all(sorted(epoch) == list(range(10)) for epoch in repeated_epochs)
    assert repeated_epochs == build_train_epochs(list(range(10)), 20, 3, seed=7)
    assert len({tuple(epoch) for epoch in repeated_epochs}) > 1
    assert list(prefetch_iterable(range(20), depth=2)) == list(range(20))

    def failing_stream():
        yield 1
        raise RuntimeError("prefetch propagation check")

    prefetched = iter(prefetch_iterable(failing_stream(), depth=2))
    assert next(prefetched) == 1
    try:
        next(prefetched)
    except RuntimeError as error:
        assert str(error) == "prefetch propagation check"
    else:
        raise AssertionError("prefetch worker exception was not propagated")

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

    turl = PreparedTurlEncoder(
        embed_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_hidden_dim=16,
        attention_budget=2_000,
    )
    turl_output = turl(
        prepared.cell_features,
        prepared.header_features,
        prepared.row_mask,
        prepared.col_mask,
    )
    assert turl_output.shape == prepared.cell_features.shape
    turl_output.sum().backward()
    assert turl.layers[0].linear1.weight.grad is not None

    tabbie = PreparedTabbieEncoder(
        embed_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_hidden_dim=16,
        max_rows=7,
        max_columns=4,
        table_microbatch_cell_budget=48,
        table_microbatch_max_tables=2,
    )
    tabbie_output = tabbie(
        prepared.cell_features,
        prepared.header_features,
        prepared.row_mask,
        prepared.col_mask,
    )
    assert tabbie_output.shape == prepared.cell_features.shape
    tabbie_output.sum().backward()
    assert tabbie.row_layers[0].linear1.weight.grad is not None

    # Size-sorted TABBIE microbatching must be an execution-only change:
    # mixed table shapes are restored in candidate order with the same
    # values as a single padded contextualization call.
    mixed_rows = prepared.row_mask.clone()
    mixed_columns = prepared.col_mask.clone()
    mixed_rows[1, 4:] = False
    mixed_rows[3, 2:] = False
    mixed_columns[2, 3:] = False
    mixed_columns[3, 2:] = False
    tabbie.eval()
    grouped_output = tabbie(
        prepared.cell_features,
        prepared.header_features,
        mixed_rows,
        mixed_columns,
    )
    tabbie.table_microbatch_cell_budget = None
    tabbie.table_microbatch_max_tables = None
    full_output = tabbie(
        prepared.cell_features,
        prepared.header_features,
        mixed_rows,
        mixed_columns,
    )
    assert torch.allclose(grouped_output, full_output, atol=1e-5, rtol=1e-4)

    # End-to-end check for the table-major evaluator: query and table
    # shards are each loaded once, yet the complete [Nq,Nt] ranking and
    # multi-positive metrics are produced.
    with tempfile.TemporaryDirectory() as directory:
        metadata = {"projection_dim": 8}
        write_eval_shard(
            os.path.join(directory, "queries_00000.pkl"), metadata, eval_queries
        )
        write_eval_shard(
            os.path.join(directory, "tables_00000.pkl"), metadata, eval_tables
        )
        with open(
            os.path.join(directory, "PREPARATION_COMPLETE"),
            "w",
            encoding="utf-8",
        ) as marker:
            marker.write("complete\n")
        metrics = evaluate_prepared(
            directory,
            table_model,
            query_model,
            scorer,
            "cpu",
            metadata,
            query_batch_size=1,
            progress=None,
        )
        assert metrics["n_queries"] == 2
        assert metrics["n_tables"] == 3
        assert 0.0 <= metrics["map"] <= 1.0
        assert 0.0 <= metrics["mrr"] <= 1.0

    # Full-test streaming mode stores each table once and freezes only global
    # candidate indices plus visibility/relevance decisions per query chunk.
    # Exercise a nontrivial candidate permutation so indexed table gathering
    # and score-column placement are both covered.
    with tempfile.TemporaryDirectory() as directory:
        metadata = {
            "projection_dim": 8,
            "projection_seed": 7,
            "model_name": "dummy",
            "max_rows": 6,
            "max_columns": 4,
            "split_sha256": "split",
            "questions_sha256": "questions",
            "evaluation_mode": "streaming_per_query_candidates",
            "table_shard_size": 3,
            "n_tables": 3,
            "n_queries": 2,
            "n_distractors": 2,
        }
        streaming_queries = PreparedStreamingEvalQueries(
            query_texts=("q0", "q1"),
            gold_table_ids=(("t0", "t2"), ("t1",)),
            features=eval_queries.features,
            mask=eval_queries.mask,
            candidate_table_indices=torch.tensor([2, 0, 1], dtype=torch.int32),
            visible_mask=torch.tensor(
                [[1, 1, 1], [1, 0, 1]], dtype=torch.bool
            ),
            positive_mask=torch.tensor(
                [[1, 1, 0], [0, 0, 1]], dtype=torch.bool
            ),
        )
        write_eval_shard(
            os.path.join(directory, "query_chunks_00000.pkl"),
            metadata,
            streaming_queries,
        )
        write_eval_shard(
            os.path.join(directory, "tables_00000.pkl"), metadata, eval_tables
        )
        with open(
            os.path.join(directory, "manifest.json"), "w", encoding="utf-8"
        ) as manifest:
            import json

            json.dump(metadata, manifest)
        with open(
            os.path.join(directory, "PREPARATION_COMPLETE"),
            "w",
            encoding="utf-8",
        ) as marker:
            marker.write("complete\n")
        streaming_metrics = evaluate_prepared_streaming(
            directory,
            table_model,
            query_model,
            scorer,
            "cpu",
            metadata,
            query_batch_size=1,
            table_batch_size=2,
            progress=None,
        )
        assert streaming_metrics["n_queries"] == 2
        assert streaming_metrics["n_tables"] == 3
        assert streaming_metrics["n_query_chunks"] == 1
        assert streaming_metrics["mean_candidate_pool"] == 3
        assert 0.0 <= streaming_metrics["map"] <= 1.0
        assert 0.0 <= streaming_metrics["mrr"] <= 1.0
    print("prepared-batch serialization/gradient check passed")


if __name__ == "__main__":
    main()
