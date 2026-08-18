"""Focused equivalence test for our shape-aware table microbatch path.

Uses a tiny deterministic cell encoder, so it requires PyTorch but no
HuggingFace model/download. Verifies both forward outputs and parameter
gradients against the original all-tables-at-once implementation.

Run:
    python -m scripts.test_ours_table_microbatch
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.table import Column, Table
from src.models.table_encoder import TableEncoder
from src.scoring.multi_score import MultiScorer
from src.training.losses import cross_score_queries_tables


class _DeterministicCellEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def encode_tables_batched(self, tables: list[Table]):
        batch_size = len(tables)
        max_cols = max(t.num_columns for t in tables)
        max_rows = max(t.num_rows for t in tables)
        device = next(self.parameters(), torch.empty(0)).device

        X = torch.zeros(batch_size, max_cols, max_rows, self.embed_dim, device=device)
        H = torch.zeros(batch_size, max_cols, self.embed_dim, device=device)
        col_mask = torch.zeros(batch_size, max_cols, device=device)
        row_mask = torch.zeros(batch_size, max_rows, device=device)
        cell_mask = torch.zeros(batch_size, max_cols, max_rows, device=device)
        basis = torch.arange(self.embed_dim, device=device, dtype=torch.float32) / self.embed_dim

        n_list = []
        for b, table in enumerate(tables):
            n_cols, n_rows = table.num_columns, table.num_rows
            table_value = int(table.table_name.removeprefix("t"))
            n_list.append(n_cols)
            col_mask[b, :n_cols] = 1
            row_mask[b, :n_rows] = 1
            cell_mask[b, :n_cols, :n_rows] = 1
            for col in range(n_cols):
                for row in range(n_rows):
                    X[b, col, row] = basis + 0.1 * table_value + 0.01 * col + 0.001 * row
        return X, H, col_mask, row_mask, cell_mask, n_list


def _table(index: int, n_cols: int, n_rows: int) -> Table:
    return Table(
        table_id=f"db#sep#t{index}",
        table_name=f"t{index}",
        columns=[Column(header=f"c{c}", cells=[str(r) for r in range(n_rows)]) for c in range(n_cols)],
    )


def main() -> None:
    torch.manual_seed(0)
    embed_dim = 16
    model = TableEncoder(
        _DeterministicCellEncoder(embed_dim),
        embed_dim=embed_dim,
        num_layers=2,
        channel_mix_hidden_dim=64,
        num_heads=4,
    )
    model.eval()
    tables = [
        _table(0, 2, 3),
        _table(1, 7, 2),
        _table(2, 3, 9),
        _table(3, 6, 8),
        _table(4, 1, 4),
    ]

    model.table_microbatch_cell_budget = None
    full = model.forward_batch_cellwise(tables)
    full_loss = (full[0] * full[3].unsqueeze(-1)).square().sum()
    full_loss.backward()
    full_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    model.zero_grad(set_to_none=True)
    model.table_microbatch_cell_budget = 32
    micro = model.forward_batch_cellwise(tables)
    micro_loss = (micro[0] * micro[3].unsqueeze(-1)).square().sum()
    micro_loss.backward()
    micro_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    # Padding outputs are deliberately discarded/re-zeroed by the
    # microbatch restore path and are never consumed by the masked scorer.
    # Compare every real cell plus the three masks themselves.
    torch.testing.assert_close(
        full[0] * full[3].unsqueeze(-1),
        micro[0] * micro[3].unsqueeze(-1),
        rtol=1e-5,
        atol=1e-6,
    )
    for full_mask, micro_mask in zip(full[1:], micro[1:]):
        torch.testing.assert_close(full_mask, micro_mask)
    assert full_grads.keys() == micro_grads.keys()
    for name in full_grads:
        torch.testing.assert_close(full_grads[name], micro_grads[name], rtol=2e-4, atol=2e-5)

    assert getattr(model, "_last_table_microbatches", 0) > 1

    # Candidate-score chunking must likewise reproduce the full score
    # matrix and gradients, since InfoNCE is applied only after concat.
    scorer = MultiScorer()
    base_q = torch.randn(4, 6, embed_dim)
    base_x = micro[0].detach()
    row_mask, col_mask = micro[2], micro[1]

    q_full = base_q.clone().requires_grad_(True)
    x_full = base_x.clone().requires_grad_(True)
    scores_full = cross_score_queries_tables(
        scorer, "row_match", q_full, x_full, row_mask, col_mask
    )
    scores_full.square().sum().backward()

    q_chunked = base_q.clone().requires_grad_(True)
    x_chunked = base_x.clone().requires_grad_(True)
    scores_chunked = torch.cat(
        [
            cross_score_queries_tables(
                scorer,
                "row_match",
                q_chunked,
                x_chunked[start : start + 2],
                row_mask[start : start + 2],
                col_mask[start : start + 2],
            )
            for start in range(0, len(tables), 2)
        ],
        dim=1,
    )
    scores_chunked.square().sum().backward()
    torch.testing.assert_close(scores_full, scores_chunked, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(q_full.grad, q_chunked.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(x_full.grad, x_chunked.grad, rtol=1e-5, atol=1e-6)

    print(
        "OK: table microbatch and score-chunk outputs/gradients match full batching; "
        f"used {model._last_table_microbatches} microbatches"
    )


if __name__ == "__main__":
    main()
