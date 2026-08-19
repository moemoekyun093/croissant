"""128-D table/query towers for fully prepared real-valued features."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.models.table_encoder import TableLayer


class PreparedQueryEncoder(nn.Module):
    """Trainable adapter on fixed random-projected query-token features."""

    def __init__(self, input_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(features)) * mask.unsqueeze(-1)


class PreparedTableEncoder(nn.Module):
    """FiLM + table contextualization with no tokenizer/backbone/lookup."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        channel_mix_hidden_dim: int | None = None,
        nonlinearity: str = "sigmoid",
        table_microbatch_cell_budget: int | None = None,
        table_microbatch_max_tables: int | None = None,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.table_microbatch_cell_budget = table_microbatch_cell_budget
        self.table_microbatch_max_tables = table_microbatch_max_tables
        self.film_content = nn.Linear(embed_dim, embed_dim)
        self.film_gen = nn.Linear(embed_dim, 2 * embed_dim)
        self.layers = nn.ModuleList(
            [
                TableLayer(
                    embed_dim,
                    nonlinearity=nonlinearity,
                    channel_mix_hidden_dim=channel_mix_hidden_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )

    def _contextualize(
        self,
        cells: torch.Tensor,
        headers: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        content = self.film_content(cells)
        gamma, beta = self.film_gen(headers).chunk(2, dim=-1)
        x = (1.0 + gamma.unsqueeze(2)) * content + beta.unsqueeze(2)
        x = x * col_mask[:, :, None, None] * row_mask[:, None, :, None]
        for layer in self.layers:
            x = layer(x, row_mask.float(), col_mask.float())
        return x

    def forward(
        self,
        cells: torch.Tensor,
        headers: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        budget = self.table_microbatch_cell_budget
        max_tables = self.table_microbatch_max_tables
        bt, global_n, global_m, _ = cells.shape
        if (budget is None and max_tables is None) or bt <= 1:
            return self._contextualize(cells, headers, row_mask, col_mask)

        # True sizes come directly from the already-prepared masks.  Grouping
        # only controls execution; records and supervision remain untouched.
        sizes = [
            (int(col_mask[i].sum().item()), int(row_mask[i].sum().item()))
            for i in range(bt)
        ]
        bins: dict[tuple[int, int], list[int]] = {}
        for i, (n, m) in enumerate(sizes):
            n_bin = 1 << (max(1, n) - 1).bit_length()
            m_bin = 1 << (max(1, m) - 1).bit_length()
            bins.setdefault((n_bin, m_bin), []).append(i)

        groups = []
        for (n_bin, m_bin), indices in sorted(bins.items()):
            group_size = len(indices)
            if budget is not None:
                group_size = min(group_size, max(1, budget // (n_bin * m_bin)))
            if max_tables is not None:
                group_size = min(group_size, max_tables)
            groups.extend(indices[i : i + group_size] for i in range(0, len(indices), group_size))

        restored: list[torch.Tensor | None] = [None] * bt
        for group in groups:
            n = max(sizes[i][0] for i in group)
            m = max(sizes[i][1] for i in group)
            index = torch.tensor(group, device=cells.device)
            gx = self._contextualize(
                cells.index_select(0, index)[:, :n, :m],
                headers.index_select(0, index)[:, :n],
                row_mask.index_select(0, index)[:, :m],
                col_mask.index_select(0, index)[:, :n],
            )
            for local, original in enumerate(group):
                restored[original] = F.pad(
                    gx[local], (0, 0, 0, global_m - m, 0, global_n - n)
                )
        if any(value is None for value in restored):
            raise RuntimeError("prepared table microbatch restoration failed")
        return torch.stack([value for value in restored if value is not None])
