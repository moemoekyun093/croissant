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


class _PreparedTurlLayer(nn.Module):
    """Visibility-masked Transformer layer operating entirely at 128-D."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.linear1 = nn.Linear(embed_dim, ffn_hidden_dim)
        self.linear2 = nn.Linear(ffn_hidden_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            x, x, x, attn_mask=blocked, need_weights=False
        )
        x = self.norm1(x + self.dropout(attended))
        feedforward = self.linear2(F.gelu(self.linear1(x)))
        return self.norm2(x + self.dropout(feedforward))


class PreparedTurlEncoder(nn.Module):
    """TURL visibility attention over already-pooled 128-D cell nodes.

    The prepared file has already performed frozen text embedding, pooling
    to one vector per header/cell, and the fixed 768->128 projection.  This
    module therefore contains only TURL's trainable structural
    contextualizer.  Headers are column-local nodes, cells see their row and
    column, and one learned global node is visible to every valid node.
    Header/global nodes are discarded before retrieval scoring.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_hidden_dim: int = 512,
        attention_budget: int = 2_000_000,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if attention_budget <= 0:
            raise ValueError("attention_budget must be positive")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_budget = attention_budget
        self.global_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.global_token, std=0.02)
        self.layers = nn.ModuleList(
            [
                _PreparedTurlLayer(embed_dim, num_heads, ffn_hidden_dim)
                for _ in range(num_layers)
            ]
        )

    def _contextualize_group(
        self,
        cells: torch.Tensor,
        headers: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_cols, n_rows, dim = cells.shape
        device = cells.device

        # Node order: one global node, N header nodes, then N*M cell nodes
        # in the same column-major order used by PreparedBatch.
        global_node = self.global_token.expand(batch_size, -1, -1)
        x = torch.cat(
            [global_node, headers, cells.reshape(batch_size, n_cols * n_rows, dim)],
            dim=1,
        )

        cell_valid = (col_mask[:, :, None] & row_mask[:, None, :]).reshape(
            batch_size, n_cols * n_rows
        )
        valid = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=torch.bool, device=device),
                col_mask,
                cell_valid,
            ],
            dim=1,
        )

        # Headers receive distinct negative row IDs, making each header
        # visible only globally, to itself, and to cells in its column.
        node_rows = torch.cat(
            [
                torch.tensor([-1], device=device),
                -torch.arange(2, n_cols + 2, device=device),
                torch.arange(n_rows, device=device).repeat(n_cols),
            ]
        )
        node_cols = torch.cat(
            [
                torch.tensor([-1], device=device),
                torch.arange(n_cols, device=device),
                torch.arange(n_cols, device=device).repeat_interleave(n_rows),
            ]
        )
        same_row = node_rows[:, None] == node_rows[None, :]
        same_col = node_cols[:, None] == node_cols[None, :]
        global_visible = (node_rows[:, None] == -1) | (node_rows[None, :] == -1)
        structural = same_row | same_col | global_visible

        visible = structural.unsqueeze(0) & valid[:, :, None] & valid[:, None, :]
        # Padded queries must see themselves to prevent all-masked softmax
        # rows. Their outputs are discarded immediately after attention.
        identity = torch.eye(x.shape[1], dtype=torch.bool, device=device).unsqueeze(0)
        visible = visible | (identity & ~valid[:, :, None])
        blocked = (~visible).repeat_interleave(self.num_heads, dim=0)

        for layer in self.layers:
            x = layer(x, blocked)
        contextualized = x[:, 1 + n_cols :].reshape(
            batch_size, n_cols, n_rows, dim
        )
        return contextualized * cell_valid.reshape(
            batch_size, n_cols, n_rows, 1
        )

    def forward(
        self,
        cells: torch.Tensor,
        headers: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        bt, global_n, global_m, _ = cells.shape
        sizes = [
            (int(col_mask[i].sum().item()), int(row_mask[i].sum().item()))
            for i in range(bt)
        ]
        order = sorted(
            range(bt), key=lambda i: 1 + sizes[i][0] + sizes[i][0] * sizes[i][1]
        )
        groups: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for index in order:
            n_cols, n_rows = sizes[index]
            sequence_length = 1 + n_cols + n_cols * n_rows
            proposed_max = max(current_max, sequence_length)
            if current and (len(current) + 1) * proposed_max**2 > self.attention_budget:
                groups.append(current)
                current = []
                current_max = 0
            current.append(index)
            current_max = max(current_max, sequence_length)
        if current:
            groups.append(current)

        restored: list[torch.Tensor | None] = [None] * bt
        for group in groups:
            n_cols = max(sizes[i][0] for i in group)
            n_rows = max(sizes[i][1] for i in group)
            indices = torch.tensor(group, device=cells.device)
            contextualized = self._contextualize_group(
                cells.index_select(0, indices)[:, :n_cols, :n_rows],
                headers.index_select(0, indices)[:, :n_cols],
                row_mask.index_select(0, indices)[:, :n_rows],
                col_mask.index_select(0, indices)[:, :n_cols],
            )
            for local, original in enumerate(group):
                restored[original] = F.pad(
                    contextualized[local],
                    (0, 0, 0, global_m - n_rows, 0, global_n - n_cols),
                )
        if any(value is None for value in restored):
            raise RuntimeError("prepared TURL microbatch restoration failed")
        return torch.stack([value for value in restored if value is not None])
