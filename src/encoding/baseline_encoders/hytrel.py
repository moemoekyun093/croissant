"""
Baseline #6 — HyTrel (Chen et al., 2023, "HyTrel: Hypergraph-enhanced
Tabular Data Representation Learning", NeurIPS'23).

Reference implementation: https://github.com/awslabs/hypergraph-tabular-lm
(built on `torch_geometric`'s `MessagePassing` + an AllSet/PMA-style
set-attention layer, see `layers.py::AllSetTrans` and `model.py::EncoderLayer`
in that repo). We deliberately re-implement the same two-stage
node<->hyperedge message passing in **plain PyTorch with padded, masked
multi-head attention** instead of pulling in `torch_geometric` as a
dependency — tables are small enough (dozens to low-hundreds of cells) that
dense padded attention per hyperedge is simpler to maintain and just as
fast, and it keeps this package free of PyG's separate wheel/CUDA build.

Hypergraph construction: every cell is a *node*. Three kinds of
*hyperedges* connect nodes: one hyperedge per row (all cells in that row),
one hyperedge per column (all cells in that column), and one hyperedge for
the whole table (every cell) — exactly the three hyperedge types described
in the paper (Fig. 1).

Header handling (confirmed against `data.py::_construct_graph` in the
reference repo, not guessed): headers are **not** folded into cell/node
text at all. Cell nodes only ever carry their own raw value. Instead:
  - each **column hyperedge** is initialized directly from that column's
    header text (`for i, head in enumerate(header): ... wordpieces_xt_all`),
  - each **row hyperedge** is initialized from a fixed special `[ROW]`
    token, identical for every row,
  - the **table hyperedge** is initialized from the table's caption (or a
    "missing caption" placeholder if there is none).
So in HyTrel, "what a hyperedge is" and "what the header says" are the same
thing for columns -- the header supplies the column hyperedge's identity
directly, rather than being copied into every cell in that column. This is
also why the reference repo has no learned per-edge-type embedding: row
vs. column vs. table hyperedges are already distinguished by what text
initializes them (a fixed `[ROW]` token vs. real header text vs. the
caption), so we drop the `edge_type_embed` an earlier version of this file
used and rely on content alone, matching the paper.

Each layer does what the reference `EncoderLayer` does:
  V2E: every hyperedge attention-pools over its member node embeddings
       (a learned seed vector attends to member nodes -> new hyperedge repr)
  fuse: the new hyperedge repr is fused (concat + linear) with its old one
  E2V: every node attention-pools over the hyperedges it belongs to
       (row-, column-, and table-hyperedge) -> new node repr

This message passing is exactly what gives HyTrel its permutation
invariance property: row order and column order never affect the result,
since hyperedge membership is a set, not a sequence.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .common import BaseTableEncoder, TableEncoding, clean_cell, validate_table

_MISSING_CAPTION = "[EMPTY]"  # stand-in for the reference repo's MISSING_CAP_TAG


class _SetAttentionPool(nn.Module):
    """AllSet/PMA-style pooling: a learned seed (query) attends over a
    padded, masked set of member vectors (key/value) to produce one output
    vector per set. Mirrors `AllSetTrans` in the HyTrel reference repo, but
    implemented with dense padding instead of a PyG scatter/message-passing
    op — equivalent result, no torch_geometric dependency.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, seeds: torch.Tensor, members: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # seeds: [N, 1, D] (query per set), members: [N, S, D], key_padding_mask: [N, S] (True = pad)
        attn_out, _ = self.mha(seeds, members, members, key_padding_mask=key_padding_mask)
        x = self.norm1(seeds + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(1)  # [N, D]


class HyTrelTableEncoder(BaseTableEncoder):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_layers: int = 3,
        num_heads: int = 8,
        cell_max_tokens: int = 8,
        device: Optional[str] = None,
    ):
        super().__init__(model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert = AutoModel.from_pretrained(model_name)
        self.token_embed = bert.embeddings.word_embeddings.to(self.device)
        self.hidden_size = bert.config.hidden_size
        self.cell_max_tokens = cell_max_tokens

        # Frozen, same as every other baseline's backbone -- see
        # bert_baseline.py's comment. Just BERT's word-embedding lookup
        # table (already cheap), frozen for consistency with the other
        # baselines; self.v2e/e2v/fuse (the hypergraph message-passing
        # layers) remain fully trainable.
        for p in self.token_embed.parameters():
            p.requires_grad = False

        self.v2e = nn.ModuleList(
            [_SetAttentionPool(self.hidden_size, num_heads).to(self.device) for _ in range(num_layers)]
        )
        self.e2v = nn.ModuleList(
            [_SetAttentionPool(self.hidden_size, num_heads).to(self.device) for _ in range(num_layers)]
        )
        self.fuse = nn.ModuleList(
            [nn.Linear(self.hidden_size * 2, self.hidden_size).to(self.device) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

        # stand-in for the reference repo's literal "[ROW]" vocabulary token:
        # every row hyperedge starts from this same learned vector, since our
        # generic tokenizer has no such token to embed.
        self.row_hyperedge_init = nn.Parameter(torch.randn(self.hidden_size) * 0.02).to(self.device)

    def _mean_pool_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Mean-pool raw (non-contextualized) BERT token embeddings for a
        batch of texts -- mirrors the reference repo's `Embedding.forward`,
        which mean-pools token embeddings directly rather than running a
        full BERT forward pass, for both node and hyperedge initialization."""
        ids_list = [
            self.tokenizer.encode(t, add_special_tokens=False)[: self.cell_max_tokens] or [self.tokenizer.unk_token_id]
            for t in texts
        ]
        maxlen = max(len(x) for x in ids_list)
        ids_t = torch.full((len(ids_list), maxlen), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        for k, ids in enumerate(ids_list):
            ids_t[k, : len(ids)] = torch.tensor(ids, device=self.device)
        mask = (ids_t != self.tokenizer.pad_token_id).unsqueeze(-1).float()
        embeds = self.token_embed(ids_t)  # [N, L, D]
        return (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [N, D]

    def _init_node_embeddings(self, rows: Sequence[Sequence[object]], n_cols: int) -> torch.Tensor:
        """Node (cell) init = mean-pooled token embeddings of the cell's own
        value ONLY -- no header text folded in (see module docstring)."""
        n_rows = len(rows)
        texts = [clean_cell(rows[i][j]) for i in range(n_rows) for j in range(n_cols)]
        pooled = self._mean_pool_texts(texts)
        return pooled.view(n_rows, n_cols, self.hidden_size)

    def _pad_stack(self, vecs: List[torch.Tensor]) -> "tuple[torch.Tensor, torch.Tensor]":
        """Pad a list of [S_i, D] tensors to [N, S_max, D] plus a bool key_padding_mask [N, S_max]."""
        n = len(vecs)
        s_max = max(v.size(0) for v in vecs)
        d = vecs[0].size(-1)
        out = torch.zeros(n, s_max, d, device=self.device)
        mask = torch.ones(n, s_max, dtype=torch.bool, device=self.device)  # True = pad
        for i, v in enumerate(vecs):
            out[i, : v.size(0)] = v
            mask[i, : v.size(0)] = False
        return out, mask

    def forward(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str] = None,
    ) -> TableEncoding:
        validate_table(headers, rows)
        n_rows, n_cols = len(rows), len(headers)

        nodes = self._init_node_embeddings(rows, n_cols)  # [R, C, D]
        flat_nodes = nodes.view(n_rows * n_cols, self.hidden_size)  # node id = i*C + j

        def node_id(i, j):
            return i * n_cols + j

        row_edges = [[node_id(i, j) for j in range(n_cols)] for i in range(n_rows)]
        col_edges = [[node_id(i, j) for i in range(n_rows)] for j in range(n_cols)]
        table_edge = [list(range(n_rows * n_cols))]
        edge_members = row_edges + col_edges + table_edge

        # hyperedge init: content-based, not mean-of-members (see docstring).
        #   row hyperedges (n_rows)   <- shared learned "[ROW]" vector
        #   column hyperedges (n_cols) <- that column's own header text
        #   table hyperedge (1)        <- the caption text (or a placeholder)
        col_header_texts = [clean_cell(h) for h in headers]
        col_edge_init = self._mean_pool_texts(col_header_texts)  # [n_cols, D]
        table_edge_init = self._mean_pool_texts([clean_cell(caption) if caption else _MISSING_CAPTION])  # [1, D]
        row_edge_init = self.row_hyperedge_init.unsqueeze(0).expand(n_rows, -1)  # [n_rows, D]

        edge_init = torch.cat([row_edge_init, col_edge_init, table_edge_init], dim=0)

        node_emb = flat_nodes
        edge_emb = edge_init

        # precompute, for each node, which hyperedges it belongs to (row, col, table)
        # -- EVERY node has exactly 3 incident edges (its row, its column,
        # the whole-table edge), a fixed/regular shape unlike V2E's ragged
        # edge_members -- so this can be one vectorized fancy-index per
        # layer instead of a Python loop over every node (up to
        # n_rows*n_cols, same cell-count scale as the per-cell loops fixed
        # in bert_baseline.py/turl.py/tapas_encoder.py) with no padding
        # needed at all (key_pad_mask2 is always all-False).
        node_edges_t = torch.tensor(
            [[i, n_rows + j, n_rows + n_cols] for i in range(n_rows) for j in range(n_cols)],
            device=self.device,
        )  # [n_nodes, 3]
        no_pad_mask = torch.zeros(node_edges_t.size(0), 3, dtype=torch.bool, device=self.device)

        for layer in range(self.num_layers):
            # ---- V2E: each hyperedge attention-pools over its member nodes ----
            # (genuinely ragged sizes -- row/col/table edges have different
            # member counts -- so padding is unavoidable here; this loop is
            # only O(n_edges) ~= n_rows+n_cols+1, not O(n_rows*n_cols))
            member_vecs = [node_emb[idx] for idx in edge_members]  # list of [S_i, D]
            padded_members, key_pad_mask = self._pad_stack(member_vecs)
            seeds = edge_emb.unsqueeze(1)  # [n_edges, 1, D]
            edge_update = F.relu(self.v2e[layer](seeds, padded_members, key_pad_mask))
            edge_emb = self.fuse[layer](torch.cat([edge_emb, edge_update], dim=-1))
            edge_emb = F.dropout(edge_emb, p=0.1, training=self.training)

            # ---- E2V: each node attention-pools over its incident hyperedges ----
            padded_incident = edge_emb[node_edges_t]  # [n_nodes, 3, D] -- one vectorized indexing op
            seeds2 = node_emb.unsqueeze(1)  # [n_nodes, 1, D]
            node_emb = F.relu(self.e2v[layer](seeds2, padded_incident, no_pad_mask))
            node_emb = F.dropout(node_emb, p=0.1, training=self.training)

        cell_emb = node_emb.view(n_rows, n_cols, self.hidden_size)
        row_emb = edge_emb[:n_rows]
        col_emb = edge_emb[n_rows : n_rows + n_cols]
        table_emb = edge_emb[n_rows + n_cols]

        return TableEncoding(cell_emb, row_emb, col_emb, table_emb)
