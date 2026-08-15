"""
Baseline #2 — TABBIE (Iida et al., 2021, "TABBIE: Pretrained Representations
of Tabular Data").

Reference implementation: https://github.com/SFIG611/tabbie (old
`allennlp` + `pytorch-pretrained-bert` codebase). Re-implemented here against
modern `transformers`, keeping the architecture described in the paper.

Header handling (confirmed against the reference repo, not guessed): TABBIE
does **not** fold the header into each cell's text. `preprocess.py` reads
`header, cells = table[0], table[1:]` and indexes the header row completely
separately from the data cells (`indexed_headers` vs `indexed_cells`).
`finetune_table.py::get_tabemb` then concatenates the header embeddings back
in as **row 0** of the table before running the row/column transformer
(`prob_tables_cls[:, 1, 1:, :]` is literally "the header row", `[:, 2:, 1:, :]`
is "the data rows", both indexed the same way after a shared CLS-row/
CLS-col prepend). So the header row gets its own row-CLS embedding and
participates in every column's column-attention exactly like a data row --
it is never mixed into a data cell's own tokens.

We reproduce that here: build an (n_rows + 1) x n_cols grid where row 0 is
the header, run the identical row/column transformer over it, then split
the header row's contextualized row-CLS embedding back out at the end.

Architecture (Sec 3.1-3.2 of the paper):
  1. Each cell (including header cells) is encoded *independently* with
     BERT; the [CLS] output is the initial, uncontextualized cell
     embedding x_ij.
  2. Learned row/column positional embeddings p_i^(r), p_j^(c) are added.
  3. For `num_layers` rounds:
       - a *row* Transformer contextualizes cells within each row
         (a CLS_ROW token is prepended to obtain the row embedding),
       - a *column* Transformer contextualizes cells within each column
         (a CLS_COL token is prepended to obtain the column embedding),
       - the new cell embedding is the average of the row- and
         column-contextualized versions, fed to the next round.
  4. Table embedding = mean of all final row/column CLS embeddings
     (header row's CLS included, since it's a real row here too).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .common import BaseTableEncoder, TableEncoding, clean_cell, validate_table


class TabbieTableEncoder(BaseTableEncoder):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_layers: int = 2,
        num_heads: int = 8,
        max_rows: int = 129,  # +1 to leave room for the header row
        max_cols: int = 64,
        cell_max_length: int = 32,
        device: Optional[str] = None,
    ):
        super().__init__(model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cell_encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.hidden_size = self.cell_encoder.config.hidden_size
        self.cell_max_length = cell_max_length

        # Frozen feature extractor -- see bert_baseline.py's comment for
        # the full rationale. Every cell in the table goes through this
        # backbone independently as one batched call (n_rows*n_cols
        # sequences per table), so this is the single most memory-hungry
        # backbone call in the whole baseline suite; freezing it removes
        # that entire activation graph from backward, leaving only the
        # row/column transformer layers below (self.row_layers/col_layers)
        # trainable, same "train only the layers on top" pattern.
        self.cell_encoder.eval()
        for p in self.cell_encoder.parameters():
            p.requires_grad = False

        # cell text -> raw [CLS] vector (CPU), same pattern as
        # TextEmbedder in cell_encoder.py -- cell_encoder is frozen, so a
        # given cell string always produces the same output, and cells
        # repeat heavily across a real corpus (dates, ids, categorical
        # values, and the same table's cells re-seen every epoch or as a
        # hard negative for many queries). See _encode_cells_isolated.
        self._cell_cache: dict[str, torch.Tensor] = {}

        self.row_pos_embed = nn.Embedding(max_rows, self.hidden_size).to(self.device)
        self.col_pos_embed = nn.Embedding(max_cols, self.hidden_size).to(self.device)
        self.cls_row = nn.Parameter(torch.randn(self.hidden_size) * 0.02).to(self.device)
        self.cls_col = nn.Parameter(torch.randn(self.hidden_size) * 0.02).to(self.device)

        def make_layer():
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=num_heads,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True,
            )
            return enc_layer.to(self.device)

        self.row_layers = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.col_layers = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.num_layers = num_layers

    def train(self, mode: bool = True):
        """Keep the frozen cell_encoder permanently in eval mode -- see
        bert_baseline.py's train() override for the full rationale."""
        super().train(mode)
        self.cell_encoder.eval()
        return self

    def _encode_cells_isolated(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode a flat list of cell texts independently with BERT (batched)
        and return the [CLS] vector as x for each -- matches the reference
        repo's `bert_embedder`, which never lets one cell's tokens attend to
        another cell's tokens at this stage.

        Cache-aware: only strings never seen before in this process
        actually get tokenized and run through cell_encoder -- see
        self._cell_cache's docstring in __init__. Duplicates (including
        repeats within this same call, e.g. a common header text
        appearing once per table) are gathered from the single computed
        result, same as TextEmbedder.forward's approach.
        """
        uncached_unique = list({t for t in texts if t not in self._cell_cache})

        if uncached_unique:
            enc = self.tokenizer(
                uncached_unique,
                padding=True,
                truncation=True,
                max_length=self.cell_max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.cell_encoder(**enc)
            new_cls = out.last_hidden_state[:, 0, :]  # [len(uncached_unique), D]
            for t, v in zip(uncached_unique, new_cls):
                self._cell_cache[t] = v.detach().cpu()

        return torch.stack([self._cell_cache[t] for t in texts], dim=0).to(self.device)

    def forward(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        caption: Optional[str] = None,
    ) -> TableEncoding:
        validate_table(headers, rows)
        n_rows, n_cols = len(rows), len(headers)
        n_rows_total = n_rows + 1  # +1 for the header row

        # row 0 = header cells (their own text, NOT "header : value"); rows 1.. = data cells
        flat_texts = [clean_cell(h) for h in headers]
        for i in range(n_rows):
            flat_texts += [clean_cell(rows[i][j]) for j in range(n_cols)]

        cls = self._encode_cells_isolated(flat_texts)  # [(1+n_rows)*n_cols, D]
        x = cls.view(n_rows_total, n_cols, self.hidden_size)

        row_ids = torch.arange(min(n_rows_total, self.row_pos_embed.num_embeddings), device=self.device)
        col_ids = torch.arange(min(n_cols, self.col_pos_embed.num_embeddings), device=self.device)
        row_pos = self.row_pos_embed(row_ids.clamp(max=self.row_pos_embed.num_embeddings - 1))
        col_pos = self.col_pos_embed(col_ids.clamp(max=self.col_pos_embed.num_embeddings - 1))
        x = x + row_pos.unsqueeze(1) + col_pos.unsqueeze(0)

        row_cls_final = None
        col_cls_final = None
        for layer_idx in range(self.num_layers):
            # --- row transformer: batch = rows (incl. header row), seq = [CLS_ROW, cell_0..cell_{C-1}]
            row_cls_tok = self.cls_row.view(1, 1, -1).expand(n_rows_total, 1, -1)
            row_input = torch.cat([row_cls_tok, x], dim=1)
            row_out = self.row_layers[layer_idx](row_input)
            row_cls_final = row_out[:, 0, :]  # [1+n_rows, D]
            row_ctx_cells = row_out[:, 1:, :]

            # --- column transformer: batch = cols, seq = [CLS_COL, cell_0..cell_{R-1}] (incl. header cell)
            x_by_col = x.transpose(0, 1)  # [C, 1+R, D]
            col_cls_tok = self.cls_col.view(1, 1, -1).expand(n_cols, 1, -1)
            col_input = torch.cat([col_cls_tok, x_by_col], dim=1)
            col_out = self.col_layers[layer_idx](col_input)
            col_cls_final = col_out[:, 0, :]  # [C, D]
            col_ctx_cells = col_out[:, 1:, :].transpose(0, 1)

            x = (row_ctx_cells + col_ctx_cells) / 2.0

        # split header row (index 0) back out from the data rows (indices 1..)
        cell_emb = x[1:]  # [n_rows, n_cols, D] -- matches TableEncoding contract
        row_emb = row_cls_final[1:]  # [n_rows, D], header row excluded
        col_emb = col_cls_final  # [n_cols, D] -- already reflects header influence via col attention
        table_emb = torch.cat([row_cls_final, col_cls_final], dim=0).mean(dim=0)

        return TableEncoding(cell_emb, row_emb, col_emb, table_emb)
