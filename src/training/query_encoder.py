"""
Query encoder for finetuning: turns a natural-language query into a set
of L vectors (one per token, ColBERT-style multi-vector), for use as the
`Q` input to src/scoring/multi_score.py's MultiScorer.

Deliberately NOT built on top of CellEncoder's TextEmbedder: a query has
no "column header" to concatenate with, and CellEncoder's TextEmbedder
projects to HALF of embed_dim specifically so concatenating cell+header
lands on the full width. A query vector needs to be directly comparable
(dot product) against a table's already-fused, full-width cell
embeddings, so this projects straight to the FULL embed_dim instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class QueryEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_dim: int,
        max_length: int = 32,
        trainable: bool = True,
        exclude_special_tokens: bool = False,
    ):
        """
        model_name:  a BERT-family checkpoint -- can be the same
                     `text_model_name` used by CellEncoder (independent
                     weights either way; this module owns its own BERT
                     instance, not shared with CellEncoder's).
        output_dim:  FULL embed_dim (matches TableEncoder's per-cell
                     width directly -- no halving, unlike CellEncoder's
                     text_dim).
        trainable:   unlike CellEncoder's frozen-BERT default, the query
                     tower defaults to trainable -- finetuning is exactly
                     the stage where the query side has real supervision
                     (query -> positive table) to learn from, whereas
                     CellEncoder's BERT stays frozen throughout to avoid
                     confounding a novel, unvalidated table architecture
                     with a drifting text backbone (see
                     src/training/training_config.md). Set False to keep
                     it frozen too, if that confound is a concern here
                     as well.
        exclude_special_tokens: if True, [CLS]/[SEP] are masked out of
                     the returned query_mask (only real word-piece
                     tokens count as query vectors). Off by default --
                     ColBERT's own convention keeps them as ordinary
                     vectors.
        """
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.output_dim = output_dim
        self.exclude_special_tokens = exclude_special_tokens

        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden_size, output_dim, bias=False)

    def forward(self, queries: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        queries: list of B raw query strings

        returns:
            Q:          [B, L, output_dim] -- one vector per token
                        (L = the batch's own max token count, padded)
            query_mask: [B, L] -- 1 for real tokens, 0 for padding (and
                        for [CLS]/[SEP] too, if exclude_special_tokens)
        """
        if len(queries) == 0:
            return (
                torch.empty(0, 0, self.output_dim),
                torch.empty(0, 0),
            )

        device = next(self.encoder.parameters()).device

        encoded = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        outputs = self.encoder(**encoded)
        Q = self.proj(outputs.last_hidden_state)  # [B, L, output_dim]

        query_mask = encoded["attention_mask"].float()  # [B, L]

        if self.exclude_special_tokens:
            special_mask = torch.tensor(
                [
                    self.tokenizer.get_special_tokens_mask(
                        ids, already_has_special_tokens=True
                    )
                    for ids in encoded["input_ids"].tolist()
                ],
                device=device,
            ).float()
            query_mask = query_mask * (1.0 - special_mask)

        return Q, query_mask
