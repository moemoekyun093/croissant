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
        trainable:   when False, self.encoder (BERT) is frozen -- same
                     "last-layer feature in, train only the layer(s) on
                     top" pattern applied to every baseline table encoder
                     (see src/encoding/baseline_encoders/*.py). Here
                     "the layer on top" is self.proj, a single Linear
                     from BERT's last_hidden_state to output_dim -- that
                     stays trainable regardless of this flag, since it's
                     never part of self.encoder. When True (the old
                     default), the query tower fully finetunes instead,
                     on the reasoning that finetuning is the one stage
                     with real query supervision to learn from -- see
                     configs/finetune.yaml's query_trainable comment for
                     the current tradeoff/choice between the two.
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
        self.trainable = trainable

        if not trainable:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden_size, output_dim, bias=False)

        # query text -> (raw last_hidden_state [true_len, H], input_ids
        # [true_len]) on CPU, UNPADDED (real tokens only) -- same pattern
        # as every frozen baseline table encoder's cache (see adapter.py's
        # save_table_cache docstring). Only meaningful when trainable is
        # False: self.encoder's output for a given query string never
        # changes then, and the SAME val/test questions get re-encoded
        # from scratch every single validation epoch by _corpus_scores
        # otherwise -- e.g. 40 epochs means the same ~3000 validation
        # questions run through BERT 40 separate times for no reason.
        # input_ids are cached alongside the hidden state (not just the
        # hidden state alone) so exclude_special_tokens can still recompute
        # get_special_tokens_mask correctly on a cache hit. self.proj
        # (always trainable) is re-applied fresh to every query regardless
        # of hit/miss.
        self._encoder_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def train(self, mode: bool = True):
        """When frozen, keep self.encoder permanently in eval mode (no
        dropout) even though Trainer.fit() calls model.train() every
        epoch, which recursively flips every submodule -- same pattern as
        the baseline table encoders' train() overrides. self.proj (the
        only trainable part when frozen) still switches train/eval
        normally via the super().train(mode) call."""
        super().train(mode)
        if not self.trainable:
            self.encoder.eval()
        return self

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

        if not self.trainable:
            # Cache-aware path -- only queries never seen before in this
            # process actually get tokenized and run through self.encoder.
            uncached_unique = list({q for q in queries if q not in self._encoder_cache})
            if uncached_unique:
                enc = self.tokenizer(
                    uncached_unique,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    outputs = self.encoder(**enc)
                attn = enc["attention_mask"]  # [U, Lmax_uncached]
                for i, q in enumerate(uncached_unique):
                    true_len = int(attn[i].sum().item())
                    hidden_i = outputs.last_hidden_state[i, :true_len].detach().cpu()  # [true_len, H]
                    ids_i = enc["input_ids"][i, :true_len].detach().cpu()  # [true_len]
                    self._encoder_cache[q] = (hidden_i, ids_i)

            per_query = [self._encoder_cache[q] for q in queries]  # [(hidden[Li,H], ids[Li]), ...]
            Lmax = max(h.shape[0] for h, _ in per_query)
            B = len(queries)
            hidden_size = per_query[0][0].shape[1]
            pad_id = self.tokenizer.pad_token_id or 0

            hidden_batch = torch.zeros(B, Lmax, hidden_size, device=device)
            ids_batch = torch.full((B, Lmax), pad_id, dtype=torch.long, device=device)
            query_mask = torch.zeros(B, Lmax, device=device)
            for i, (h, ids) in enumerate(per_query):
                L = h.shape[0]
                hidden_batch[i, :L] = h.to(device)
                ids_batch[i, :L] = ids.to(device)
                query_mask[i, :L] = 1.0

            Q = self.proj(hidden_batch)  # [B, Lmax, output_dim] -- self.proj always trainable, applied fresh
            input_ids_for_mask = ids_batch
        else:
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
            input_ids_for_mask = encoded["input_ids"]

        if self.exclude_special_tokens:
            special_mask = torch.tensor(
                [
                    self.tokenizer.get_special_tokens_mask(
                        ids, already_has_special_tokens=True
                    )
                    for ids in input_ids_for_mask.tolist()
                ],
                device=device,
            ).float()
            query_mask = query_mask * (1.0 - special_mask)

        return Q, query_mask

    def save_frozen_cache(self, path: str) -> None:
        """Persists self._encoder_cache (query text -> frozen BERT output)
        to disk -- same torch.save pattern as every other frozen-substep
        cache this session (adapter.py's save_table_cache, tabbie.py/
        strubert.py's save_frozen_cache, cell_encoder.py's
        save_cache_to_disk). No-op (empty dict) if trainable=True, since
        caching is skipped entirely in that case -- a query's encoding
        legitimately changes every step when the query tower itself is
        being finetuned."""
        torch.save(self._encoder_cache, path)

    def load_frozen_cache(self, path: str, merge: bool = True) -> None:
        """Loads a previously-saved query cache. merge=True keeps
        existing in-memory entries on a key collision (see adapter.py's
        load_table_cache for why plain dict.update() would get this
        backwards); merge=False replaces the cache entirely."""
        loaded = torch.load(path, map_location="cpu")
        if merge:
            for k, v in loaded.items():
                self._encoder_cache.setdefault(k, v)
        else:
            self._encoder_cache = loaded
