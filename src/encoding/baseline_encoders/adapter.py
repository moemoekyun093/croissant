"""
Adapter making any BaseTableEncoder subclass (bert/tabbie/strubert/tapas/
turl/hytrel, all in this package) expose the SAME
forward_batch_cellwise(tables) -> (X, col_mask, row_mask, cell_mask)
interface our own TableEncoder (src/models/table_encoder.py) does. This
is what lets PretrainTrainer/FinetuneTrainer (src/training/trainer.py)
train ANY baseline through the exact same code path as our own model --
same ELECTRA corruption/discriminator loss, same MAP-based early
stopping, same everything -- with only the encoder swapped out. "Same
training paradigm across all models" means the same trainer classes
actually run unmodified against a baseline, not just a similar-looking
reimplementation per baseline.

Baseline encoders operate on ONE table at a time (headers: list[str],
rows: list[list[str]], row-major -- see common.py's TableEncoding
contract) and return cell_embeddings shaped [n_rows, n_cols, native_dim].
This adapter loops over a batch of our own Table objects (src/data/
table.py), converts each to that headers/rows shape, transposes
cell_embeddings to our own [n_cols, n_rows, dim] (column-major)
convention, pads across the batch the same way CellEncoder.
encode_tables_batched does, and optionally projects native_dim to a
configured embed_dim so every baseline (and our own model) can be run
at the SAME internal dimension -- required for the "consistency in
model parameters (internal dimensions, number of epochs, etc.)" a fair
comparison needs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.table import Table
from src.encoding.baseline_encoders.common import BaseTableEncoder


class BaselineCellwiseAdapter(nn.Module):
    def __init__(
        self,
        baseline_encoder: BaseTableEncoder,
        embed_dim: int | None = None,
        cacheable: bool = False,
    ):
        """
        baseline_encoder: any BaseTableEncoder subclass instance
                          (already constructed) -- this adapter doesn't
                          know or care which paper it implements.
        embed_dim:        if given and different from the baseline's own
                          native hidden size, a trainable Linear
                          projects every cell embedding down/up to this
                          width. If None, uses the baseline's native
                          dimension unprojected (no cross-model
                          dimension consistency in that case).
        cacheable:        True ONLY for baselines with zero trainable
                          parameters of their own (bert/tapas -- the
                          frozen backbone IS the whole model for those
                          two, see build_baseline_model's docstring).
                          When True, baseline_encoder(headers, rows)'s
                          raw output is cached per table_id -- since
                          nothing in that computation ever changes
                          during training, a table seen again (extremely
                          common: hard negatives get reused across many
                          queries in the same db, and every table is
                          re-seen every epoch) skips tokenization AND
                          the backbone forward AND cell-pooling entirely,
                          just a dict lookup. Do NOT set this for
                          baselines with their own trainable stack
                          (tabbie/strubert/turl/hytrel) -- their output
                          changes every step as those layers train, so
                          caching it would silently serve stale,
                          increasingly-wrong embeddings.
        """
        super().__init__()
        self.baseline_encoder = baseline_encoder
        native_dim = self._infer_native_dim(baseline_encoder)
        self.embed_dim = embed_dim if embed_dim is not None else native_dim
        self.cacheable = cacheable
        self._table_cache: dict[str, torch.Tensor] = {}  # table_id -> raw [n_cols, n_rows, native_dim], CPU

        self.projection = (
            nn.Linear(native_dim, self.embed_dim) if self.embed_dim != native_dim else nn.Identity()
        )

    @staticmethod
    def _infer_native_dim(baseline_encoder: BaseTableEncoder) -> int:
        # every current baseline (bert/tabbie/strubert/tapas/turl/hytrel)
        # exposes .hidden_size (all BERT-backed) -- fail loudly rather
        # than silently guessing if a future baseline doesn't.
        if hasattr(baseline_encoder, "hidden_size"):
            return baseline_encoder.hidden_size
        raise AttributeError(
            f"{type(baseline_encoder).__name__} has no .hidden_size -- "
            "BaselineCellwiseAdapter needs to know the native embedding "
            "width up front to size its projection layer."
        )

    @staticmethod
    def _table_to_headers_rows(table: Table) -> tuple[list[str], list[list[str]]]:
        headers = [col.header for col in table.columns]
        n_rows = table.num_rows
        rows = [[col.cells[r] for col in table.columns] for r in range(n_rows)]
        return headers, rows

    def forward_batch_cellwise(
        self, tables: list[Table], ablation: str | None = None, profile: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same contract as TableEncoder.forward_batch_cellwise. ablation
        isn't supported here -- baselines don't share our
        concat(cell, header)-then-fuse scheme, so there's nothing
        analogous to isolate. profile is accepted for interface parity
        but currently a no-op (baselines don't have a comparable
        cell-encoding-vs-table-layers split to time separately)."""
        if ablation is not None:
            raise NotImplementedError(
                f"ablation={ablation!r} is not supported by baseline encoders"
            )

        device = next(self.parameters()).device

        per_table_cell: list = [None] * len(tables)  # each [n_cols_i, n_rows_i, embed_dim]

        cache_hit_indices = []
        cache_miss_indices = []
        for i, table in enumerate(tables):
            if self.cacheable and table.table_id in self._table_cache:
                cache_hit_indices.append(i)
            else:
                cache_miss_indices.append(i)

        # Cache hits: skip tokenization + backbone forward + cell-pooling
        # entirely -- see cacheable's docstring. Cached in raw
        # (pre-projection) form on CPU, same convention as TextEmbedder's
        # cache, so self.projection (trainable, though typically Identity
        # for bert/tapas when embed_dim matches native_dim) is still
        # re-applied fresh every call.
        for i in cache_hit_indices:
            raw_cell = self._table_cache[tables[i].table_id].to(device)
            per_table_cell[i] = self.projection(raw_cell)

        # Cache misses: use the baseline's forward_batch (ONE backbone
        # forward pass for every miss table at once) when it exposes one
        # -- currently bert_baseline.py, tapas_encoder.py, tabbie.py, and
        # strubert.py, see their forward_batch docstrings for the full
        # "why this matters" rationale (this used to call
        # self.baseline_encoder(headers, rows) once PER TABLE here, i.e.
        # one batch-of-1 (or, for strubert, batch-of-1 TWICE) backbone
        # forward pass per table -- the single biggest reason those
        # baselines were slower than 'ours', which always batches every
        # cell across the whole table batch into one call). turl.py and
        # hytrel.py deliberately do NOT implement forward_batch: their
        # expensive part isn't a frozen-backbone forward call at all
        # (self.token_embed is a cheap nn.Embedding lookup for both) --
        # it's their OWN trainable per-table architecture (turl's
        # visibility-masked attention stack, sized by that table's own
        # n_rows*n_cols visibility matrix; hytrel's hypergraph message
        # passing, sized by that table's own row/column/table hyperedges)
        # that's inherently table-shape-dependent, not something a
        # "concatenate + one combined call + slice back out" fix like the
        # other four applies to without a much larger restructuring
        # (padding every table's structure to a shared max shape and
        # batching the whole trainable stack itself). They fall back to
        # the original per-table forward() loop, unchanged.
        if cache_miss_indices:
            if hasattr(self.baseline_encoder, "forward_batch"):
                batch_inputs = [
                    (*self._table_to_headers_rows(tables[i]), None) for i in cache_miss_indices
                ]
                encodings = self.baseline_encoder.forward_batch(batch_inputs)
                for miss_pos, i in enumerate(cache_miss_indices):
                    encoding = encodings[miss_pos]
                    raw_cell = encoding.cell_embeddings.transpose(0, 1)
                    if self.cacheable:
                        self._table_cache[tables[i].table_id] = raw_cell.detach().cpu()
                    per_table_cell[i] = self.projection(raw_cell)
            else:
                for i in cache_miss_indices:
                    headers, rows = self._table_to_headers_rows(tables[i])
                    encoding = self.baseline_encoder(headers, rows)
                    # encoding.cell_embeddings: [n_rows, n_cols, native_dim]
                    # (row-major, per common.py) -> [n_cols, n_rows, native_dim]
                    # (column-major, our convention)
                    raw_cell = encoding.cell_embeddings.transpose(0, 1)
                    if self.cacheable:
                        self._table_cache[tables[i].table_id] = raw_cell.detach().cpu()
                    per_table_cell[i] = self.projection(raw_cell)

        B = len(tables)
        max_n = max((c.shape[0] for c in per_table_cell), default=1)
        max_m = max((c.shape[1] for c in per_table_cell), default=1)

        X = torch.zeros(B, max_n, max_m, self.embed_dim, device=device)
        col_mask = torch.zeros(B, max_n, device=device)
        row_mask = torch.zeros(B, max_m, device=device)
        cell_mask = torch.zeros(B, max_n, max_m, device=device)

        # Walking table.columns/col.cells to find which cells are
        # non-null is an irreducible Python-level loop over raw strings
        # (same as cell_encoder.py's encode_tables_batched -- see its
        # docstring) -- but the actual mask WRITE doesn't need to be:
        # previously this did `cell_mask[b, c_idx, r_idx] = 1.0` inside
        # the loop itself, i.e. one single-element GPU tensor write per
        # non-null cell (up to n_rows*n_cols of them, PER TABLE, every
        # single training/validation step, for every one of the 6
        # baseline encoders that share this adapter -- not something
        # specific to any one baseline). Collecting positions in plain
        # Python lists (cheap) and scattering ONCE via advanced indexing
        # at the end is the same fix cell_encoder.py already uses for
        # its own nonnull cell_mask.
        nonnull_b: list[int] = []
        nonnull_c: list[int] = []
        nonnull_r: list[int] = []

        for b, (table, cell) in enumerate(zip(tables, per_table_cell)):
            n, m = cell.shape[0], cell.shape[1]
            X[b, :n, :m, :] = cell
            col_mask[b, :n] = 1.0
            row_mask[b, :m] = 1.0
            for c_idx, col in enumerate(table.columns):
                for r_idx, val in enumerate(col.cells):
                    if val.strip() != "":
                        nonnull_b.append(b)
                        nonnull_c.append(c_idx)
                        nonnull_r.append(r_idx)

        if nonnull_b:
            bt = torch.tensor(nonnull_b, device=device)
            ct = torch.tensor(nonnull_c, device=device)
            rt = torch.tensor(nonnull_r, device=device)
            cell_mask[bt, ct, rt] = 1.0

        return X, col_mask, row_mask, cell_mask



# Every baseline encodes cells/tokens with a FULL pretrained BERT (or
# TAPAS) backbone -- none of them expose a way to truncate that backbone
# itself, so its depth (e.g. 12 layers for bert-base-uncased) is fixed
# and identical across every baseline regardless of num_layers, same as
# it's fixed (frozen) for our own CellEncoder. What num_layers actually
# controls, for both "ours" and these baselines, is the TABLE-LEVEL
# stack built ON TOP of that frozen per-cell/per-token encoding -- our
# RCPE table layers, TABBIE's row/col transformer layers, TURL's
# visibility-masked encoder layers, HyTrel's set-attention-pool layers.
# That's the apples-to-apples "same number of layers" comparison across
# models. bert/tapas have no such on-top stack at all -- per their own
# papers, the pretrained backbone itself IS the whole model, with no
# additional table-level layers to speak of -- so num_layers has nothing
# to apply to for those two; that's a genuine architectural difference
# between papers, not an oversight, and forcing a fake stack onto them
# to "use" the setting would misrepresent what those papers actually do.
_NUM_LAYERS_KWARG = {
    "tabbie": "num_layers",
    "strubert": "num_attn_layers",  # same concept, different constructor kwarg name
    "turl": "num_layers",
    "hytrel": "num_layers",
    # "bert" and "tapas" deliberately absent -- see comment above.
}

# Same "bert/tapas have no on-top trainable stack" fact from above also
# means their entire per-table output is deterministic given the same
# input -- see BaselineCellwiseAdapter's cacheable docstring. Every other
# baseline has its own trainable layers whose output changes every
# training step, so caching their full per-table output would be wrong
# (silently stale). Kept as a separate set from _NUM_LAYERS_KWARG's
# absence rather than reusing that as a proxy, in case a future baseline
# ever has an on-top stack that's ALSO frozen (deliberately not the same
# condition, even though they happen to coincide for bert/tapas today).
_FULLY_FROZEN_ENCODERS = {"bert", "tapas"}


def build_baseline_model(
    encoder_name: str,
    embed_dim: int,
    model_name: str | None = None,
    num_layers: int | None = None,
    device: str | None = None,
) -> BaselineCellwiseAdapter:
    """Convenience factory -- build any registered baseline by name
    (see ENCODER_REGISTRY in this package's __init__.py) already wrapped
    in BaselineCellwiseAdapter, ready to hand to PretrainTrainer/
    FinetuneTrainer exactly like our own TableEncoder.

    model_name: left as None by default so each baseline uses ITS OWN
    class default -- NOT forced to a shared checkpoint. This matters
    concretely for TAPAS, whose default is "google/tapas-base" (its own
    checkpoint family, with row/column-id embeddings a plain BERT
    checkpoint doesn't have) -- passing a generic "bert-base-uncased" to
    it would silently produce a broken/mismatched model. "Consistency in
    model parameters" (per-instruction) means embed_dim/epochs/etc.
    matching across models, not forcing architecturally-incompatible
    baselines to share one literal checkpoint identity. Only pass
    model_name explicitly if you specifically want to override a given
    baseline's own default (e.g. to a different BERT variant for the
    BERT-backed ones).

    num_layers: forwarded to whichever baselines expose a comparable
    on-top table-level stack depth (see _NUM_LAYERS_KWARG above) --
    silently ignored (baseline keeps its own class default) for bert/
    tapas, which have no such stack. Pass the SAME value used for
    --encoder ours to keep that one architectural axis consistent across
    every model that actually has it.
    """
    from src.encoding.baseline_encoders import ENCODER_REGISTRY

    if encoder_name not in ENCODER_REGISTRY:
        raise ValueError(
            f"unknown baseline encoder {encoder_name!r} -- choices: "
            f"{sorted(ENCODER_REGISTRY)}"
        )

    encoder_cls = ENCODER_REGISTRY[encoder_name]
    kwargs = {"device": device}
    if model_name is not None:
        kwargs["model_name"] = model_name
    if num_layers is not None:
        layer_kwarg = _NUM_LAYERS_KWARG.get(encoder_name)
        if layer_kwarg is not None:
            kwargs[layer_kwarg] = num_layers
    baseline_encoder = encoder_cls(**kwargs)

    adapter = BaselineCellwiseAdapter(
        baseline_encoder,
        embed_dim=embed_dim,
        cacheable=encoder_name in _FULLY_FROZEN_ENCODERS,
    )
    if device is not None:
        adapter = adapter.to(device)
    return adapter
