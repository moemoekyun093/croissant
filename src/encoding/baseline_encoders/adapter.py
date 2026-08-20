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
from src.encoding.cache_utils import downcast_cache, upcast_cache


class BaselineCellwiseAdapter(nn.Module):
    def __init__(
        self,
        baseline_encoder: BaseTableEncoder,
        embed_dim: int | None = None,
        cacheable: bool = False,
        table_microbatch_cell_budget: int | None = None,
        table_microbatch_max_tables: int | None = None,
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
        for name, value in (
            ("table_microbatch_cell_budget", table_microbatch_cell_budget),
            ("table_microbatch_max_tables", table_microbatch_max_tables),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")
        self.table_microbatch_cell_budget = table_microbatch_cell_budget
        self.table_microbatch_max_tables = table_microbatch_max_tables
        self._table_cache: dict[str, torch.Tensor] = {}  # table_id -> raw [n_cols, n_rows, native_dim], CPU

        self.projection = (
            nn.Linear(native_dim, self.embed_dim) if self.embed_dim != native_dim else nn.Identity()
        )

    def save_table_cache(self, path: str) -> None:
        """Persists the current in-memory _table_cache to disk -- same
        rationale and pattern as TextEmbedder.save_cache_to_disk (see
        cell_encoder.py). Only meaningful when self.cacheable is True
        (bert/tapas): for a fully-frozen encoder, a table's raw cell
        embeddings are a deterministic function of its text and the
        (frozen, never-updated) backbone weights -- they never change
        across the WHOLE run, or even across separate runs with the same
        encoder/model_name/tokenization config. Right now that
        determinism is only exploited within one process's lifetime
        (this dict dies when the process exits); persisting it means the
        corpus's embeddings, once computed, never need to be computed
        again by any future run -- no re-tokenizing, no backbone forward
        pass, just a dict load. A no-op (empty file) if cacheable=False
        or nothing has been encoded yet.

        NOTE: keyed only by table_id -- if you ever change --model_name,
        --max_length, or the cell-serialization convention
        (_serialize_cell in bert_baseline.py etc.) for the SAME encoder,
        delete the cache file first, since old entries would otherwise
        be silently reused despite no longer matching what a fresh
        encode would produce.
        """
        torch.save(downcast_cache(self._table_cache), path)

    def load_table_cache(self, path: str, merge: bool = True) -> None:
        """Loads a previously-saved _table_cache from disk. merge=True
        adds to whatever's already in memory, keeping any EXISTING
        in-memory entry on a key collision rather than the loaded one
        (plain dict.update() would do the opposite -- loaded values
        silently overwriting in-memory ones -- which doesn't match this
        docstring's own claim, so this uses setdefault() instead to
        actually implement "existing wins"). merge=False replaces the
        in-memory cache entirely. See save_table_cache's docstring for
        the staleness caveat."""
        loaded = upcast_cache(torch.load(path, map_location="cpu"))
        if merge:
            for k, v in loaded.items():
                self._table_cache.setdefault(k, v)
        else:
            self._table_cache = loaded

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

    @staticmethod
    def _cache_key(table: Table) -> str:
        """table_id ALONE is not a safe cache key -- scripts/
        finetune_query_table.py::cap_columns() builds a NEW Table with
        the SAME table_id but truncated `columns` (`table.columns[:max_columns]`)
        for outlier-wide tables during TRAINING batch construction. If a
        wide table gets cache-MISSED (and so cached) in its capped form
        during training, then the SAME table_id shows up again during
        VALIDATION scoring against the (uncapped) corpus -- table_id-only
        keying would cache-HIT and hand back the capped tensor (fewer
        columns) while the mask-building loop below iterates the real,
        uncapped table.columns -- writing column indices past the capped
        tensor's own width. That's a genuine CUDA
        "index out of bounds" (IndexKernel.cu) crash, not a shape error
        PyTorch catches cleanly, and it's asynchronous, so it doesn't
        surface until whatever GPU call happens to sync next (confirmed:
        this is what was actually causing the bert crashes previously
        misattributed to index_copy_/masked_fill in trainer.py's
        _corpus_scores -- those were just wherever the deferred error
        happened to surface, not where it originated).

        Including both column and row counts in the key (row count
        SHOULD already be uniform across every call site via
        SynSQLTableDataset's own max_rows, applied once at load time --
        but there's no cap_columns-style per-call row truncation
        currently, unlike columns) means a capped and an uncapped
        version of the same table_id simply never collide -- each shape
        variant gets its own cache entry instead of silently
        overwriting/misreading the other's.
        """
        return f"{table.table_id}#c{len(table.columns)}#r{table.num_rows}"

    def _shape_microbatches(self, indices: list[int], tables: list[Table]) -> list[list[int]]:
        """Group candidate indices by padded table shape under memory caps.

        This is shared by every baseline exposing ``forward_batch``. It
        changes only how candidate tables ride GPU kernels; adapter outputs
        are written back to their original indices before global scoring.
        """
        if not indices:
            return []
        cell_budget = self.table_microbatch_cell_budget
        max_tables = self.table_microbatch_max_tables
        if cell_budget is None and max_tables is None:
            return [indices]

        def ceil_power_of_two(value: int) -> int:
            return 1 << (max(1, value) - 1).bit_length()

        bins: dict[tuple[int, int], list[int]] = {}
        for i in indices:
            table = tables[i]
            key = (
                ceil_power_of_two(table.num_columns),
                ceil_power_of_two(table.num_rows),
            )
            bins.setdefault(key, []).append(i)

        groups: list[list[int]] = []
        for (col_bin, row_bin), bin_indices in sorted(
            bins.items(), key=lambda item: (item[0][0] * item[0][1], item[0])
        ):
            group_size = len(bin_indices)
            if cell_budget is not None:
                group_size = min(group_size, max(1, cell_budget // (col_bin * row_bin)))
            if max_tables is not None:
                group_size = min(group_size, max_tables)
            groups.extend(
                bin_indices[start : start + group_size]
                for start in range(0, len(bin_indices), group_size)
            )
        return groups

    def forward_batch_cellwise(
        self, tables: list[Table], ablation: str | None = None, profile: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same contract as TableEncoder.forward_batch_cellwise. ablation
        isn't supported here -- baselines don't share our
        concat(cell, header)-then-fuse scheme, so there's nothing
        analogous to isolate. profile is accepted for interface parity
        (unused directly here -- see self._last_frozen_s/_last_network_s
        below, which are ALWAYS recorded, not gated behind profile,
        mirroring TableEncoder's cell-encoding-vs-table-layers split so
        trainer.py's _score_batch can read the same two attribute names
        off either model type)."""
        if ablation is not None:
            raise NotImplementedError(
                f"ablation={ablation!r} is not supported by baseline encoders"
            )

        import time

        device = next(self.parameters()).device
        is_cuda = device.type == "cuda"
        self.baseline_encoder._profile_timings = profile
        self._last_table_microbatches = 1

        per_table_cell: list = [None] * len(tables)  # each [n_cols_i, n_rows_i, embed_dim]

        cache_hit_indices = []
        cache_miss_indices = []
        for i, table in enumerate(tables):
            if self.cacheable and self._cache_key(table) in self._table_cache:
                cache_hit_indices.append(i)
            else:
                cache_miss_indices.append(i)

        # Cache hits: skip tokenization + backbone forward + cell-pooling
        # entirely -- see cacheable's docstring. Cached in raw
        # (pre-projection) form on CPU, same convention as TextEmbedder's
        # cache, so self.projection (trainable, though typically Identity
        # for bert/tapas when embed_dim matches native_dim) is still
        # re-applied fresh every call. Counted as "frozen" time (it's a
        # cache load + a cheap Linear/Identity, not the backbone) -- ~0 on
        # a fully-warm cache.
        if is_cuda and profile:
            torch.cuda.synchronize()
        t_hits_0 = time.perf_counter()
        for i in cache_hit_indices:
            raw_cell = self._table_cache[self._cache_key(tables[i])].to(device)
            per_table_cell[i] = self.projection(raw_cell)
        if is_cuda and profile:
            torch.cuda.synchronize()
        frozen_s = time.perf_counter() - t_hits_0
        network_s = 0.0

        # Cache misses: use the baseline's forward_batch (ONE backbone
        # forward pass for every miss table at once) when it exposes one
        # -- currently bert_baseline.py, tapas_encoder.py, tabbie.py,
        # strubert.py, and turl.py; see their forward_batch docstrings for
        # the full
        # "why this matters" rationale (this used to call
        # self.baseline_encoder(headers, rows) once PER TABLE here, i.e.
        # one batch-of-1 (or, for strubert, batch-of-1 TWICE) backbone
        # forward pass per table -- the single biggest reason those
        # baselines were slower than 'ours', which always batches every
        # cell across the whole table batch into one call). TURL now uses
        # size-sorted, visibility-masked dynamic microbatches: differently
        # shaped tables are padded, independently masked, and restored to
        # candidate order afterward. HyTrel is the remaining per-table
        # fallback because its node/hyperedge incidence structure needs a
        # separate batching representation.
        if cache_miss_indices:
            if hasattr(self.baseline_encoder, "forward_batch"):
                groups = self._shape_microbatches(cache_miss_indices, tables)
                for group in groups:
                    batch_inputs = [
                        (*self._table_to_headers_rows(tables[i]), None) for i in group
                    ]

                    if is_cuda and profile:
                        torch.cuda.synchronize()
                    t_fb_0 = time.perf_counter()
                    encodings = self.baseline_encoder.forward_batch(batch_inputs)
                    if is_cuda and profile:
                        torch.cuda.synchronize()
                    t_fb_1 = time.perf_counter()

                    # TABBIE/StruBERT/TURL record their own internal
                    # frozen-vs-network split. BERT/TAPAS have no trainable
                    # table stack, so their whole group call is frozen time.
                    if hasattr(self.baseline_encoder, "_last_frozen_s"):
                        frozen_s += self.baseline_encoder._last_frozen_s
                        network_s += self.baseline_encoder._last_network_s
                    else:
                        frozen_s += t_fb_1 - t_fb_0

                    for group_pos, i in enumerate(group):
                        encoding = encodings[group_pos]
                        raw_cell = encoding.cell_embeddings.transpose(0, 1)
                        if self.cacheable:
                            self._table_cache[self._cache_key(tables[i])] = raw_cell.detach().cpu()
                        per_table_cell[i] = self.projection(raw_cell)
                self._last_table_microbatches = len(groups)
            else:
                self._last_table_microbatches = max(1, len(cache_miss_indices))
                if is_cuda and profile:
                    torch.cuda.synchronize()
                t_loop_0 = time.perf_counter()
                for i in cache_miss_indices:
                    headers, rows = self._table_to_headers_rows(tables[i])
                    encoding = self.baseline_encoder(headers, rows)
                    # encoding.cell_embeddings: [n_rows, n_cols, native_dim]
                    # (row-major, per common.py) -> [n_cols, n_rows, native_dim]
                    # (column-major, our convention)
                    raw_cell = encoding.cell_embeddings.transpose(0, 1)
                    if self.cacheable:
                        self._table_cache[self._cache_key(tables[i])] = raw_cell.detach().cpu()
                    per_table_cell[i] = self.projection(raw_cell)
                if is_cuda and profile:
                    torch.cuda.synchronize()
                frozen_s += time.perf_counter() - t_loop_0

        self._last_frozen_s = frozen_s
        self._last_network_s = network_s

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
            # Bound c_idx/r_idx to the ENCODED cell tensor's own [n, m]
            # dims. table.num_rows is the SHORTEST column (min over
            # columns -- see Table.num_rows), and that's what the encoder
            # used to build this table's rows, so the encoded tensor is
            # n x m. But a RAGGED table (columns of unequal length -- real
            # in the uncapped corpus) has longer columns whose extra cells
            # have r_idx >= m. Writing those into cell_mask (sized max_m
            # across the batch from these same encoded dims) indexes past
            # the tensor -- the CUDA "IndexKernel.cu ... index out of
            # bounds" device-side assert, which (being async) only
            # surfaces later, e.g. in _corpus_scores' masked_fill. Those
            # cells were never encoded and aren't real rows of the
            # rectangular table, so drop them here, consistent with how
            # row_mask/col_mask are built from n/m above.
            for c_idx, col in enumerate(table.columns):
                if c_idx >= n:
                    break
                for r_idx, val in enumerate(col.cells):
                    if r_idx >= m:
                        break
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
    tabbie_ffn_hidden_dim: int | None = None,
    strubert_ffn_hidden_dim: int | None = None,
    turl_attention_budget: int | None = None,
    table_microbatch_cell_budget: int | None = None,
    table_microbatch_max_tables: int | None = None,
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

    turl_attention_budget: TURL-only dynamic batching cap measured as
    B*S_max^2 dense attention elements. It changes execution grouping, not
    the visibility graph or learned architecture.
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
    if tabbie_ffn_hidden_dim is not None and encoder_name == "tabbie":
        kwargs["ffn_hidden_dim"] = tabbie_ffn_hidden_dim
    if strubert_ffn_hidden_dim is not None and encoder_name == "strubert":
        kwargs["ffn_hidden_dim"] = strubert_ffn_hidden_dim
    if turl_attention_budget is not None and encoder_name == "turl":
        kwargs["max_attention_elements"] = turl_attention_budget
    baseline_encoder = encoder_cls(**kwargs)

    adapter = BaselineCellwiseAdapter(
        baseline_encoder,
        embed_dim=embed_dim,
        cacheable=encoder_name in _FULLY_FROZEN_ENCODERS,
        table_microbatch_cell_budget=table_microbatch_cell_budget,
        table_microbatch_max_tables=table_microbatch_max_tables,
    )
    if device is not None:
        adapter = adapter.to(device)
    return adapter
