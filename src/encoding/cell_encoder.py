"""
Cell encoding: turns raw cell text into R^k embeddings and directly
produces a padded, masked batch tensor -- no per-cell Python assignment
loop, no separate padding pass downstream.

Simplified (current) design -- everything is text, nothing is routed:
    - Every cell, numeric-looking or not, is treated as text and run
      through BERT (TextEmbedder). The earlier numeric/text split
      (CellType detection -> NumericEmbedder vs. TextEmbedder) has been
      dropped for now -- see src/encoding/numeric_embedder.py and
      src/encoding/cell_type.py, both unused on this path currently.
    - Headers are embedded independently with the SAME TextEmbedder,
      each header's own [CLS] token, taken separately per header (no
      joining headers into one sequence, no extra trainable
      cross-header contextualizer stacked on top).
    - A cell's final embedding is the RAW CONCATENATION of its own text
      embedding and its column's header embedding -- concat only, NO
      projection back down afterward. This means the requested
      `output_dim` is the FINAL, post-concatenation width; internally,
      BERT's projection targets `output_dim // 2` for both the cell and
      header halves, so concatenating the two lands exactly on
      `output_dim`. `output_dim` must be even.

Contains:
    CellEncoder   -- embeds every cell as text, embeds headers via BERT
                     CLS (independently, per header), fuses cell+header
                     via raw concatenation (no projection), and scatters
                     results directly into a padded tensor.

One loop here is irreducible: walking each table/column/row to collect
raw strings for a single flattened BERT call, and to figure out which
cells are null (for cell_mask). Every actual embedding computation is
vectorized (one batched BERT call + advanced-indexing scatter).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.data.table import Column, Table


# ==========================================================
# TEXT EMBEDDER (shared BERT + projection, used for cells AND headers)
# ==========================================================
# Note: this is a local, fuller variant of src/encoding/text_embedder.py
# (adds internal chunking via max_batch_size + a frozen-BERT embedding
# cache) -- kept here, inline, since CellEncoder relies on both.

class TextEmbedder(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_dim: int,
        max_length: int = 32,
        trainable: bool = False,
        max_batch_size: int = 2048,
        cache_enabled: bool = True,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.max_batch_size = max_batch_size

        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Caching only makes sense when BERT itself is frozen: a cached
        # value is a plain detached tensor, disconnected from the
        # computation graph. proj stays trainable and safe to
        # cache-around regardless, since it's re-applied fresh to the
        # cached hidden state on every call.
        self.cache_enabled = cache_enabled and not trainable
        self._cache: dict[str, torch.Tensor] = {}  # cell/header text -> raw hidden vector (CPU)

        hidden_size = self.encoder.config.hidden_size

        # Always a trainable linear map, even if hidden_size == output_dim --
        # matching CLIP's design: the backbone's native space and the
        # shared comparison space aren't assumed to be the same thing.
        self.proj = nn.Linear(hidden_size, output_dim, bias=False)

    def _encode_chunk_raw(self, cells: list[str]) -> torch.Tensor:
        """BERT forward pass only, no projection. [len(cells), hidden_size]"""

        device = next(self.encoder.parameters()).device

        encoded = self.tokenizer(
            cells,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        outputs = self.encoder(**encoded)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token

    def _encode_chunk(self, cells: list[str]) -> torch.Tensor:
        return self.proj(self._encode_chunk_raw(cells))

    def forward(self, cells: list[str]) -> torch.Tensor:
        """
        cells: list of N raw text strings (cell values OR header strings
               -- same pathway, [CLS] pooled independently per string)
        returns: [N, output_dim]

        Internally chunked at max_batch_size so a very large combined
        batch doesn't trigger one unbounded forward pass.

        When cache_enabled: only strings never seen before actually run
        through BERT -- real corpora have heavy duplication (repeated
        dates, categorical labels, common IDs, repeated headers), and
        BERT being frozen means the same string always produces the
        same output, so recomputing it is pure waste. Cache stores the
        PRE-projection hidden state (proj is trainable and applied
        fresh every call), on CPU (to avoid the cache itself becoming a
        growing GPU memory cost over a large corpus).
        """

        if len(cells) == 0:
            return torch.empty(0, self.proj.out_features)

        if not self.cache_enabled:
            if len(cells) <= self.max_batch_size:
                return self._encode_chunk(cells)
            chunks = [
                self._encode_chunk(cells[i : i + self.max_batch_size])
                for i in range(0, len(cells), self.max_batch_size)
            ]
            return torch.cat(chunks, dim=0)

        device = next(self.encoder.parameters()).device

        uncached_unique = list({c for c in cells if c not in self._cache})

        if uncached_unique:
            chunks = [
                self._encode_chunk_raw(uncached_unique[i : i + self.max_batch_size])
                for i in range(0, len(uncached_unique), self.max_batch_size)
            ]
            # Move the whole [N, half_k] block GPU->CPU in ONE transfer,
            # then index it -- instead of `for ...: h.detach().cpu()`,
            # which fired one tiny (synchronizing) device-to-host copy PER
            # cell string. .clone() copies each row out of the moved block
            # so entries stand alone and the block can be freed (cheap
            # CPU-side, no transfer).
            new_hidden = torch.cat(chunks, dim=0).detach().cpu()  # [N, half_k], one D2H copy
            for i, c in enumerate(uncached_unique):
                self._cache[c] = new_hidden[i].clone()

        # gather in ORIGINAL requested order -- preserves duplicates
        # correctly, including duplicates within this same call
        gathered = torch.stack([self._cache[c] for c in cells], dim=0).to(device)

        return self.proj(gathered)

    def clear_cache(self) -> None:
        self._cache.clear()

    def cache_size(self) -> int:
        return len(self._cache)

    def save_cache_to_disk(self, path: str) -> None:
        """
        Persists the current in-memory cell-embedding cache to disk --
        so future runs (new processes, resumed training, hyperparameter
        sweeps) can reuse it without ever calling BERT again for a
        string already seen, not just within this one process.
        """
        torch.save(self._cache, path)

    def load_cache_from_disk(self, path: str, merge: bool = True) -> None:
        """
        Loads a previously-saved cache from disk.

        merge=True  -- add to whatever's already in memory (existing
                        entries kept as-is if there's a key collision --
                        note plain dict.update() does the OPPOSITE,
                        silently letting loaded values win instead, so
                        this uses setdefault() to actually match this
                        docstring's contract)
        merge=False -- replace the in-memory cache entirely
        """
        loaded = torch.load(path, map_location="cpu")
        if merge:
            for k, v in loaded.items():
                self._cache.setdefault(k, v)
        else:
            self._cache = loaded


# ==========================================================
# CELL ENCODER (BERT for cells + BERT for headers, raw concat)
# ==========================================================

class CellEncoder(nn.Module):
    def __init__(
        self,
        text_model_name: str,
        output_dim: int,
        text_max_length: int = 32,
        text_trainable: bool = False,
        text_max_batch_size: int = 2048,
    ):
        super().__init__()

        if output_dim % 2 != 0:
            raise ValueError(
                f"output_dim must be even (cell half + header half "
                f"concatenated, no projection): got {output_dim}"
            )

        self.output_dim = output_dim  # final, post-concatenation width
        self.text_dim = output_dim // 2  # BERT projection width, per half

        # single shared BERT + projection, used for BOTH cell text and
        # header text -- same pathway, same weights. Projects to
        # text_dim (half of output_dim), since a cell's final embedding
        # is the two halves concatenated, unprojected.
        self.text_embedder = TextEmbedder(
            model_name=text_model_name,
            output_dim=self.text_dim,
            max_length=text_max_length,
            trainable=text_trainable,
            max_batch_size=text_max_batch_size,
        )

    def save_text_cache(self, path: str) -> None:
        """Persists the shared cell/header BERT-embedding cache to disk
        -- see TextEmbedder.save_cache_to_disk. Only meaningful when BERT
        is frozen (the default); a no-op is fine to call regardless, it
        just saves whatever's been accumulated in memory so far (empty
        if text_trainable=True, since caching is disabled in that case)."""
        self.text_embedder.save_cache_to_disk(path)

    def load_text_cache(self, path: str, merge: bool = True) -> None:
        """Loads a previously-saved cell/header BERT-embedding cache --
        e.g. one saved by save_text_cache() at the end of a PRETRAINING
        run, loaded here at the start of FINETUNING (a separate process,
        so the in-memory cache built during pretraining would otherwise
        be lost) -- every cell/header string seen during pretraining is
        then already cached instead of re-running BERT on it from
        scratch during finetuning/eval. See TextEmbedder.load_cache_from_disk."""
        self.text_embedder.load_cache_from_disk(path, merge=merge)

    def encode_column(self, column: Column) -> torch.Tensor:
        """
        Single-column entry point -- kept for debugging/inspection (e.g.
        scripts/debug_nan.py isolates one column at a time), not used on
        the main training path anymore. That path goes through
        encode_tables_batched instead, which batches across whole
        tables at once.

        column: a raw Column with N cells
        returns: [N, output_dim]
        """

        n = len(column.cells)

        if n == 0:
            return torch.empty(0, self.output_dim)

        device = next(self.parameters()).device

        cell_embeds = self.text_embedder(list(column.cells)).to(device)  # [N, text_dim]
        header_embed = self.text_embedder([column.header]).to(device)    # [1, text_dim]
        header_embeds = header_embed.expand(n, -1)                       # [N, text_dim]

        return torch.cat([cell_embeds, header_embeds], dim=-1)  # [N, output_dim] -- raw concat, no projection

    def encode_tables_batched(
        self, tables: list[Table]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """
        Batches text encoding across ALL given tables at once (cells AND
        headers, both via the same TextEmbedder), and scatters the
        results directly into a padded [B, max_n, max_m, k] tensor -- no
        Python loop copies embeddings into place; that's done with a
        single vectorized advanced-indexing assignment.

        The only irreducible Python-level loop here is walking the
        table/column/row structure to collect raw strings and null-cell
        positions -- not a tensor operation.

        returns:
            X:         [B, max_n, max_m, k] -- padded cell embeddings (k = output_dim,
                       each cell = raw concat(cell_text_vec, header_vec), unprojected)
            H:         [B, max_n, text_dim] -- padded header embeddings (text_dim =
                       output_dim // 2 -- each header's own raw BERT vector, the
                       same half that's concatenated into every cell in that column)
            col_mask:  [B, max_n]           -- 1 for real columns
            row_mask:  [B, max_m]           -- 1 for real rows
            cell_mask: [B, max_n, max_m]    -- 1 for non-null cells
            n_list:    list of B ints       -- each table's true column count
        """

        device = next(self.parameters()).device
        B = len(tables)
        k = self.output_dim
        half_k = self.text_dim

        n_list = [len(t.columns) for t in tables]
        m_list = [t.num_rows for t in tables]
        max_n = max(n_list) if n_list else 1
        max_m = max(m_list) if m_list else 1

        # -- irreducible part: walk structure, collect raw strings + positions --
        cell_strings: list[str] = []
        cell_t: list[int] = []
        cell_c: list[int] = []
        cell_r: list[int] = []

        nonnull_t: list[int] = []
        nonnull_c: list[int] = []
        nonnull_r: list[int] = []

        for t_idx, table in enumerate(tables):
            # num_rows is the SHORTEST column (min over columns, see
            # Table.num_rows), and m_list/max_m/row_mask above are built
            # from it -- so the grid is [max_n, max_m] with max_m derived
            # from num_rows. A RAGGED table (columns of unequal length --
            # real in the uncapped corpus) has longer columns whose extra
            # cells have row_idx >= num_rows >= that table's max_m slot.
            # Collecting them would scatter into X[ct,cc,cr] and
            # cell_mask[nnt,nnc,nnr] at cr/nnr past max_m -- the CUDA
            # "IndexKernel.cu ... index out of bounds" device-side assert
            # (async, so it surfaces later, e.g. in _corpus_scores'
            # masked_fill). They aren't real rows of the rectangular
            # table, so drop them, consistent with num_rows/row_mask.
            n_rows_t = table.num_rows
            for col_idx, column in enumerate(table.columns):
                for row_idx, cell in enumerate(column.cells):
                    if row_idx >= n_rows_t:
                        break
                    if cell.strip() != "":
                        nonnull_t.append(t_idx)
                        nonnull_c.append(col_idx)
                        nonnull_r.append(row_idx)

                    cell_strings.append(cell)
                    cell_t.append(t_idx)
                    cell_c.append(col_idx)
                    cell_r.append(row_idx)

        # -- masks: vectorized broadcasted comparison, no loop --
        n_list_t = torch.tensor(n_list, device=device)
        m_list_t = torch.tensor(m_list, device=device)
        col_mask = (
            torch.arange(max_n, device=device).unsqueeze(0) < n_list_t.unsqueeze(1)
        ).float()
        row_mask = (
            torch.arange(max_m, device=device).unsqueeze(0) < m_list_t.unsqueeze(1)
        ).float()

        # headers: each encoded independently via BERT's own [CLS] token,
        # one header at a time (batched in one call, but with no joining
        # of headers into a shared sequence, and no cross-header
        # contextualization step on top).
        header_strings: list[str] = []
        header_t: list[int] = []
        header_c: list[int] = []
        for t_idx, table in enumerate(tables):
            for col_idx, column in enumerate(table.columns):
                header_strings.append(column.header)
                header_t.append(t_idx)
                header_c.append(col_idx)

        H = torch.zeros(B, max_n, half_k, device=device)
        if header_strings:
            header_embeds = self.text_embedder(header_strings).to(device)  # [N_headers, half_k]
            ht = torch.tensor(header_t, device=device)
            hc = torch.tensor(header_c, device=device)
            H[ht, hc] = header_embeds

        # -- cells: one batched BERT call over every cell string in the
        #    batch (numeric-looking or not -- no routing), then fuse each
        #    cell's embedding with its own column's header embedding via
        #    RAW concatenation (no projection afterward), then scatter
        #    into X --
        X = torch.zeros(B, max_n, max_m, k, device=device)
        if cell_strings:
            raw_cell_embeds = self.text_embedder(cell_strings).to(device)  # [N, half_k]

            ct = torch.tensor(cell_t, device=device)
            cc = torch.tensor(cell_c, device=device)
            cr = torch.tensor(cell_r, device=device)

            # each cell's OWN column's header embedding -- H is already
            # fully populated at this point -- vectorized gather.
            header_for_cell = H[ct, cc]  # [N, half_k]

            fused = torch.cat([raw_cell_embeds, header_for_cell], dim=-1)  # [N, k] -- concat only

            X[ct, cc, cr] = fused

        # cell_mask: 1 for non-null cells specifically (a stricter
        # subset of row_mask -- padding is always null too, but a real,
        # non-padding row can still be null)
        cell_mask = torch.zeros(B, max_n, max_m, device=device)
        if nonnull_t:
            nnt = torch.tensor(nonnull_t, device=device)
            nnc = torch.tensor(nonnull_c, device=device)
            nnr = torch.tensor(nonnull_r, device=device)
            cell_mask[nnt, nnc, nnr] = 1.0

        return X, H, col_mask, row_mask, cell_mask, n_list
