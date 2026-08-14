# Current Training Configuration

Legend: [deliberate] = reasoned choice, checked against something concrete.
[default] = reasonable starting point, not tuned or validated yet.

This describes the current pipeline: ELECTRA-style cell-corruption
pretraining (PretrainTrainer), followed by finetuning on real query to
table pairs (FinetuneTrainer). It replaces an earlier augmentation +
table-table InfoNCE contrastive pretraining setup, which has been
removed from the codebase (see git history / configs/ if you need to
reconstruct it). The values below match configs/model.yaml,
configs/pretrain.yaml, and configs/finetune.yaml -- those files are the
source of truth for defaults and sweep candidates; this document is
just the reasoning behind them.

## Architecture (shared by both stages)

| Setting | Value | Rationale |
|---|---|---|
| `embed_dim` (k) | 64 | [default] Final, post-concatenation cell width. Never swept; worth an actual sweep once a pretraining run is validated end to end. Must be even -- BERT projects to `embed_dim // 2` for the cell half and the header half separately. |
| `text_model_name` | `bert-base-uncased` | [default] Arbitrary standard choice, never compared against alternatives (smaller/faster models, domain-specific variants). |
| `text_trainable` (CellEncoder) | `False` (frozen) | [deliberate] Training a novel, unvalidated table architecture on top of a fully fine-tuned 110M-param BERT risks confounding "is the new architecture learning" with "is BERT drifting." Frozen BERT + trainable projection isolates what's actually being learned. Revisit unfreezing once the rest is validated. |
| Cell/header fusion | raw concatenation, no projection afterward | [deliberate] A cell's embedding is `concat(cell_BERT_vector, header_BERT_vector)`, both already post-BERT, post-projection vectors -- not raw text concatenation, and no linear layer collapses the two back down. This also means header info can no longer be isolated from cell content at the TableEncoder level (see `forward_batch`'s ablation handling). |
| Numeric cell routing | none -- every cell is text | [deliberate] The earlier numeric/text split (a separate sinusoidal-embedding path for numeric-looking cells) has been dropped. Every cell, numeric-looking or not, goes through BERT as text. `src/encoding/numeric_embedder.py` and `src/encoding/cell_type.py` are unused on the main path now; revisit if numeric-aware encoding turns out to matter. |
| Header contextualization | none -- independent per-header BERT CLS | [deliberate] Headers are embedded independently via BERT's own CLS token, one at a time (batched in one call, not joined into a shared sequence). The earlier cross-header contextualizer transformer has been removed from the main path. |
| `text_max_length` | 32 tokens | [deliberate, known limitation] Sized for short cell values (names, numbers, short phrases) -- confirmed inadequate for prose-heavy cells (100+ word cells get heavily truncated). Unresolved trade-off, not yet revisited. |
| `ColumnAggregator`/RCPE nonlinearity (`sigma`) | `sigmoid` | [deliberate] Follows the original architecture spec (`sigma(XX^T)`); `tanh`/`relu` are wired as alternatives but untested. |
| `ChannelMix` hidden dim | `input_dim x 2` | [default] Lighter than the standard transformer FFN convention (usually x4) -- chosen as a smaller default, not validated either way. |
| Row cap per table (`MAX_ROWS`) | 50 | [deliberate] Caps compute in the row-attention steps, which are quadratic in row count. |

## Pretraining (ELECTRA-style cell corruption)

| Setting | Value | Rationale |
|---|---|---|
| Mechanism | cell corruption + per-cell discriminator | [deliberate] Replaces the earlier augmentation-based table-table contrastive task. A random subset of cells get swapped for another real value from the same column; a discriminator head predicts real vs. swapped per cell, using the row-resolved embeddings from before RowCollapse. |
| Corruption scheme | cheap same-column swap, no generator network | [deliberate] The replacement for a corrupted cell is just another real value from the same column (matched by exact header text, anywhere in the batch, including the same table) -- not a value produced by a trained generator model. Same idea TABBIE's own corrupt-cell-detection pretraining task uses. Avoids training a second network alongside the discriminator. |
| `corrupt_frac` | 0.15 | [default] Fraction of real cells corrupted per table, chosen by analogy to BERT's 15% masking rate -- not derived for this specific task. |
| Discriminator granularity | per-cell | [deliberate] One real/corrupted logit per cell, from the row-resolved embeddings before RowCollapse -- matches the granularity corruption itself operates at, and is the same tensor shape the finetuning scorer needs. |
| Discriminator head | 2-layer MLP, GELU, hidden dim = embed_dim | [default] Small and simple by design; not swept. |
| `lr` | 1e-4 | [default] Carried over from the earlier contrastive setup's default, not re-tuned for this task. |
| `batch_size` | 64 | [default, known risk] Carried over from the earlier pipeline's GPU-headroom-based choice. Not re-checked: pretraining now carries the full row-resolved `[B,N,M,k]` tensor through backprop instead of a pooled `[B,N,k]` tensor, so memory pressure at a given batch size is higher than before. |
| `num_epochs` | 15 | [default] No principled stopping criterion yet -- watch train vs. val loss. |
| Optimizer / schedule | AdamW, linear warmup (10%) then linear decay to 0, grad clip norm 1.0 | [deliberate] Standard, low-risk transformer training convention, unchanged from the earlier setup. |
| Batch reshuffling | shuffled once before the train/val split, batch order reshuffled every epoch | [deliberate] Batch composition (which tables land together, from size-bucketing) stays fixed across epochs; only the order batches are seen in changes. |

## Finetuning (real query to table contrastive)

| Setting | Value | Rationale |
|---|---|---|
| Data source | real (question, positive table) pairs, e.g. `SynSQLQueryDataset` | [deliberate] Replaces the earlier augmentation-derived synthetic positives with real supervision. |
| Query encoder | separate BERT instance + projection to full `embed_dim` | [deliberate] Not built on CellEncoder's TextEmbedder, which projects to half of `embed_dim` specifically so cell+header concatenation lands on the full width. A query has no header counterpart, so it needs the full width directly. |
| Query encoder trainable | `True` | [deliberate] Unlike CellEncoder's frozen BERT, the query tower trains from the start -- finetuning is exactly the stage with real query supervision to learn from. |
| Scoring | `MultiScorer`, mode = `row_match` | [deliberate] Chosen over `column_match`/`mixture`/others as the starting point; structurally closest to the table-table MaxSim convention the architecture was originally built around. All six MultiScorer modes are implemented and swappable via config. |
| Negatives | in-batch, one positive table per query | [known issue, not just untuned] If two queries in the same batch share a positive table, that table is currently treated as a false negative for the other query. Not yet fixed -- see `configs/finetune.yaml`. |
| `temperature` | 0.07 | [default] Carried over from the table-table InfoNCE default, not re-derived for query-table scoring's different score scale. |
| `lr` | 1e-4 | [default, worth checking] Finetuning often wants a lower learning rate than pretraining, since the encoder already has learned structure to preserve. Not yet checked explicitly. |
| `batch_size` | 32 | [default] Smaller than pretraining's default by guess only -- the cross-query-vs-table scoring loop's cost grows faster with batch size here than in the other stages. Not measured. |
| Checkpoint transfer | `load_pretrained_encoder()` | [deliberate] Loads only the TableEncoder weights from a pretraining checkpoint, discarding the discriminator head -- the discriminator has no role once pretraining ends. |

## Data / corpus

| Setting | Value | Rationale |
|---|---|---|
| `val_frac` | 0.1 (10% held out) | [default] Standard split ratio, not tuned. |
| Table source (SynSQL) | live SQLite reads via `SynSQLTableDataset` | [deliberate] Table and column names are read directly from each database's own SQLite schema (`sqlite_master` + `PRAGMA table_info`), not from `tables.json` -- confirmed that dataset's `tables.json` is Spider-style, with a `column_names` field that holds human-readable descriptions rather than real column identifiers. Reading the live schema instead can't drift out of sync with the real data. |

## Hardware / infra

| Setting | Value | Rationale |
|---|---|---|
| `device` | `cuda:2` | [deliberate, needs rechecking] Originally chosen from a specific `nvidia-smi` snapshot -- re-check before reuse, GPU availability changes. |
| `TextEmbedder` internal chunk size | 2048 | [deliberate] Safety bound preventing one unbounded BERT forward pass when many tables' cells get combined into a single batched call. |

---

## The honest summary

The architecture-level choices (frozen BERT for cell/header encoding, raw concatenation instead of a learned projection, cheap same-column corruption instead of a generator network, per-cell discriminator granularity, in-batch negatives for finetuning) are reasoned through and defensible given what's been discussed and tested. The numeric hyperparameters (`lr`, `embed_dim`, `corrupt_frac`, `temperature`, `batch_size`, `num_epochs`) are almost entirely untuned defaults, several carried over unchanged from the earlier contrastive setup without being re-checked against this task's different memory profile and loss scale. Nothing here has been run against real torch or real data yet -- see `scripts/real_data_check.py` and `scripts/sanity_checks.py` for the checks meant to catch a first-run bug before a full pretraining launch, and `configs/pretrain.yaml` / `configs/finetune.yaml` for sweep candidates once a first run is confirmed to work.
