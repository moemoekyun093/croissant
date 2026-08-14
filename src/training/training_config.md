# Current Training Configuration

Legend: 🎯 = deliberately reasoned choice · 🤷 = reasonable default, not tuned/validated

## Architecture

| Setting | Value | Rationale |
|---|---|---|
| `embed_dim` (k) | **64** in `pilot_train.py`, **32** in `sanity_checks.py` | 🤷 **Inconsistent between scripts — pick one before real training.** Neither was derived from anything principled; 64 was just chosen as "a bit bigger" for the pilot. Worth an actual sweep once other things are stable. |
| `text_model_name` | `bert-base-uncased` | 🤷 Arbitrary standard choice, never compared against alternatives (smaller/faster models, domain-specific variants). |
| `text_trainable` | `False` (frozen) | 🎯 Deliberate: training a novel, unvalidated architecture on top of a fully fine-tuned 110M-param BERT risked confounding "is the new architecture learning" with "is BERT drifting/forgetting." Frozen BERT + trainable projection (CLIP-style adapter) isolates what's actually being learned right now. Revisit unfreezing once the rest is validated. |
| `text_max_length` | 32 tokens | 🎯/⚠️ Deliberately sized for short cell values (names, numbers, short phrases) — but explicitly known to be inadequate for prose-heavy cells, confirmed directly against your science-standards table example (100+ word cells get heavily truncated). Unresolved trade-off, not yet revisited. |
| `numeric_sinusoidal_dim` | 128 | 🤷 Matches the convention used for diffusion timestep embeddings; never tuned specifically for this use case. |
| `ColumnAggregator` nonlinearity (`sigma`) | `sigmoid` | 🎯 Directly follows your original architecture spec (`sigma(XX^T)`); `tanh`/`relu` are wired as alternatives but untested. |
| `ChannelMix` hidden dim | `input_dim × 2` | 🤷 Lighter than the standard transformer FFN convention (usually ×4) — chosen as a smaller default, not validated either way. |

## Data / Corpus

| Setting | Value | Rationale |
|---|---|---|
| Row cap per table | 50 (`MAX_ROWS`) | 🎯 Inherited from your original corpus-building scripts, predates this pipeline — caps compute in `ColumnAggregator`'s `O(m²)` attention. |
| `n_tables` (pilot) | 10,000 | 🎯 Chosen specifically for fast iteration, given the ~144hr/epoch estimate on the full 1M corpus made further tuning there impractical. |
| `val_frac` | 0.1 (10% held out) | 🤷 Standard default split ratio, added specifically to catch overfitting to the small pilot corpus — the *decision* to hold out data was deliberate, the exact fraction wasn't tuned. |
| Augmentation type | row/column **subset-dropping**, not shuffling | 🎯 Directly follows from your architecture's proven exact permutation invariance (confirmed via the sanity check) — shuffling would be a zero-gradient augmentation. Also matches Starmie's own ablation finding that `drop_col` outperformed shuffle-based augmentations even for their non-invariant encoder. |
| `row_keep_frac`, `col_keep_frac` | 0.7, 0.7 | 🤷 "Moderate" subset size chosen by feel — not swept. Too close to 1.0 makes the positive pair trivially easy; too low makes it an unfair ask. No data on where the right value actually is yet. |

## Loss / Contrastive Setup

| Setting | Value | Rationale |
|---|---|---|
| `temperature` | 0.07 | 🎯 Taken directly from Starmie's fixed value, not derived independently — reasonable starting point given the similarity of the two setups, not re-validated for this architecture. |
| Loss direction | Symmetric (query→doc and doc→query averaged) | 🎯 Standard CLIP/Starmie convention — ensures both "sides" of the positive pair get gradient signal. |
| Negatives | Pure in-batch (every other table + its augmented view) | 🎯 The standard assumption given no labeled data exists — "two randomly drawn tables are dissimilar." No hard-negative mining yet (the BIRD/BM25-based weak supervision discussed earlier remains unimplemented). |
| Learned temperature (`logit_scale`) | Not used — fixed instead | 🎯 Deliberately deferred earlier in favor of Starmie's fixed value; a learned temperature is a reasonable thing to add later, once there's a way to tell if it actually helps. |

## Optimization

| Setting | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | 🤷 Standard default for transformer-adjacent training, not compared against alternatives. |
| Learning rate | 1e-4 | 🤷 Standard transformer fine-tuning default, never swept for this specific setup. |
| Weight decay | 0.01 | 🤷 Standard AdamW default. |
| LR schedule | Linear warmup (10% of steps) → linear decay to 0 | 🎯 Standard, well-established transformer training convention — low-risk default choice. |
| Gradient clipping | max norm 1.0 | 🎯 Standard safeguard against exploding gradients; conventional value, not tuned. |
| `batch_size` | 64 | 🎯 Chosen specifically because `nvidia-smi` showed ~95GB free on `cuda:2` — this is a conservative starting point given that headroom, not a derived optimum. Likely can go higher. |
| `num_epochs` | 15 (just proposed, not yet run) | 🤷 No principled stopping criterion yet beyond "watch train vs. val loss diverge" — this is exactly what the val-loss tracking just added is for. |

## Hardware / Infra

| Setting | Value | Rationale |
|---|---|---|
| `device` | `cuda:2` | 🎯 Directly chosen from your `nvidia-smi` output — GPUs 0/1/3 were heavily used by other jobs, GPU 2 was at 12% util / 2.6GB used. |
| `TextEmbedder` internal chunk size | 512 | 🎯 Safety bound added specifically to prevent one unbounded BERT forward pass when many tables' cells get combined into a single batched call — not stress-tested at the extremes of what your GPU could actually hold. |

---

## The honest summary

The **architecture-level choices** (frozen BERT, sigmoid nonlinearity, subset-dropping augmentation, symmetric InfoNCE, in-batch negatives) are reasoned through and defensible given what's been discussed and tested. The **numeric hyperparameters** (`lr`, `embed_dim`, `keep_frac`, `temperature`, `batch_size`, `num_epochs`) are almost entirely untuned defaults — reasonable starting points borrowed from adjacent work (Starmie, standard transformer training), not values derived from any experiment on your actual data. That's expected and fine at this stage — the pilot run with held-out validation is exactly the mechanism for starting to replace "reasonable guess" with "checked against evidence" for these, one at a time, rather than tuning all of them blind.