"""
Sanity checks to run once, before a real training launch -- the full
Table -> TableEncoder -> DiscriminatorHead -> electra_discriminator_loss
-> PretrainTrainer chain has been built and reasoned through piece by
piece, but never actually executed together end to end. This catches the
class of bug that's expensive to discover mid-training.

Checks 1-2 are architecture-only (no trainer involved, still apply
regardless of pretraining/finetuning strategy). Checks 3-4 exercise the
ELECTRA pretraining path specifically (PretrainTrainer + DiscriminatorHead).

Run with: python scripts/sanity_checks.py
"""

import random

import torch
import torch.nn.functional as F

from src.data.table import Column, Table
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import DiscriminatorHead, TableEncoder
from src.training.trainer import PretrainTrainer


# ==========================================================
# DUMMY DATA
# ==========================================================

def make_dummy_table(table_id: int, n_cols: int = 4, n_rows: int = 8) -> Table:
    columns = []
    for c in range(n_cols):
        if c % 2 == 0:
            # distinct numeric range per table, so magnitude itself carries
            # table identity, not just noise
            base = table_id * 1000
            cells = [str(base + random.randint(0, 100)) for _ in range(n_rows)]
        else:
            cells = [f"table{table_id}_text_{c}_{r}" for r in range(n_rows)]
        # header now includes table_id -- avoids every table sharing
        # identical header embeddings, which was masking any real
        # per-table signal in the earlier version of this fixture
        columns.append(Column(header=f"table{table_id}_col_{c}", cells=cells))

    return Table(table_id=str(table_id), table_name=f"table_{table_id}", columns=columns)


def shuffle_table(table: Table) -> Table:
    """Permute row order (consistently across columns) AND column order."""

    n_rows = table.num_rows
    row_perm = list(range(n_rows))
    random.shuffle(row_perm)

    shuffled_cols = [
        Column(header=col.header, cells=[col.cells[i] for i in row_perm])
        for col in table.columns
    ]
    random.shuffle(shuffled_cols)

    return Table(table_id=table.table_id, table_name=table.table_name, columns=shuffled_cols)


# ==========================================================
# CHECK 1: SHAPE
# ==========================================================

def check_shapes(model: TableEncoder, table: Table) -> None:
    X, col_mask, row_mask, cell_mask = model.forward_batch_cellwise([table])

    expected = (1, table.num_columns, table.num_rows, model.embed_dim)
    assert X.shape == expected, f"expected shape {expected}, got {tuple(X.shape)}"
    assert col_mask.shape == (1, table.num_columns), f"unexpected col_mask shape {tuple(col_mask.shape)}"
    assert row_mask.shape == (1, table.num_rows), f"unexpected row_mask shape {tuple(row_mask.shape)}"
    assert cell_mask.shape == (1, table.num_columns, table.num_rows), f"unexpected cell_mask shape {tuple(cell_mask.shape)}"

    print(f"[1/4 shape check] OK -- output shape {tuple(X.shape)}")


# ==========================================================
# CHECK 2: PERMUTATION INVARIANCE
# ==========================================================

def _masked_mean_pool(X: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
    """X: [B,N,M,k], cell_mask: [B,N,M] -> [B,k], masked mean over real
    cells. A sum/mean over an unordered set of real cells doesn't care
    what order those cells came in, so this is order-invariant by
    construction -- a lightweight standalone replacement for the
    RowCollapse-based table representation this check used to compare
    (which is no longer part of the pipeline)."""
    mask = cell_mask.unsqueeze(-1)  # [B,N,M,1]
    summed = (X * mask).sum(dim=(1, 2))  # [B,k]
    count = mask.sum(dim=(1, 2)).clamp(min=1.0)  # [B,1]
    return summed / count


def check_permutation_invariance(
    model: TableEncoder, table: Table, tol: float = 1e-3
) -> None:
    model.eval()

    with torch.no_grad():
        X_orig, _cm, _rm, cell_mask_orig = model.forward_batch_cellwise([table])
        X_shuf, _cm2, _rm2, cell_mask_shuf = model.forward_batch_cellwise([shuffle_table(table)])

        v_orig = _masked_mean_pool(X_orig, cell_mask_orig)
        v_shuf = _masked_mean_pool(X_shuf, cell_mask_shuf)

        sim_self = F.cosine_similarity(v_orig, v_orig).item()
        sim_shuffled = F.cosine_similarity(v_orig, v_shuf).item()

    diff = abs(sim_self - sim_shuffled)
    print(
        f"[2/4 invariance check] sim(orig, orig)={sim_self:.4f}  "
        f"sim(orig, shuffled)={sim_shuffled:.4f}  diff={diff:.6f}"
    )

    assert diff < tol, (
        "permutation invariance broken -- shuffled similarity diverges "
        "from self-similarity. Check for an operation that's accidentally "
        "position-sensitive (e.g. a stray reshape, or an unintended fixed "
        "ordering somewhere)."
    )
    print("[2/4 invariance check] OK")


# ==========================================================
# CHECK 3: GRADIENT FLOW (ELECTRA pretraining path)
# ==========================================================

def check_gradient_flow(trainer: PretrainTrainer, tables: list[Table]) -> None:
    trainer.optimizer = torch.optim.AdamW(trainer._trainable_params(), lr=1e-4)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda step: 1.0
    )

    loss_value = trainer.train_step(tables)
    assert not (loss_value != loss_value), "loss is NaN"  # NaN != NaN

    n_with_grad, n_zero_grad, n_missing_grad = 0, 0, 0

    named_params = list(trainer.model.named_parameters()) + [
        (f"discriminator.{n}", p) for n, p in trainer.discriminator.named_parameters()
    ]

    for name, p in named_params:
        if not p.requires_grad:
            continue
        if p.grad is None:
            print(f"  [WARN] no gradient at all for: {name}")
            n_missing_grad += 1
            continue
        if torch.isnan(p.grad).any():
            raise AssertionError(f"NaN gradient in {name}")
        if p.grad.abs().sum().item() == 0:
            n_zero_grad += 1
        else:
            n_with_grad += 1

    print(
        f"[3/4 gradient check] loss={loss_value:.4f}  "
        f"params_with_grad={n_with_grad}  zero_grad={n_zero_grad}  "
        f"missing_grad={n_missing_grad}"
    )
    assert n_missing_grad == 0, "some trainable parameters received no gradient at all"
    print("[3/4 gradient check] OK")


# ==========================================================
# CHECK 4: TINY-BATCH OVERFIT (ELECTRA pretraining path)
# ==========================================================

def check_tiny_batch_overfit(
    model_builder,
    discriminator_builder,
    tables: list[Table],
    steps: int = 100,
    lr: float = 1e-3,
    corrupt_frac: float = 0.3,
) -> None:
    """corrupt_frac deliberately higher than pretrain.yaml's default
    (0.15) -- a tiny fixed batch needs enough corrupted cells to give the
    discriminator a real signal to overfit to."""

    model = model_builder()
    discriminator = discriminator_builder()
    trainer = PretrainTrainer(model, discriminator, lr=lr, warmup_ratio=0.0, corrupt_frac=corrupt_frac)
    trainer.optimizer = torch.optim.AdamW(trainer._trainable_params(), lr=lr)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda step: 1.0
    )

    losses = []
    for step in range(steps):
        loss_value = trainer.train_step(tables)
        losses.append(loss_value)
        if step % 20 == 0:
            print(f"  step {step}: loss {loss_value:.4f}")

    print(f"[4/4 overfit check] first loss={losses[0]:.4f}  last loss={losses[-1]:.4f}")

    # diagnostic: on this same fixed (re-corrupted) batch, report the
    # discriminator's final accuracy against the trivial "always predict
    # real" baseline (which would score 1 - corrupt_frac) -- a model
    # that's actually learned something should clear that baseline by a
    # wide margin on data it's been directly overfitting to.
    from src.data.electra_corruption import corrupt_tables, pad_labels
    model.eval()
    discriminator.eval()
    with torch.no_grad():
        corrupted, label_grids = corrupt_tables(tables, corrupt_frac)
        labels = pad_labels(label_grids, device=next(model.parameters()).device)
        X, col_mask, row_mask, cell_mask = model.forward_batch_cellwise(corrupted)
        preds = (torch.sigmoid(discriminator(X)) > 0.5).float()
        correct = ((preds == labels).float() * cell_mask).sum()
        total = cell_mask.sum().clamp(min=1.0)
        accuracy = (correct / total).item()
        trivial_baseline = 1.0 - corrupt_frac
        print(
            f"  final discriminator accuracy: {accuracy:.3f} "
            f"(trivial always-real baseline: {trivial_baseline:.3f})"
        )

    assert losses[-1] < losses[0] * 0.5, (
        "loss did not drop meaningfully on a tiny, fixed batch -- this points "
        "to a bug in the loss/model wiring itself, independent of data or "
        "hyperparameters."
    )
    print("[4/4 overfit check] OK")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    def build_model() -> TableEncoder:
        cell_encoder = CellEncoder(text_model_name="bert-base-uncased", output_dim=32)
        return TableEncoder(cell_encoder, embed_dim=32)

    def build_discriminator() -> DiscriminatorHead:
        return DiscriminatorHead(embed_dim=32)

    model = build_model()
    table = make_dummy_table(1, n_cols=4, n_rows=8)

    check_shapes(model, table)
    check_permutation_invariance(model, table)

    trainer = PretrainTrainer(
        model, build_discriminator(), lr=1e-4, checkpoint_dir="/tmp/sanity_checkpoints"
    )
    tables_batch = [make_dummy_table(i, n_cols=4, n_rows=8) for i in range(3)]
    check_gradient_flow(trainer, tables_batch)

    check_tiny_batch_overfit(build_model, build_discriminator, tables_batch, steps=100)

    print("\nAll sanity checks passed.")
