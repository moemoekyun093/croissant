"""
Sanity checks to run once, before a real training launch -- the full
Table -> TableEncoder -> maxsim -> info_nce_loss -> Trainer chain has
been built and reasoned through piece by piece, but never actually
executed together end to end. This catches the class of bug that's
expensive to discover mid-training.

Run with: python scripts/sanity_checks.py
"""

import random

import torch

from src.data.augmentation import drop_columns, drop_rows
from src.data.table import Column, Table
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import TableEncoder
from src.scoring.maxsim import maxsim
from src.training.trainer import Trainer


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
    out = model(table)
    expected = (table.num_columns, model.row_collapse.input_dim)

    assert out.shape == expected, (
        f"expected shape {expected}, got {tuple(out.shape)}"
    )
    print(f"[1/4 shape check] OK -- output shape {tuple(out.shape)}")


# ==========================================================
# CHECK 2: PERMUTATION INVARIANCE
# ==========================================================

def check_permutation_invariance(
    model: TableEncoder, table: Table, tol: float = 1e-3
) -> None:
    model.eval()

    with torch.no_grad():
        out_orig = model(table)
        out_shuffled = model(shuffle_table(table))

        sim_self = maxsim(out_orig, out_orig).item()
        sim_shuffled = maxsim(out_orig, out_shuffled).item()

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
# CHECK 3: GRADIENT FLOW
# ==========================================================

def check_gradient_flow(trainer: Trainer, tables: list[Table]) -> None:
    trainer.optimizer = torch.optim.AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad], lr=1e-4
    )
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda step: 1.0
    )

    loss_value = trainer.train_step(tables)
    assert not (loss_value != loss_value), "loss is NaN"  # NaN != NaN

    n_with_grad, n_zero_grad, n_missing_grad = 0, 0, 0

    for name, p in trainer.model.named_parameters():
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
# CHECK 4: TINY-BATCH OVERFIT
# ==========================================================

def check_tiny_batch_overfit(
    model_builder, tables: list[Table], steps: int = 100, lr: float = 1e-3
) -> None:
    model = model_builder()
    trainer = Trainer(model, lr=lr, warmup_ratio=0.0)
    trainer.optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
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

    # diagnostic: print the final pairwise similarity matrix so a collapse
    # (near-uniform similarities across all pairs) is visible directly,
    # rather than only inferred from the loss plateauing near ln(negatives)
    model.eval()
    with torch.no_grad():
        reprs = [model(t) for t in tables]
        print("  final pairwise maxsim matrix (rows=query tables):")
        for i, r_i in enumerate(reprs):
            row = [f"{maxsim(r_i, r_j).item():.3f}" for r_j in reprs]
            print(f"    table {i}: {row}")

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

    model = build_model()
    table = make_dummy_table(1, n_cols=4, n_rows=8)

    check_shapes(model, table)
    check_permutation_invariance(model, table)

    trainer = Trainer(model, lr=1e-4, checkpoint_dir="/tmp/sanity_checkpoints")
    tables_batch = [make_dummy_table(i, n_cols=4, n_rows=8) for i in range(3)]
    check_gradient_flow(trainer, tables_batch)

    check_tiny_batch_overfit(build_model, tables_batch, steps=100)

    print("\nAll sanity checks passed.")