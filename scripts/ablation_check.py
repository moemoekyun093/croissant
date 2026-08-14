"""
Discriminator sanity check for the ELECTRA pretraining path: loads a
PretrainTrainer checkpoint (model + discriminator) and measures held-out
discriminator loss/accuracy across a sweep of corrupt_frac values,
compared against the TRIVIAL baseline of always predicting "real"
(which scores exactly 1 - corrupt_frac by construction).

Replaces the old header-vs-content ablation from before the ELECTRA
rewrite: that ablation zeroed out header or cell-content embeddings
separately to see which signal the model leaned on, which is no longer
possible now that CellEncoder fuses header + cell content via raw
concatenation before TableEncoder ever sees them (see
src/models/table_encoder.py's forward_batch -- "headers_only"/
"content_only" now raise NotImplementedError explicitly, rather than
silently returning a meaningless answer). This is the natural
analogous check for the CURRENT mechanism: is the discriminator
actually learning something about column-conditional value plausibility,
or just exploiting the corruption rate itself (e.g. by always
predicting "real", which alone would already look deceptively good at
low corrupt_frac)? A model that's clearing the trivial baseline by a
meaningful margin at every corrupt_frac in the sweep is a much stronger
signal than a single held-out loss number.

Usage:
    python -m scripts.ablation_check \
        --corpus_jsonl /path/to/corpus.jsonl \
        --checkpoint eval/report_runs/pretrain/checkpoint_epoch14.pt \
        --n_tables 10000 --embed_dim 64

Note: this rebuilds a fresh random val split (seeded) rather than
reusing pretrain_electra.py's exact held-out set, since the val table
IDs weren't persisted to disk during training. Close enough to answer
this question, but worth fixing (save val table IDs alongside
checkpoints) if you need the literal same held-out set for later
comparisons.
"""

import argparse
import random

import torch

from src.data.corpus_loader import iter_tables_from_jsonl
from src.data.electra_corruption import corrupt_tables, pad_labels
from src.data.table import Table
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import DiscriminatorHead, TableEncoder
from src.training.losses import electra_discriminator_loss


def load_tables(corpus_jsonl: str, n_tables: int) -> list[Table]:
    tables = []
    for t in iter_tables_from_jsonl(corpus_jsonl):
        tables.append(t)
        if len(tables) >= n_tables:
            break
    return tables


def make_batches(tables: list[Table], batch_size: int, max_columns: int = 20) -> list[list[Table]]:
    capped = []
    for t in tables:
        if len(t.columns) > max_columns:
            t = Table(table_id=t.table_id, table_name=t.table_name, columns=t.columns[:max_columns])
        capped.append(t)
    capped.sort(key=lambda t: (t.num_columns, t.num_rows))
    batches = [capped[i:i+batch_size] for i in range(0, len(capped), batch_size)]
    return [b for b in batches if len(b) >= 2]


@torch.no_grad()
def eval_discriminator(
    model: TableEncoder,
    discriminator: DiscriminatorHead,
    val_batches: list[list[Table]],
    corrupt_frac: float,
    device: str,
) -> tuple[float, float]:
    """returns: (avg BCE loss, accuracy) over every real, non-null cell
    across the val set, at this corrupt_frac."""
    model.eval()
    discriminator.eval()

    losses = []
    total_correct = 0.0
    total_cells = 0.0

    for batch in val_batches:
        corrupted, label_grids = corrupt_tables(batch, corrupt_frac)
        labels = pad_labels(label_grids, device=device)

        X, col_mask, row_mask, cell_mask = model.forward_batch_cellwise(corrupted)
        logits = discriminator(X)

        loss = electra_discriminator_loss(logits, labels, cell_mask)
        losses.append(loss.item())

        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += ((preds == labels).float() * cell_mask).sum().item()
        total_cells += cell_mask.sum().item()

    avg_loss = sum(losses) / max(1, len(losses))
    accuracy = total_correct / max(1.0, total_cells)
    return avg_loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True, help="a PretrainTrainer checkpoint (has both model_state_dict and discriminator_state_dict)")
    parser.add_argument("--n_tables", type=int, default=10_000)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=64, help="must match the checkpoint being loaded")
    parser.add_argument("--num_layers", type=int, default=1, help="must match the checkpoint being loaded")
    parser.add_argument(
        "--corrupt_fracs", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.30],
        help="sweep of corruption rates to evaluate at -- default matches configs/pretrain.yaml's sweep list",
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"loading up to {args.n_tables} tables ...")
    tables = load_tables(args.corpus_jsonl, args.n_tables)
    random.shuffle(tables)
    n_val = max(1, int(len(tables) * args.val_frac))
    val_tables = tables[:n_val]
    print(f"held-out val set: {len(val_tables)} tables")

    val_batches = make_batches(val_tables, args.batch_size)
    print(f"{len(val_batches)} val batches")

    cell_encoder = CellEncoder(text_model_name="bert-base-uncased", output_dim=args.embed_dim)
    model = TableEncoder(cell_encoder, embed_dim=args.embed_dim, num_layers=args.num_layers)
    discriminator = DiscriminatorHead(embed_dim=args.embed_dim)

    print(f"loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    model = model.to(args.device)
    discriminator = discriminator.to(args.device)

    print("\n== discriminator accuracy vs. trivial always-real baseline, by corrupt_frac ==")
    results = {}
    for corrupt_frac in args.corrupt_fracs:
        loss, accuracy = eval_discriminator(model, discriminator, val_batches, corrupt_frac, args.device)
        trivial_baseline = 1.0 - corrupt_frac
        margin = accuracy - trivial_baseline
        results[corrupt_frac] = (loss, accuracy, trivial_baseline, margin)
        print(
            f"  corrupt_frac={corrupt_frac:.2f}: loss={loss:.4f}  "
            f"accuracy={accuracy:.3f}  trivial_baseline={trivial_baseline:.3f}  "
            f"margin={margin:+.3f}"
        )

    print("\n== summary ==")
    for corrupt_frac, (loss, accuracy, trivial_baseline, margin) in results.items():
        verdict = "learning real signal" if margin > 0.05 else "close to trivial baseline -- check for a bug"
        print(f"  corrupt_frac={corrupt_frac:.2f}: margin={margin:+.3f} -- {verdict}")
