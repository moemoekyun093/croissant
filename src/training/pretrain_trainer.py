"""PretrainTrainer: ELECTRA-style cell-corruption discriminator pretraining."""


import os
import random
from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.data.table import Table
from src.data.electra_corruption import build_non_fk_mask, corrupt_tables, pad_labels
from src.training.losses import (
    electra_discriminator_loss,
)

from src.training.scheduler import build_scheduler


# ==========================================================
# PRETRAIN TRAINER (ELECTRA-style cell-corruption discriminator)
# ==========================================================
# The self-supervised pretraining stage: corrupts a random subset of
# cells (src/data/electra_corruption.py -- cheap same-column swap, no
# generator network) and trains a per-cell discriminator to spot which
# cells were swapped. Shares TableEncoder/CellEncoder with
# the rest of this codebase; the ONLY new trainable piece is
# DiscriminatorHead (src/models/table_encoder.py).

class PretrainTrainer:
    def __init__(
        self,
        model: torch.nn.Module,           # TableEncoder
        discriminator: torch.nn.Module,   # DiscriminatorHead
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        grad_clip_norm: float = 1.0,
        corrupt_frac: float = 0.15,
        checkpoint_dir: str = "eval/report_runs/pretrain",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
    ):
        self.model = model.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.grad_clip_norm = grad_clip_norm
        self.corrupt_frac = corrupt_frac

        # owns its own seeded RNG (not the bare `random` module) so
        # between-epoch batch reshuffling is reproducible given the same
        # seed -- construction-time batch/table sampling in the calling
        # script should use this same seed for a fully reproducible run.
        self._rng = random.Random(seed)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_path = os.path.join(checkpoint_dir, "train.log")
        self._log_file = open(log_path, "a", buffering=1)
        print(f"logging to: {log_path}")

        self.optimizer: AdamW | None = None
        self.scheduler: LambdaLR | None = None
        self.global_step = 0

    def _log(self, message: str) -> None:
        print(message)
        self._log_file.write(message + "\n")

    def _trainable_params(self):
        return [
            p
            for module in (self.model, self.discriminator)
            for p in module.parameters()
            if p.requires_grad
        ]

    def train_step(self, batch_tables: list[Table]) -> float:
        self.model.train()
        self.discriminator.train()
        self.optimizer.zero_grad()

        corrupted_tables, label_grids = corrupt_tables(batch_tables, self.corrupt_frac)
        labels = pad_labels(label_grids, device=self.device)  # [B, N, M]

        X, col_mask, row_mask, cell_mask = self.model.forward_batch_cellwise(corrupted_tables)
        logits = self.discriminator(X)  # [B, N, M]

        # exclude declared FOREIGN KEY columns from the loss -- FK values
        # legitimately repeat across rows, so a same-column swap there is
        # usually indistinguishable from a normal repeat (unlike a
        # PRIMARY KEY duplicate, which is a genuine anomaly and stays in
        # the loss). See src/data/electra_corruption.py::build_non_fk_mask.
        non_fk_mask = build_non_fk_mask(corrupted_tables, device=self.device)
        effective_mask = cell_mask * non_fk_mask

        loss = electra_discriminator_loss(logits, labels, effective_mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._trainable_params(), self.grad_clip_norm)

        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        return loss.item()

    def fit(
        self,
        dataloader: Iterable[list[Table]],
        num_epochs: int,
        steps_per_epoch: int,
        log_every: int = 50,
        val_batches: list[list[Table]] | None = None,
        resume_from: str | None = None,
    ) -> None:
        import time

        # Materialized once (not just iterated) specifically so it can be
        # RESHUFFLED between epochs below -- an arbitrary Iterable might
        # only support a single pass, but every actual caller here passes
        # a concrete list of pre-built batches anyway (see
        # scripts/pretrain_electra.py), so this is safe and just makes
        # that assumption explicit.
        batches = list(dataloader)

        total_steps = num_epochs * steps_per_epoch
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.optimizer = AdamW(self._trainable_params(), lr=self.lr, weight_decay=self.weight_decay)

        start_epoch = 0
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            start_epoch = self.global_step // steps_per_epoch
            self._log(f"resumed from {resume_from}: global_step={self.global_step}, starting at epoch {start_epoch}")

        self.scheduler = build_scheduler(
            self.optimizer, warmup_steps, total_steps, last_epoch=self.global_step - 1
        )

        run_start = time.time()

        for epoch in range(start_epoch, num_epochs):
            # reshuffle BATCH ORDER between epochs (not batch contents --
            # each batch's own table composition stays whatever
            # make_batches' size-bucketing produced) so every epoch sees
            # a different training order instead of the exact same
            # sequence every time. Uses this trainer's own seeded RNG
            # (not the bare random module) so the whole run is
            # reproducible given the same seed.
            self._rng.shuffle(batches)

            epoch_losses = []
            epoch_start = time.time()

            for step, batch_tables in enumerate(batches):
                loss_value = self.train_step(batch_tables)
                epoch_losses.append(loss_value)

                if self.global_step % log_every == 0:
                    avg_recent = sum(epoch_losses[-log_every:]) / min(len(epoch_losses), log_every)
                    elapsed = time.time() - run_start
                    self._log(
                        f"[pretrain] epoch {epoch} step {step} (global {self.global_step}) "
                        f"loss {loss_value:.4f} (avg last {log_every}: {avg_recent:.4f}) "
                        f"[{elapsed/60:.1f} min elapsed]"
                    )

            epoch_avg = sum(epoch_losses) / max(1, len(epoch_losses))
            epoch_elapsed = time.time() - epoch_start

            val_msg = ""
            if val_batches:
                val_loss = self.evaluate(val_batches)
                val_msg = f", val loss {val_loss:.4f}"

            self._log(
                f"== [pretrain] epoch {epoch} done, avg train loss {epoch_avg:.4f}{val_msg}, "
                f"took {epoch_elapsed/60:.1f} min =="
            )
            self.save_checkpoint(epoch)

    def evaluate(self, val_batches: list[list[Table]]) -> float:
        self.model.eval()
        self.discriminator.eval()
        losses = []

        with torch.no_grad():
            for batch_tables in val_batches:
                corrupted_tables, label_grids = corrupt_tables(batch_tables, self.corrupt_frac)
                labels = pad_labels(label_grids, device=self.device)

                X, col_mask, row_mask, cell_mask = self.model.forward_batch_cellwise(corrupted_tables)
                logits = self.discriminator(X)

                non_fk_mask = build_non_fk_mask(corrupted_tables, device=self.device)
                effective_mask = cell_mask * non_fk_mask

                loss = electra_discriminator_loss(logits, labels, effective_mask)
                losses.append(loss.item())

        return sum(losses) / max(1, len(losses))

    def save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                # NOTE: discriminator is saved under its OWN key, never
                # merged into model_state_dict -- this is exactly what
                # lets load_pretrained_encoder() (src/models/
                # table_encoder.py) load just the encoder for finetuning
                # and cleanly discard the discriminator head.
                "model_state_dict": self.model.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        self._log(f"saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
