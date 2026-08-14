"""
Trainer: epoch loop over batches of raw Tables, using the augmentation +
InfoNCE setup from augmentation.py / losses.py.

Deliberately decoupled from any specific data source -- it accepts any
iterable that yields batches of `list[Table]`. Building the actual
Dataset/DataLoader that reads your corpus files and produces those
batches is a separate, still-open piece (see the accompanying reply).

AdamW + linear warmup/decay is the standard default and isn't specific
to this architecture -- nothing here is unusual on the optimization
side, only the loss (info_nce_loss, using maxsim) and the forward pass
(TableEncoder) are project-specific.
"""

import os
from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.data.table import Table
from src.data.augmentation import augment_table
from src.data.electra_corruption import corrupt_tables, pad_labels
from src.training.losses import (
    cross_score_queries_tables,
    electra_discriminator_loss,
    info_nce_loss,
    query_table_info_nce_loss,
)
from src.scoring.multi_score import MultiScorer


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> AdamW:
    """Only trainable parameters -- this naturally excludes frozen BERT
    weights inside CellEncoder's TextEmbedder."""

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Linear warmup, then linear decay to 0 -- standard transformer
    training schedule. last_epoch lets a resumed run continue the
    schedule from where it left off instead of restarting warmup."""

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        remaining = num_training_steps - step
        total_decay_steps = max(1, num_training_steps - num_warmup_steps)
        return max(0.0, remaining / total_decay_steps)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        grad_clip_norm: float = 1.0,
        temperature: float = 0.07,
        row_keep_frac: float = 0.7,
        col_keep_frac: float = 0.7,
        checkpoint_dir: str = "eval/report_runs",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.grad_clip_norm = grad_clip_norm
        self.temperature = temperature
        self.row_keep_frac = row_keep_frac
        self.col_keep_frac = col_keep_frac

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        log_path = os.path.join(checkpoint_dir, "train.log")
        self._log_file = open(log_path, "a", buffering=1)  # line-buffered -- visible immediately, not just on close
        print(f"logging to: {log_path}")

        self.optimizer: AdamW | None = None
        self.scheduler: LambdaLR | None = None
        self.global_step = 0

    def _log(self, message: str) -> None:
        print(message)
        self._log_file.write(message + "\n")

    def train_step(self, batch_tables: list[Table]) -> float:
        """One optimizer step over one batch of tables."""

        self.model.train()
        self.optimizer.zero_grad()

        augmented_tables = [
            augment_table(t, self.row_keep_frac, self.col_keep_frac)
            for t in batch_tables
        ]

        # combine originals + augmented into ONE batched forward pass;
        # forward_batch stays in padded+masked tensor form the whole
        # way through, so info_nce_loss consumes it directly -- no
        # unpadding/repadding round trip
        B = len(batch_tables)
        all_tables = batch_tables + augmented_tables
        X, mask = self.model.forward_batch(all_tables)

        loss = info_nce_loss(X, mask, B, temperature=self.temperature)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_norm
        )

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
        """
        dataloader: yields batches of list[Table]. Must be re-iterable
                    across epochs (e.g. a DataLoader, or a list of
                    pre-built batches).
        steps_per_epoch: needed up front to size the LR schedule --
                    pass len(dataloader) if known.
        val_batches: optional held-out batches, evaluated once per epoch
                    (no gradient updates) -- watch this alongside train
                    loss, especially when training many epochs over a
                    small, fixed corpus.
        resume_from: optional path to a checkpoint saved by
                    save_checkpoint(). If given, restores model/optimizer
                    state and global_step, then continues from the first
                    not-yet-completed epoch (num_epochs still means the
                    same total as the original run -- pass the same value
                    you used before, not a smaller "remaining" count).
        """
        import time

        total_steps = num_epochs * steps_per_epoch
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.optimizer = build_optimizer(self.model, self.lr, self.weight_decay)

        start_epoch = 0
        if resume_from is not None:
            self.load_checkpoint(resume_from)  # optimizer already exists, so its state loads too
            start_epoch = self.global_step // steps_per_epoch
            self._log(
                f"resumed from {resume_from}: global_step={self.global_step}, "
                f"starting at epoch {start_epoch}"
            )

        self.scheduler = build_scheduler(
            self.optimizer, warmup_steps, total_steps, last_epoch=self.global_step - 1
        )

        run_start = time.time()
        total_tables_seen = 0

        for epoch in range(start_epoch, num_epochs):
            epoch_losses = []
            epoch_start = time.time()

            for step, batch_tables in enumerate(dataloader):
                step_start = time.time()
                loss_value = self.train_step(batch_tables)
                step_elapsed = time.time() - step_start

                total_tables_seen += 2 * len(batch_tables)  # originals + augmented
                epoch_losses.append(loss_value)

                if self.global_step % log_every == 0:
                    avg_recent = sum(epoch_losses[-log_every:]) / min(
                        len(epoch_losses), log_every
                    )
                    elapsed = time.time() - run_start
                    tables_per_sec = total_tables_seen / max(elapsed, 1e-6)

                    mem_msg = ""
                    if torch.cuda.is_available() and "cuda" in self.device:
                        allocated = torch.cuda.memory_allocated(self.device) / 1e9
                        reserved = torch.cuda.memory_reserved(self.device) / 1e9
                        mem_msg = f" [mem: {allocated:.1f}GB alloc / {reserved:.1f}GB reserved]"

                    self._log(
                        f"epoch {epoch} step {step} "
                        f"(global {self.global_step}) "
                        f"loss {loss_value:.4f} "
                        f"(avg last {log_every}: {avg_recent:.4f}) "
                        f"[{step_elapsed:.2f}s/step, {tables_per_sec:.1f} tables/s, "
                        f"{elapsed/60:.1f} min elapsed]{mem_msg}"
                    )

            epoch_avg = sum(epoch_losses) / max(1, len(epoch_losses))
            epoch_elapsed = time.time() - epoch_start

            val_msg = ""
            if val_batches:
                val_loss = self.evaluate(val_batches)
                val_msg = f", val loss {val_loss:.4f}"

            self._log(
                f"== epoch {epoch} done, avg train loss {epoch_avg:.4f}{val_msg}, "
                f"took {epoch_elapsed/60:.1f} min =="
            )

            self.save_checkpoint(epoch)

        total_elapsed = time.time() - run_start
        self._log(
            f"\n== training complete: {num_epochs} epoch(s), "
            f"{total_tables_seen} table-forwards, "
            f"{total_elapsed/60:.1f} min total "
            f"({total_tables_seen/max(total_elapsed,1e-6):.1f} tables/s avg) =="
        )

    def evaluate(self, val_batches: list[list[Table]]) -> float:
        """
        Held-out loss, no gradient updates -- run this periodically
        against a validation split to catch overfitting to a small
        training corpus before it goes unnoticed.
        """

        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch_tables in val_batches:
                augmented_tables = [
                    augment_table(t, self.row_keep_frac, self.col_keep_frac)
                    for t in batch_tables
                ]
                B = len(batch_tables)
                all_tables = batch_tables + augmented_tables
                X, mask = self.model.forward_batch(all_tables)

                loss = info_nce_loss(X, mask, B, temperature=self.temperature)
                losses.append(loss.item())

        return sum(losses) / max(1, len(losses))

    def save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        self._log(f"saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]


# ==========================================================
# PRETRAIN TRAINER (ELECTRA-style cell-corruption discriminator)
# ==========================================================
# Replaces the augmentation+InfoNCE self-supervised task above as the
# pretraining stage: instead of contrasting a table against a
# row/column-subset augmented view of itself, this corrupts a random
# subset of cells (src/data/electra_corruption.py -- cheap same-column
# swap, no generator network) and trains a per-cell discriminator to
# spot which cells were swapped. Shares TableEncoder/CellEncoder with
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
    ):
        self.model = model.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.grad_clip_norm = grad_clip_norm
        self.corrupt_frac = corrupt_frac

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

        loss = electra_discriminator_loss(logits, labels, cell_mask)

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
        import random
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
            # sequence every time.
            random.shuffle(batches)

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

                loss = electra_discriminator_loss(logits, labels, cell_mask)
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


# ==========================================================
# FINETUNE TRAINER (real query -> table contrastive)
# ==========================================================
# Uses real (question, positive table) pairs -- e.g. from
# src/data/synsql_dataset.py's SynSQLQueryDataset -- instead of
# augmentation-derived positives. Scoring goes through
# src/scoring/multi_score.py's MultiScorer over row-resolved cell
# embeddings (TableEncoder.forward_batch_cellwise), NOT table-table
# MaxSim -- queries and tables are different kinds of things, MaxSim
# assumes both sides are tables. Default scoring mode: "row_match".
#
# Expects the model to already be initialized from a pretraining
# checkpoint via src/models/table_encoder.py::load_pretrained_encoder()
# -- this trainer itself doesn't do that loading (keeps it decoupled
# from any specific checkpoint format/location, same philosophy as the
# base Trainer being decoupled from any specific data source).

class FinetuneTrainer:
    def __init__(
        self,
        model: torch.nn.Module,           # TableEncoder
        query_encoder: torch.nn.Module,   # QueryEncoder
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        grad_clip_norm: float = 1.0,
        temperature: float = 0.07,
        scoring_mode: str = "row_match",
        checkpoint_dir: str = "eval/report_runs/finetune",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.query_encoder = query_encoder.to(device)
        self.scorer = MultiScorer().to(device)
        self.device = device

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.grad_clip_norm = grad_clip_norm
        self.temperature = temperature
        self.scoring_mode = scoring_mode

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
            for module in (self.model, self.query_encoder, self.scorer)
            for p in module.parameters()
            if p.requires_grad
        ]

    def _score_batch(self, batch: list[tuple[str, Table]]) -> torch.Tensor:
        """batch: list of B (question, positive_table) pairs, ALREADY
        resolved to exactly one positive table per query (if a query
        has multiple valid positives in your dataset, pick one -- e.g.
        randomly -- before calling this; this trainer stays agnostic to
        how that resolution happened, same as it's agnostic to the data
        source). Other tables in the batch serve as in-batch negatives
        for every query, so avoid two queries in the same batch sharing
        an identical positive table unless that's intentional.

        returns: [B, B] cross-score matrix (query i vs. table j)
        """
        queries = [q for q, _ in batch]
        tables = [t for _, t in batch]

        Q, query_mask = self.query_encoder(queries)  # [B, L, k], [B, L]
        Q = Q * query_mask.unsqueeze(-1)  # zero out padding-token vectors

        X, col_mask, row_mask, cell_mask = self.model.forward_batch_cellwise(tables)

        return cross_score_queries_tables(self.scorer, self.scoring_mode, Q, X, row_mask, col_mask)

    def train_step(self, batch: list[tuple[str, Table]]) -> float:
        self.model.train()
        self.query_encoder.train()
        self.scorer.train()
        self.optimizer.zero_grad()

        cross_scores = self._score_batch(batch)
        loss = query_table_info_nce_loss(cross_scores, temperature=self.temperature)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._trainable_params(), self.grad_clip_norm)

        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        return loss.item()

    def fit(
        self,
        dataloader: Iterable[list[tuple[str, Table]]],
        num_epochs: int,
        steps_per_epoch: int,
        log_every: int = 50,
        val_batches: list[list[tuple[str, Table]]] | None = None,
        resume_from: str | None = None,
    ) -> None:
        import random
        import time

        # Materialized once so it can be RESHUFFLED between epochs below --
        # same rationale as PretrainTrainer.fit.
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
            # reshuffle BATCH ORDER between epochs (query/table pairing
            # within each batch stays fixed -- only the sequence of
            # batches changes) so every epoch sees a different order
            # instead of the exact same sequence every time.
            random.shuffle(batches)

            epoch_losses = []
            epoch_start = time.time()

            for step, batch in enumerate(batches):
                loss_value = self.train_step(batch)
                epoch_losses.append(loss_value)

                if self.global_step % log_every == 0:
                    avg_recent = sum(epoch_losses[-log_every:]) / min(len(epoch_losses), log_every)
                    elapsed = time.time() - run_start
                    self._log(
                        f"[finetune] epoch {epoch} step {step} (global {self.global_step}) "
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
                f"== [finetune] epoch {epoch} done, avg train loss {epoch_avg:.4f}{val_msg}, "
                f"took {epoch_elapsed/60:.1f} min =="
            )
            self.save_checkpoint(epoch)

    def evaluate(self, val_batches: list[list[tuple[str, Table]]]) -> float:
        self.model.eval()
        self.query_encoder.eval()
        self.scorer.eval()
        losses = []

        with torch.no_grad():
            for batch in val_batches:
                cross_scores = self._score_batch(batch)
                loss = query_table_info_nce_loss(cross_scores, temperature=self.temperature)
                losses.append(loss.item())

        return sum(losses) / max(1, len(losses))

    def save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "query_encoder_state_dict": self.query_encoder.state_dict(),
                "scorer_state_dict": self.scorer.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        self._log(f"saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.query_encoder.load_state_dict(checkpoint["query_encoder_state_dict"])
        self.scorer.load_state_dict(checkpoint["scorer_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]