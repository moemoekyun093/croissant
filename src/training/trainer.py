"""
PretrainTrainer (ELECTRA-style cell-corruption discriminator) and
FinetuneTrainer (real query -> table contrastive) -- epoch loops over
batches of raw Tables / (question, Table) pairs.

Deliberately decoupled from any specific data source -- both accept any
iterable that yields the right batch shape. AdamW + linear warmup/decay
is the standard default and isn't specific to this architecture --
nothing on the optimization side is unusual, only the losses
(electra_discriminator_loss, query_table_info_nce_loss) and the forward
pass (TableEncoder) are project-specific.
"""

import os
import random
import time
from typing import Callable, Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.data.table import Table
from src.data.electra_corruption import build_non_fk_mask, corrupt_tables, pad_labels
from src.eval.retrieval_metrics import compute_map, compute_ranking_metrics
from src.training.losses import (
    cross_score_queries_tables,
    electra_discriminator_loss,
    query_table_info_nce_loss,
)
from src.scoring.multi_score import MultiScorer


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
# from any specific checkpoint format/location, same philosophy as
# PretrainTrainer being decoupled from any specific data source).

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
        seed: int = 42,
        profile: bool = False,
        profile_every: int = 20,
    ):
        self.model = model.to(device)
        self.query_encoder = query_encoder.to(device)
        self.scorer = MultiScorer().to(device)
        self.device = device

        # Per-stage timing (query encoding vs. frozen-backbone table
        # encoding vs. trainable network-on-top vs. scoring) inside
        # _score_batch, to expose which part of a training step is
        # actually slow -- see _record_profile. Off by default since it
        # adds a few torch.cuda-syncing time.perf_counter() calls per step.
        self.profile = profile
        self.profile_every = profile_every
        self._profile_accum = {
            "n": 0, "n_queries": 0, "n_tables": 0,
            "query_s": 0.0, "frozen_s": 0.0, "network_s": 0.0, "score_s": 0.0,
        }

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.grad_clip_norm = grad_clip_norm
        self.temperature = temperature
        self.scoring_mode = scoring_mode
        self.seed = seed
        # Unlike PretrainTrainer, this trainer doesn't own the batch
        # construction itself -- fit()'s batch_fn (see its docstring) is
        # called fresh every epoch by the CALLER (e.g.
        # scripts/finetune_query_table.py's build_train_batches), which
        # should use its own random.Random(seed) with this same `seed`
        # for a fully reproducible run; no separate RNG is kept here.

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

    def _score_batch(
        self, batch: tuple[list[tuple[str, Table]], list[Table]]
    ) -> torch.Tensor:
        """batch: (pairs, hard_negatives).

        pairs: list of Bq (question, positive_table) pairs, ALREADY
            resolved to exactly one positive table per query (if a
            query has multiple valid positives in your dataset, pick
            one -- e.g. randomly -- before calling this; this trainer
            stays agnostic to how that resolution happened, same as
            it's agnostic to the data source). Every OTHER query's
            positive table in `pairs` doubles as an in-batch negative
            for this query, so avoid two queries in the same batch
            sharing an identical positive table unless that's
            intentional.
        hard_negatives: extra candidate tables with no query of their
            own, appended as columns Bq..Bq+len(hard_negatives)-1 --
            e.g. same-database, non-gold tables (see
            scripts/finetune_query_table.py's resolve_train_batches),
            which are harder negatives than a random other query's
            positive table since they share schema/domain vocabulary.
            Pass an empty list to disable (plain in-batch negatives
            only, previous behavior).

        returns: [Bq, Bq + len(hard_negatives)] cross-score matrix
            (query i vs. table j); query i's positive is always at
            column i (see query_table_info_nce_loss's docstring).
        """
        pairs, hard_negatives = batch

        # Same empty-table guard _corpus_scores applies to the validation
        # corpus -- a table with 0 rows or 0 columns reaching
        # forward_batch_cellwise produces degenerate tokenization/pooling
        # (e.g. _pool_cells' index_add_ into a buffer sized n_rows*n_cols,
        # which collapses to 0 elements) that's a strong match for the
        # CUDA IndexKernel.cu "index out of bounds" assert seen during
        # training. This path never had the check the validation path
        # has, so a bad table here would fail as a queued/async CUDA
        # error only reported later, at validation's first sync point --
        # which is exactly what every crash trace pointed at.
        #
        # Positive tables (`pairs`) can't be silently dropped -- doing so
        # would misalign query_table_info_nce_loss's column-i-is-query-i's
        # positive convention -- so a bad positive raises loudly instead.
        # Hard negatives are safe to just drop.
        bad_pairs = [(q, t) for q, t in pairs if t.num_rows == 0 or t.num_columns == 0]
        if bad_pairs:
            raise ValueError(
                f"_score_batch: {len(bad_pairs)} query's positive table has "
                f"0 rows or 0 columns -- would otherwise reach "
                f"forward_batch_cellwise and likely crash as an unattributed "
                f"CUDA device-side assert instead. offending (question, "
                f"table_id, rows, cols): "
                f"{[(q, t.table_id, t.num_rows, t.num_columns) for q, t in bad_pairs[:5]]}"
            )
        bad_negs = [t for t in hard_negatives if t.num_rows == 0 or t.num_columns == 0]
        if bad_negs:
            print(
                f"[_score_batch] WARNING: dropping {len(bad_negs)} empty "
                f"hard-negative table(s): "
                f"{[(t.table_id, t.num_rows, t.num_columns) for t in bad_negs[:5]]}"
            )
            hard_negatives = [t for t in hard_negatives if t.num_rows > 0 and t.num_columns > 0]

        queries = [q for q, _ in pairs]
        tables = [t for _, t in pairs] + list(hard_negatives)

        # Each stage wrapped separately so a crash's traceback -- and any
        # printed diagnostic -- clearly names WHICH stage (query encoding
        # vs. table encoding vs. scoring) failed, instead of one opaque
        # exception somewhere inside this function. On a table-encoding
        # failure, tables are retried ONE AT A TIME to isolate the exact
        # offending table (or prove it's a batch-composition interaction,
        # not a single bad table) -- modular, isolate first, ask questions
        # later.
        #
        # CRITICAL: self._maybe_sync() is called at the END of each try
        # block, INSIDE it, not after. CUDA errors are asynchronous -- a
        # bad kernel launched during, say, table encoding here doesn't
        # necessarily raise a Python exception until the NEXT GPU->CPU
        # sync point, which could be several unrelated calls (even a
        # different epoch's VALIDATION pass) later, well outside any of
        # these try/except blocks. Forcing a sync here, immediately after
        # each stage's own GPU work, makes a deferred error from THIS
        # stage surface HERE, inside the matching except clause, instead
        # of silently erupting downstream where it's much harder to
        # attribute -- this is exactly what let a training-induced crash
        # slip past this instrumentation and surface later inside
        # _corpus_scores instead (see that method's own crash history).
        t_start = time.perf_counter()
        try:
            Q, query_mask = self.query_encoder(queries)  # [Bq, L, k], [Bq, L]
            Q = Q * query_mask.unsqueeze(-1)  # zero out padding-token vectors
            self._maybe_sync()
        except Exception:
            print(
                f"[_score_batch] CRASHED during QUERY encoding "
                f"({len(queries)} question(s)). first few: {queries[:5]!r}"
            )
            raise
        t_query = time.perf_counter()

        try:
            X, col_mask, row_mask, cell_mask = self.model.forward_batch_cellwise(tables)
            self._maybe_sync()
        except Exception:
            print(
                f"[_score_batch] CRASHED during TABLE encoding "
                f"({len(tables)} table(s)) -- isolating which one ..."
            )
            self._isolate_table_encoding_failure(tables)
            raise  # unreachable if isolation above found+raised the culprit
        t_table = time.perf_counter()

        try:
            scores = cross_score_queries_tables(self.scorer, self.scoring_mode, Q, X, row_mask, col_mask)
            self._maybe_sync()
        except Exception:
            print(
                f"[_score_batch] CRASHED during SCORING. "
                f"Q shape={tuple(Q.shape)}, X shape={tuple(X.shape)}, "
                f"row_mask shape={tuple(row_mask.shape)}, col_mask shape={tuple(col_mask.shape)}, "
                f"scoring_mode={self.scoring_mode!r}"
            )
            raise
        t_score = time.perf_counter()

        if self.profile:
            # self.model (either our own TableEncoder or a
            # BaselineCellwiseAdapter wrapping bert/tapas/tabbie/strubert/
            # turl/hytrel) always records _last_frozen_s/_last_network_s
            # on itself during forward_batch_cellwise -- the frozen-
            # backbone pass (cacheable, ~0 on a warm cache hit) vs the
            # trainable network on top of it (never cacheable). Falls
            # back to lumping everything under table_s if a model
            # somehow doesn't expose the split (shouldn't happen for any
            # of the 7 current encoders, but keeps this from crashing if
            # one is added without it).
            frozen_s = getattr(self.model, "_last_frozen_s", None)
            network_s = getattr(self.model, "_last_network_s", None)
            table_s = t_table - t_query
            if frozen_s is None or network_s is None:
                frozen_s, network_s = table_s, 0.0
            self._record_profile(
                n_queries=len(queries),
                n_tables=len(tables),
                query_s=t_query - t_start,
                frozen_s=frozen_s,
                network_s=network_s,
                score_s=t_score - t_table,
            )

        return scores

    def _isolate_table_encoding_failure(self, tables: list[Table]) -> None:
        """Retry table encoding ONE TABLE AT A TIME to pinpoint exactly
        which table (if any single one) triggers the failure a batched
        forward_batch_cellwise call just hit. Raises a RuntimeError naming
        the offending table's id/shape/headers the moment one is found;
        if every table succeeds in isolation, raises a different error
        pointing at batch-composition (e.g. padding to a shared max
        length) as the likely cause instead of any single table's data.
        """
        for i, t in enumerate(tables):
            try:
                self.model.forward_batch_cellwise([t])
            except Exception as e:
                raise RuntimeError(
                    f"table encoding fails in ISOLATION (batch of 1) on "
                    f"table {i}/{len(tables)}: table_id={t.table_id!r}, "
                    f"rows={t.num_rows}, cols={t.num_columns}, "
                    f"headers={list(t.headers)[:10]!r}"
                ) from e
        raise RuntimeError(
            f"table encoding failed as a batch of {len(tables)}, but EVERY "
            f"table in it succeeded when retried one at a time -- this "
            f"points at a batch-composition interaction (e.g. padding/"
            f"stacking multiple tables to a shared max sequence length), "
            f"not a single bad table's data. table_ids in this batch: "
            f"{[t.table_id for t in tables]}"
        )

    def _maybe_sync(self) -> None:
        """Force any CUDA error queued by a kernel that just ran to
        surface HERE, synchronously, instead of deferring to whatever
        GPU->CPU sync happens to come next -- which can be several
        unrelated calls later (a different training step, a different
        epoch's validation pass), making the real cause nearly
        impossible to attribute. Called at the end of every try block in
        _score_batch/train_step that just did GPU work. No-op on CPU
        (torch.cuda.synchronize would raise if no CUDA device exists)."""
        dev = self.device
        is_cuda = (isinstance(dev, str) and dev.startswith("cuda")) or (
            isinstance(dev, torch.device) and dev.type == "cuda"
        )
        if is_cuda:
            torch.cuda.synchronize(dev)

    def _record_profile(
        self, n_queries: int, n_tables: int, query_s: float, frozen_s: float, network_s: float, score_s: float
    ) -> None:
        """Accumulate per-stage timings and print a running average every
        self.profile_every steps, so the printed numbers reflect a stable
        average rather than one noisy step. Reset after each print.

        frozen_s/network_s split what used to be a single "table encode"
        bucket: frozen_s is the frozen-backbone pass (BERT/TAPAS -- the
        part disk caching actually helps), network_s is whatever
        trainable table-level architecture sits on top of it (our
        TableLayer stack, tabbie's row/col transformer, strubert's
        vertical/horizontal attention + fuse_proj, etc.) and is NEVER
        cacheable regardless of how warm the frozen cache is."""
        acc = self._profile_accum
        acc["n"] += 1
        acc["n_queries"] += n_queries
        acc["n_tables"] += n_tables
        acc["query_s"] += query_s
        acc["frozen_s"] += frozen_s
        acc["network_s"] += network_s
        acc["score_s"] += score_s

        if acc["n"] >= self.profile_every:
            n = acc["n"]
            total = acc["query_s"] + acc["frozen_s"] + acc["network_s"] + acc["score_s"]
            self._log(
                f"[profile] avg over {n} step(s) -- "
                f"query encode: {1000 * acc['query_s'] / n:.1f}ms "
                f"({100 * acc['query_s'] / total:.0f}%, avg {acc['n_queries'] / n:.0f} queries/step) | "
                f"frozen backbone: {1000 * acc['frozen_s'] / n:.1f}ms "
                f"({100 * acc['frozen_s'] / total:.0f}%, avg {acc['n_tables'] / n:.0f} tables/step) | "
                f"network on top: {1000 * acc['network_s'] / n:.1f}ms "
                f"({100 * acc['network_s'] / total:.0f}%) | "
                f"scoring: {1000 * acc['score_s'] / n:.1f}ms "
                f"({100 * acc['score_s'] / total:.0f}%)"
            )
            for k in acc:
                acc[k] = 0

    def train_step(self, batch: tuple[list[tuple[str, Table]], list[Table]]) -> float:
        self.model.train()
        self.query_encoder.train()
        self.scorer.train()
        self.optimizer.zero_grad()

        cross_scores = self._score_batch(batch)
        loss = query_table_info_nce_loss(cross_scores, temperature=self.temperature)

        # Same async-CUDA-deferral concern as _score_batch: a bad kernel
        # queued during backward() or the optimizer step would otherwise
        # silently defer to some LATER unrelated sync point (e.g. the next
        # validation pass's index_copy_) instead of raising here, where we
        # can actually attribute it to this step/batch. See _score_batch's
        # comment block for the full rationale.
        try:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._trainable_params(), self.grad_clip_norm)
            self.optimizer.step()
            self._maybe_sync()
        except Exception:
            # Do NOT touch loss.item() here -- after a CUDA device-side
            # assert, the CUDA context itself is poisoned and ANY further
            # CUDA call (including a GPU->CPU sync like .item()) just
            # raises again, masking the real traceback. Print only what's
            # already on CPU.
            print(f"[train_step] CRASHED during backward/optimizer step (global_step={self.global_step})")
            raise

        self.scheduler.step()
        self.global_step += 1

        return loss.item()

    def _corpus_scores(
        self,
        examples: list[tuple[str, str, list[str]]],
        corpus_tables: list[Table],
        query_batch_size: int | None = None,
        table_batch_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Scores EVERY table in corpus_tables against each query in
        `examples` -- not just in-batch negatives the way
        train_step/evaluate's InfoNCE loss does -- and returns the raw
        [n_queries, n_corpus] score matrix plus the matching positive
        mask. Shared groundwork for evaluate_map (MAP only) and
        evaluate_ranking_metrics (MAP + MRR from the same pass) -- not
        meant to be called directly by training scripts.

        examples:      list of (question, db_id, table_names) -- kept as
                       plain tuples rather than importing
                       src.data.synsql_dataset.QueryTableExample, since
                       this trainer is deliberately decoupled from any
                       specific data source (see module docstring).
        corpus_tables: the FIXED, un-split table corpus (see
                       scripts/build_query_splits.py's docstring -- only
                       QUERIES are split into train/val/test, the corpus
                       never is). Pass the SAME corpus_tables for every
                       split so MAP numbers are comparable across splits
                       and across epochs. Each Table's table_id must
                       follow the "{db_id}#sep#{table_name}" convention
                       SynSQLTableDataset.get_table() uses, since that's
                       how a query's positive table_names get matched
                       against corpus_tables here.

        query_batch_size: None (default) encodes every query in `examples`
                       in ONE QueryEncoder call -- val/test query sets are
                       plain text through a BERT-sized encoder and are
                       typically hundreds to a few thousand examples, far
                       smaller than the corpus, so this is normally cheap
                       and lets scoring run as one batched op per corpus
                       chunk. Pass an int only if you have an unusually
                       large query set that itself doesn't fit in memory.
        table_batch_size: the corpus-chunk size. This is a hard memory
                       constraint, not a leftover Python-loop
                       inefficiency: forward_batch_cellwise pads every
                       table in a chunk to that chunk's own max
                       rows/columns, so encoding the entire 100k+ table
                       corpus in one call would pad every table to the
                       single largest table's shape and blow up memory.
                       Chunking the corpus is therefore unavoidable at
                       this scale; the loop below is the one loop that
                       has to remain. Tables are sorted by (num_columns,
                       num_rows) before chunking so similarly-sized
                       tables land in the same chunk -- this minimizes
                       wasted padding within each chunk, which matters a
                       lot once the corpus spans very different table
                       sizes (as a real ~160k-table corpus will).
        """
        self.model.eval()
        self.query_encoder.eval()
        self.scorer.eval()

        # Defensive re-check: corpus_tables SHOULD already be filtered by
        # SynSQLTableDataset.load_corpus (see _drop_empty_tables there),
        # but that only applies to processes started after that filter
        # existed -- a long-running process that loaded its corpus
        # earlier still carries whatever load_corpus returned at the
        # time. A table with 0 rows or 0 columns here would zero out an
        # entire chunk's row_mask/col_mask dimension and crash
        # MultiScorer's max() reduction (see multi_score.py), losing the
        # whole epoch's progress with no resume support. Cheap to check
        # again right here, and fails loudly with which tables were bad
        # instead of a confusing shape error deep in scoring code.
        bad = [t for t in corpus_tables if t.num_rows == 0 or t.num_columns == 0]
        if bad:
            print(
                f"[_corpus_scores] WARNING: dropping {len(bad)} empty table(s) "
                f"that slipped past load_corpus's own filter (stale in-memory "
                f"corpus from before that filter existed, or a genuine new "
                f"bug -- first few ids: {[t.table_id for t in bad[:5]]}): "
            )
            corpus_tables = [t for t in corpus_tables if t.num_rows > 0 and t.num_columns > 0]

        corpus_ids = [t.table_id for t in corpus_tables]
        id_to_idx = {tid: i for i, tid in enumerate(corpus_ids)}

        n_queries = len(examples)
        n_corpus = len(corpus_tables)
        scores = torch.zeros(n_queries, n_corpus)
        positive_mask = torch.zeros(n_queries, n_corpus)

        for qi, (_question, db_id, table_names) in enumerate(examples):
            for t in table_names:
                tid = f"{db_id}#sep#{t}"
                if tid in id_to_idx:
                    positive_mask[qi, id_to_idx[tid]] = 1.0

        # Sort corpus indices by size before chunking (see table_batch_size
        # docstring above); `order[i]` is the original corpus_tables index
        # of the i-th table in size-sorted order, so results can be
        # scattered back to their true column in `scores` regardless of
        # this reordering.
        order = sorted(
            range(n_corpus),
            key=lambda i: (corpus_tables[i].num_columns, corpus_tables[i].num_rows),
        )

        # Same async-CUDA-deferral concern as _score_batch/train_step (see
        # their comment blocks) -- this loop previously had NO sync/try-
        # except instrumentation at all, so a bad kernel queued by ANY
        # call in here (query encoding, table encoding, or scoring) could
        # surface at whatever unrelated call happened to sync next --
        # including, misleadingly, the index_copy_ two lines below the
        # ACTUAL culprit, or the masked_fill inside a LATER chunk's
        # scoring call. This is exactly why earlier crashes pointed at
        # index_copy_/masked_fill without those actually being the real
        # cause. Every GPU-touching call below now gets its own
        # try/except + self._maybe_sync(), so the NEXT crash here prints
        # which stage AND which corpus chunk (with table ids) it
        # actually happened in.
        with torch.no_grad():
            # Encode every query ONCE, up front -- no outer query-chunk
            # loop at all in the common case (query_batch_size=None).
            try:
                if query_batch_size is None:
                    questions = [q for q, _, _ in examples]
                    Q, query_mask = self.query_encoder(questions)
                    Q = Q * query_mask.unsqueeze(-1)
                    query_chunks = [(0, n_queries, Q)]
                else:
                    query_chunks = []
                    for q_start in range(0, n_queries, query_batch_size):
                        q_chunk = examples[q_start : q_start + query_batch_size]
                        questions = [q for q, _, _ in q_chunk]
                        Qc, query_mask = self.query_encoder(questions)
                        Qc = Qc * query_mask.unsqueeze(-1)
                        query_chunks.append((q_start, q_start + len(q_chunk), Qc))
                self._maybe_sync()
            except Exception:
                print(f"[_corpus_scores] CRASHED during QUERY encoding ({n_queries} question(s))")
                raise

            # The one remaining loop: chunk the corpus (memory-bound, see
            # docstring), encode each chunk once, score it against every
            # query chunk with a single vectorized cross_score_queries_tables
            # call (no per-query Python loop -- see MultiScorer.score_cross).
            for c_start in range(0, n_corpus, table_batch_size):
                idx_chunk = order[c_start : c_start + table_batch_size]
                c_chunk = [corpus_tables[i] for i in idx_chunk]

                try:
                    X, col_mask, row_mask, cell_mask = self.model.forward_batch_cellwise(c_chunk)
                    self._maybe_sync()
                except Exception:
                    print(
                        f"[_corpus_scores] CRASHED during TABLE encoding, "
                        f"corpus chunk starting at {c_start} ({len(c_chunk)} table(s)) "
                        f"-- isolating which one ..."
                    )
                    self._isolate_table_encoding_failure(c_chunk)
                    raise  # unreachable if isolation above found+raised the culprit

                idx_tensor = torch.tensor(idx_chunk, dtype=torch.long)
                for q_lo, q_hi, Q in query_chunks:
                    try:
                        cross = cross_score_queries_tables(
                            self.scorer, self.scoring_mode, Q, X, row_mask, col_mask
                        )  # [q_hi - q_lo, len(idx_chunk)]
                        scores[q_lo:q_hi].index_copy_(1, idx_tensor, cross.to(scores.device))
                        self._maybe_sync()
                    except Exception:
                        print(
                            f"[_corpus_scores] CRASHED during SCORING. "
                            f"corpus chunk starting at {c_start} ({len(c_chunk)} table(s), "
                            f"table_ids={[t.table_id for t in c_chunk][:5]!r}...), "
                            f"query range [{q_lo}:{q_hi}], "
                            f"Q shape={tuple(Q.shape)}, X shape={tuple(X.shape)}, "
                            f"row_mask shape={tuple(row_mask.shape)}, col_mask shape={tuple(col_mask.shape)}, "
                            f"scoring_mode={self.scoring_mode!r}"
                        )
                        raise

        return scores, positive_mask

    def evaluate_map(
        self,
        examples: list[tuple[str, str, list[str]]],
        corpus_tables: list[Table],
        query_batch_size: int | None = None,
        table_batch_size: int = 32,
    ) -> float:
        """Mean Average Precision only -- see _corpus_scores' docstring
        for every argument. If you also want MRR from the same ranking,
        call evaluate_ranking_metrics instead: this and that both score
        the corpus from scratch, so calling both back to back re-runs
        the (expensive) encoder forward passes twice for no reason."""
        scores, positive_mask = self._corpus_scores(
            examples, corpus_tables, query_batch_size, table_batch_size
        )
        return compute_map(scores, positive_mask)

    def evaluate_ranking_metrics(
        self,
        examples: list[tuple[str, str, list[str]]],
        corpus_tables: list[Table],
        query_batch_size: int | None = None,
        table_batch_size: int = 32,
    ) -> dict:
        """MAP and MRR together, from ONE corpus-scoring pass (see
        _corpus_scores) and one ranking pass per query (see
        src/eval/retrieval_metrics.py::compute_ranking_metrics) -- use
        this instead of evaluate_map when you want both metrics, to
        avoid re-scoring the whole corpus twice.

        returns: {"map": float, "mrr": float}
        """
        scores, positive_mask = self._corpus_scores(
            examples, corpus_tables, query_batch_size, table_batch_size
        )
        return compute_ranking_metrics(scores, positive_mask)

    def fit(
        self,
        batch_fn: Callable[[], list[tuple[list[tuple[str, Table]], list[Table]]]],
        num_epochs: int,
        steps_per_epoch: int,
        val_examples: list[tuple[str, str, list[str]]],
        corpus_tables: list[Table],
        patience: int = 3,
        log_every: int = 50,
        resume_from: str | None = None,
        val_query_batch_size: int | None = None,
        on_checkpoint: Callable[[], None] | None = None,
    ) -> float:
        """
        Early stopping on validation MAP (not loss) -- per-instruction:
        after each epoch, compute val MAP over the full fixed corpus; if
        it's better than the best seen so far, save the model (only
        then); if it hasn't improved for `patience` consecutive epochs,
        stop. The number that matters at the end is best validation
        MAP, returned here and logged -- nothing else is used for model
        selection.

        batch_fn: called ONCE PER EPOCH (not once total) to build a
            fresh list of training batches -- e.g.
            scripts/finetune_query_table.py's resolve_train_batches
            bound to this trainer's own seeded RNG. This is deliberate,
            not just an epoch-order reshuffle: calling it fresh every
            epoch also re-samples which positive table is used for a
            query with multiple valid positives, and re-samples which
            hard-negative tables get appended, so a query doesn't see
            the exact same in-batch/hard negatives on every single
            epoch of the whole run.

        val_examples/corpus_tables: see evaluate_map's docstring.
        val_query_batch_size: forwarded as evaluate_ranking_metrics'
            query_batch_size for the per-epoch validation pass -- see
            _corpus_scores' docstring for what it does. Left at its
            default (None -- encode every val_examples question in ONE
            QueryEncoder call) unless val_examples is itself large
            enough that a single unbatched call risks OOMing; pass an
            int (e.g. 1000-4000) if so. NOTE this only guards the
            per-epoch val pass -- the separate FINAL test-set evaluation
            (scripts/train_model.py's call to evaluate_map after fit()
            returns) needs its own query_batch_size passed directly to
            that call, since it isn't routed through fit() at all and
            typically uses the FULL, unsampled test split.
        on_checkpoint: called (no args) immediately after every
            save_checkpoint() -- i.e. every time val MAP improves, same
            frequency as best_model.pt itself. Intended for saving any
            frozen-substep caches (text/table/frozen/query -- see
            adapter.py/tabbie.py/strubert.py/query_encoder.py's
            save_*cache methods) alongside the model checkpoint, so a
            crash later in the SAME run doesn't lose all cache progress
            accumulated since the last save -- this is exactly what
            happened to an earlier run that crashed during the final
            test-set evaluation, after training had already finished:
            best_model.pt existed, but nothing had EVER been saved to
            any cache path, since the only save call was previously at
            the very end of the whole script, after evaluate_map.
        """
        import time

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
        best_map = -1.0
        epochs_without_improvement = 0

        for epoch in range(start_epoch, num_epochs):
            # Rebuilt from scratch every epoch (see batch_fn's docstring
            # above) -- not just a reshuffle of one fixed list. Uses this
            # trainer's own seeded RNG internally (not the bare random
            # module) so the whole run is still reproducible given the
            # same seed, while still giving every epoch a genuinely
            # different batch order, positive-table choice, and
            # hard-negative sample.
            batches = batch_fn()

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
            epoch_train_elapsed = time.time() - epoch_start

            val_start = time.time()
            val_metrics = self.evaluate_ranking_metrics(
                val_examples, corpus_tables, query_batch_size=val_query_batch_size
            )
            val_map, val_mrr = val_metrics["map"], val_metrics["mrr"]
            val_elapsed = time.time() - val_start

            self._log(
                f"[finetune] epoch {epoch} summary: "
                f"epoch loss {epoch_avg:.4f} | "
                f"val MAP {val_map:.4f} | val MRR {val_mrr:.4f} | "
                f"train time {epoch_train_elapsed:.1f}s | val time {val_elapsed:.1f}s"
            )

            if val_map > best_map:
                best_map = val_map
                epochs_without_improvement = 0
                self.save_checkpoint(epoch, extra={"val_map": val_map, "val_mrr": val_mrr})
                if on_checkpoint is not None:
                    on_checkpoint()
                self._log(f"== [finetune] epoch {epoch} done (NEW BEST val MAP) ==")
            else:
                epochs_without_improvement += 1
                self._log(
                    f"== [finetune] epoch {epoch} done "
                    f"(best val MAP so far: {best_map:.4f}, "
                    f"{epochs_without_improvement}/{patience} epoch(s) without improvement) =="
                )
                if epochs_without_improvement >= patience:
                    self._log(
                        f"[finetune] early stopping at epoch {epoch}: no val MAP "
                        f"improvement for {patience} epoch(s)."
                    )
                    break

        self._log(f"[finetune] training complete. best validation MAP: {best_map:.4f}")
        return best_map

    def evaluate(self, val_batches: list[tuple[list[tuple[str, Table]], list[Table]]]) -> float:
        """In-batch-negative InfoNCE loss over pre-built batches --
        diagnostic only (e.g. to sanity-check training is proceeding
        sensibly step to step). NOT used for early stopping or model
        selection -- see evaluate_map for that."""
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

    def save_checkpoint(self, epoch: int, extra: dict | None = None) -> None:
        """Overwrites a single best_model.pt -- FinetuneTrainer only
        ever saves when val MAP improves (see fit()), so there's no
        reason to accumulate one file per epoch the way PretrainTrainer
        does; the only checkpoint worth keeping is the best one."""
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "query_encoder_state_dict": self.query_encoder.state_dict(),
            "scorer_state_dict": self.scorer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        self._log(f"saved new best checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.query_encoder.load_state_dict(checkpoint["query_encoder_state_dict"])
        self.scorer.load_state_dict(checkpoint["scorer_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]