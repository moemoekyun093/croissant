"""Train the 128-D model directly from deterministic prepared pickle streams."""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
from torch.optim import AdamW

from src.data.prepared_batches import (
    iter_prepared_batches,
    prefetch_iterable,
    read_prepared_metadata,
)
from src.models.prepared_table_encoder import (
    PreparedQueryEncoder,
    PreparedTabbieEncoder,
    PreparedTableEncoder,
    PreparedTurlEncoder,
)
from src.scoring.multi_score import MultiScorer
from src.training.losses import cross_score_queries_tables, query_table_info_nce_loss
from src.training.prepared_evaluator import evaluate_prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help="checkpoint to resume; by default checkpoint_latest.pt is loaded "
        "automatically when it exists",
    )
    parser.add_argument(
        "--skip_resume",
        action="store_true",
        help="start fresh even if checkpoint_latest.pt exists",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoder", choices=("ours", "tabbie", "turl"), default="ours")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--channel_mix_hidden_dim", type=int, default=512)
    parser.add_argument("--tabbie_ffn_hidden_dim", type=int, default=512)
    parser.add_argument("--turl_ffn_hidden_dim", type=int, default=512)
    parser.add_argument("--turl_attention_budget", type=int, default=2_000_000)
    parser.add_argument("--nonlinearity", default="sigmoid")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_every", type=int, default=20)
    parser.add_argument("--val_prepared_dir", default=None)
    parser.add_argument("--eval_query_batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--keep_epoch_checkpoints",
        action="store_true",
        help="retain checkpoint_epochN.pt snapshots; default keeps only "
        "overwritten latest and best checkpoints to avoid disk bloat",
    )
    parser.add_argument("--table_microbatch_cell_budget", type=int, default=None)
    parser.add_argument("--table_microbatch_max_tables", type=int, default=None)
    parser.add_argument(
        "--score_table_chunk_size",
        type=int,
        default=None,
        help="score this many candidate tables at once, then concatenate the "
        "complete [Bq,Bt] matrix before the unchanged InfoNCE loss",
    )
    parser.add_argument("--poll_seconds", type=float, default=5.0)
    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=2,
        help="background NAS read/unpickle queue depth (default: 2; 0 disables)",
    )
    parser.add_argument(
        "--amp_dtype",
        choices=("none", "bfloat16"),
        default="none",
        help="optional CUDA mixed precision; none preserves FP32 training",
    )
    args = parser.parse_args()
    if args.score_table_chunk_size is not None and args.score_table_chunk_size <= 0:
        parser.error("--score_table_chunk_size must be positive or omitted")
    if args.profile_every <= 0:
        parser.error("--profile_every must be positive")
    if args.eval_query_batch_size <= 0 or args.patience <= 0:
        parser.error("--eval_query_batch_size and --patience must be positive")
    if args.prefetch_batches < 0:
        parser.error("--prefetch_batches must be non-negative")
    if args.poll_seconds <= 0:
        parser.error("--poll_seconds must be positive")
    if args.amp_dtype != "none" and not args.device.startswith("cuda"):
        parser.error("--amp_dtype currently requires a CUDA device")
    if args.skip_resume and args.resume_checkpoint is not None:
        parser.error("--skip_resume and --resume_checkpoint are mutually exclusive")

    # A producer publishes only closed, atomically-renamed *.pkl shards.
    # Wait for the first one so preparation and training can be launched
    # simultaneously in either order.
    first_paths = []
    while not first_paths:
        first_paths = sorted(
            glob.glob(os.path.join(args.prepared_dir, "epoch_*_shard_*.pkl"))
        )
        if not first_paths:
            if os.path.exists(os.path.join(args.prepared_dir, "PREPARATION_COMPLETE")):
                parser.error("preparation completed without publishing any pickle shards")
            print("[prepared] waiting for the first completed pickle shard ...", flush=True)
            time.sleep(args.poll_seconds)
    metadata = read_prepared_metadata(first_paths[0])
    dim = int(metadata["projection_dim"])
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else None

    if args.encoder == "ours":
        table_model = PreparedTableEncoder(
            embed_dim=dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            channel_mix_hidden_dim=args.channel_mix_hidden_dim,
            nonlinearity=args.nonlinearity,
            table_microbatch_cell_budget=args.table_microbatch_cell_budget,
            table_microbatch_max_tables=args.table_microbatch_max_tables,
        ).to(device)
    elif args.encoder == "tabbie":
        table_model = PreparedTabbieEncoder(
            embed_dim=dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_hidden_dim=args.tabbie_ffn_hidden_dim,
            max_rows=int(metadata["max_rows"]) + 1,
            max_columns=int(metadata["max_columns"]),
            table_microbatch_cell_budget=args.table_microbatch_cell_budget,
            table_microbatch_max_tables=args.table_microbatch_max_tables,
        ).to(device)
    else:
        table_model = PreparedTurlEncoder(
            embed_dim=dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_hidden_dim=args.turl_ffn_hidden_dim,
            attention_budget=args.turl_attention_budget,
        ).to(device)
    query_model = PreparedQueryEncoder(dim, dim).to(device)
    scorer = MultiScorer().to(device)
    parameters = [
        parameter
        for module in (table_model, query_model, scorer)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    resume_path = args.resume_checkpoint
    if resume_path is None and not args.skip_resume:
        automatic = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
        if os.path.exists(automatic):
            resume_path = automatic

    resume_state = None
    if resume_path is not None:
        if not os.path.exists(resume_path):
            parser.error(f"resume checkpoint does not exist: {resume_path}")
        try:
            resume_state = torch.load(
                resume_path, map_location=device, weights_only=True
            )
        except TypeError:
            resume_state = torch.load(resume_path, map_location=device)
        checkpoint_encoder = resume_state.get("encoder")
        if checkpoint_encoder is not None and checkpoint_encoder != args.encoder:
            raise ValueError(
                f"checkpoint encoder {checkpoint_encoder!r} does not match "
                f"--encoder {args.encoder!r}"
            )
        checkpoint_metadata = resume_state.get("metadata", {})
        for key in (
            "projection_dim",
            "projection_seed",
            "model_name",
            "max_rows",
            "max_columns",
            "batch_size",
            "n_hard_negatives",
            "split_sha256",
            "questions_sha256",
        ):
            if checkpoint_metadata.get(key) != metadata.get(key):
                raise ValueError(
                    f"resume checkpoint prepared metadata mismatch for {key!r}"
                )
        table_model.load_state_dict(resume_state["table_model_state_dict"])
        query_model.load_state_dict(resume_state["query_model_state_dict"])
        scorer.load_state_dict(resume_state["scorer_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])

    log_path = os.path.join(args.checkpoint_dir, "train.log")
    with open(log_path, "a", buffering=1) as log:
        trainable_parameters = sum(parameter.numel() for parameter in parameters)
        startup = (
            f"[prepared] encoder={args.encoder} projection_dim={dim} "
            f"trainable_parameters={trainable_parameters:,} "
            f"prefetch_batches={args.prefetch_batches} amp_dtype={args.amp_dtype}"
        )
        print(startup, flush=True)
        log.write(startup + "\n")
        if args.val_prepared_dir is None:
            validation_startup = (
                "[prepared-val] DISABLED: no --val_prepared_dir was supplied; "
                "this run cannot report validation MAP/MRR or select best_model.pt"
            )
        else:
            validation_marker = os.path.join(
                args.val_prepared_dir, "PREPARATION_COMPLETE"
            )
            validation_state = (
                "ready" if os.path.exists(validation_marker) else "still preparing"
            )
            validation_startup = (
                f"[prepared-val] ENABLED: {args.val_prepared_dir}; "
                f"MAP/MRR every completed epoch, patience={args.patience}; "
                f"validation artifacts are {validation_state}"
            )
        print(validation_startup, flush=True)
        log.write(validation_startup + "\n")
        global_step = int(resume_state.get("global_step", 0)) if resume_state else 0
        start_time = time.time()
        processed: set[str] = set(
            resume_state.get("processed_shards", []) if resume_state else []
        )
        finalized_epochs: set[int] = set(
            resume_state.get("finalized_epochs", []) if resume_state else []
        )
        epoch_losses: dict[int, list[float]] = dict(
            resume_state.get("epoch_losses", {}) if resume_state else {}
        )
        epoch_steps: dict[int, int] = dict(
            resume_state.get("epoch_steps", {}) if resume_state else {}
        )
        best_val_map = (
            float(resume_state.get("best_val_map", float("-inf")))
            if resume_state
            else float("-inf")
        )
        best_epoch = resume_state.get("best_epoch") if resume_state else None
        epochs_without_improvement = int(
            resume_state.get("epochs_without_improvement", 0)
        ) if resume_state else 0

        # Checkpoints written before processed_shards was added still carry
        # an exact (epoch, shard) boundary. Infer the completed prefix once;
        # shards are published strictly in epoch/shard order.
        if resume_state is not None and not processed:
            boundary = (int(resume_state["epoch"]), int(resume_state["shard"]))
            for existing_path in sorted(
                glob.glob(os.path.join(args.prepared_dir, "epoch_*_shard_*.pkl"))
            ):
                existing_metadata = read_prepared_metadata(existing_path)
                position = (
                    int(existing_metadata["epoch"]),
                    int(existing_metadata.get("shard", 0)),
                )
                if position <= boundary:
                    processed.add(os.path.basename(existing_path))
            finalized_epochs.update(range(boundary[0]))
            if boundary[0] not in epoch_steps:
                batches_per_shard = int(metadata.get("batches_per_shard", 20))
                epoch_steps[boundary[0]] = (boundary[1] + 1) * batches_per_shard
            if boundary[0] not in epoch_losses and "epoch_loss" in resume_state:
                # Older checkpoints retained only the running average. Fill
                # an equivalent weighted prefix so the eventual full-epoch
                # average remains correct after appending resumed steps.
                epoch_losses[boundary[0]] = [float(resume_state["epoch_loss"])] * int(
                    epoch_steps[boundary[0]]
                )

        if resume_state is not None:
            resume_message = (
                f"[prepared] resumed {args.encoder} from {resume_path}: "
                f"epoch={resume_state['epoch']} shard={resume_state['shard']} "
                f"global_step={global_step}, skipped_shards={len(processed)}"
            )
            print(resume_message, flush=True)
            log.write(resume_message + "\n")
        profile_totals = {
            "pickle load": 0.0,
            "GPU materialize": 0.0,
            "query adapter": 0.0,
            "table contextualization": 0.0,
            "scoring + loss": 0.0,
            "backward + optimizer": 0.0,
        }
        profile_steps = 0

        def report(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")

        def profile_sync() -> None:
            if args.profile and device.type == "cuda":
                torch.cuda.synchronize(device)

        def save_checkpoint(path: str, epoch: int, shard: int) -> None:
            losses = epoch_losses.get(epoch, [])
            state = {
                "epoch": epoch,
                "shard": shard,
                "global_step": global_step,
                "metadata": metadata,
                "encoder": args.encoder,
                "table_model_state_dict": table_model.state_dict(),
                "query_model_state_dict": query_model.state_dict(),
                "scorer_state_dict": scorer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch_loss": sum(losses) / max(1, len(losses)),
                "processed_shards": sorted(processed),
                "finalized_epochs": sorted(finalized_epochs),
                "epoch_losses": epoch_losses,
                "epoch_steps": epoch_steps,
                "best_val_map": best_val_map,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "training_config": {
                    "amp_dtype": args.amp_dtype,
                    "prefetch_batches": args.prefetch_batches,
                },
                "model_config": {
                    "encoder": args.encoder,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "channel_mix_hidden_dim": args.channel_mix_hidden_dim,
                    "tabbie_ffn_hidden_dim": args.tabbie_ffn_hidden_dim,
                    "nonlinearity": args.nonlinearity,
                    "turl_ffn_hidden_dim": args.turl_ffn_hidden_dim,
                    "turl_attention_budget": args.turl_attention_budget,
                },
            }
            partial = path + ".partial"
            torch.save(
                state,
                partial,
            )
            os.replace(partial, path)

        stop_training = False
        while True:
            paths = sorted(
                glob.glob(os.path.join(args.prepared_dir, "epoch_*_shard_*.pkl"))
            )
            new_paths = [
                path for path in paths if os.path.basename(path) not in processed
            ]
            # Consume one shard per outer pass so an epoch-completion marker
            # is handled immediately after that epoch's final shard. If an
            # existing backlog spans many epochs, processing the whole list
            # first would validate every old epoch using the same newest
            # model state, making best-checkpoint selection meaningless.
            new_paths = new_paths[:1]
            for path in new_paths:
                current = read_prepared_metadata(path)
                for key in (
                    "projection_dim",
                    "projection_seed",
                    "model_name",
                    "max_rows",
                    "max_columns",
                    "batch_size",
                    "n_hard_negatives",
                    "split_sha256",
                    "questions_sha256",
                ):
                    if current.get(key) != metadata.get(key):
                        raise ValueError(f"prepared shard metadata mismatch for {key!r}: {path}")
                epoch = int(current["epoch"])
                shard = int(current.get("shard", 0))
                losses = epoch_losses.setdefault(epoch, [])
                step_in_epoch = epoch_steps.get(epoch, 0)

                table_model.train()
                query_model.train()
                scorer.train()
                batch_iterator = prefetch_iterable(
                    iter_prepared_batches(path, validate=True),
                    depth=args.prefetch_batches,
                )
                previous_step_end = time.perf_counter()
                for shard_step, cpu_batch in enumerate(batch_iterator):
                    pickle_load_s = time.perf_counter() - previous_step_end
                    if step_in_epoch == 0:
                        bq, length = cpu_batch.query_features.shape[:2]
                        bt, n_cols = cpu_batch.col_mask.shape
                        n_rows = cpu_batch.row_mask.shape[1]
                        score_gib = 2 * bq * bt * length * n_cols * n_rows * 4 / 1024**3
                        active_tables = (
                            bt
                            if args.score_table_chunk_size is None
                            else min(bt, args.score_table_chunk_size)
                        )
                        active_score_gib = (
                            2
                            * bq
                            * active_tables
                            * length
                            * n_cols
                            * n_rows
                            * 4
                            / 1024**3
                        )
                        message = (
                            f"[prepared] epoch {epoch} first batch: Bq={bq}, Bt={bt}, "
                            f"L={length}, N={n_cols}, M={n_rows}; row-match forward "
                            f"temporaries >= {score_gib:.2f} GiB unchunked, "
                            f">= {active_score_gib:.2f} GiB active with "
                            f"table chunk={active_tables}"
                        )
                        print(message, flush=True)
                        log.write(message + "\n")

                    stage_started = time.perf_counter()
                    batch = cpu_batch.materialize(
                        device,
                        dtype=torch.float32 if amp_dtype is None else amp_dtype,
                    )
                    profile_sync()
                    materialize_s = time.perf_counter() - stage_started
                    optimizer.zero_grad(set_to_none=True)

                    stage_started = time.perf_counter()
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_dtype is not None,
                    ):
                        q = query_model(batch.query_features, batch.query_mask)
                    profile_sync()
                    query_s = time.perf_counter() - stage_started

                    stage_started = time.perf_counter()
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_dtype is not None,
                    ):
                        x = table_model(
                            batch.cell_features,
                            batch.header_features,
                            batch.row_mask,
                            batch.col_mask,
                        )
                    profile_sync()
                    table_s = time.perf_counter() - stage_started

                    stage_started = time.perf_counter()
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_dtype is not None,
                    ):
                        if args.score_table_chunk_size is None:
                            scores = cross_score_queries_tables(
                                scorer, "row_match", q, x,
                                batch.row_mask.float(), batch.col_mask.float(),
                            )
                        else:
                            score_chunks = []
                            for table_start in range(
                                0, x.shape[0], args.score_table_chunk_size
                            ):
                                table_end = min(
                                    x.shape[0],
                                    table_start + args.score_table_chunk_size,
                                )
                                score_chunks.append(
                                    cross_score_queries_tables(
                                        scorer,
                                        "row_match",
                                        q,
                                        x[table_start:table_end],
                                        batch.row_mask[table_start:table_end].float(),
                                        batch.col_mask[table_start:table_end].float(),
                                    )
                                )
                            scores = torch.cat(score_chunks, dim=1)
                    loss = query_table_info_nce_loss(
                        scores.float(),
                        positive_mask=batch.positive_mask,
                        temperature=args.temperature,
                    )
                    profile_sync()
                    scoring_s = time.perf_counter() - stage_started

                    stage_started = time.perf_counter()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm)
                    optimizer.step()
                    profile_sync()
                    backward_s = time.perf_counter() - stage_started

                    if args.profile:
                        measurements = {
                            "pickle load": pickle_load_s,
                            "GPU materialize": materialize_s,
                            "query adapter": query_s,
                            "table contextualization": table_s,
                            "scoring + loss": scoring_s,
                            "backward + optimizer": backward_s,
                        }
                        for name, seconds in measurements.items():
                            profile_totals[name] += seconds
                        profile_steps += 1
                        if profile_steps >= args.profile_every:
                            total = sum(profile_totals.values())
                            pieces = [
                                f"{name}: {1000 * seconds / profile_steps:.1f}ms "
                                f"({100 * seconds / total:.0f}%)"
                                for name, seconds in profile_totals.items()
                            ]
                            profile_message = (
                                f"[profile] avg over {profile_steps} step(s) -- "
                                + " | ".join(pieces)
                            )
                            print(profile_message, flush=True)
                            log.write(profile_message + "\n")
                            for name in profile_totals:
                                profile_totals[name] = 0.0
                            profile_steps = 0

                    value = loss.item()
                    losses.append(value)
                    global_step += 1
                    step_in_epoch += 1
                    if global_step % args.log_every == 0:
                        recent = losses[-args.log_every :]
                        message = (
                            f"[prepared] epoch {epoch} step {step_in_epoch - 1} "
                            f"global {global_step} loss {value:.4f} "
                            f"avg {sum(recent)/len(recent):.4f} "
                            f"[{(time.time()-start_time)/60:.1f} min]"
                        )
                        print(message, flush=True)
                        log.write(message + "\n")
                    previous_step_end = time.perf_counter()
                epoch_steps[epoch] = step_in_epoch
                processed.add(os.path.basename(path))
                save_checkpoint(
                    os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"), epoch, shard
                )
                message = f"[prepared] consumed epoch {epoch} shard {shard}: {path}"
                print(message, flush=True)
                log.write(message + "\n")

            # An epoch marker is published only after all of its shards.
            for marker in sorted(glob.glob(os.path.join(args.prepared_dir, "epoch_*.complete"))):
                with open(marker, "r", encoding="utf-8") as f:
                    completed = json.load(f)
                epoch = int(completed["epoch"])
                expected_shards = int(completed["shards"])
                consumed_shards = sum(
                    read_prepared_metadata(path).get("epoch") == epoch
                    for path in paths
                    if os.path.basename(path) in processed
                )
                if epoch not in finalized_epochs and consumed_shards >= expected_shards:
                    finalized_epochs.add(epoch)
                    losses = epoch_losses.get(epoch, [])
                    message = (
                        f"[prepared] epoch {epoch} complete: "
                        f"loss {sum(losses)/max(1, len(losses)):.4f}"
                    )
                    print(message, flush=True)
                    log.write(message + "\n")

                    if args.val_prepared_dir is not None:
                        validation_marker = os.path.join(
                            args.val_prepared_dir, "PREPARATION_COMPLETE"
                        )
                        waiting_logged_at = 0.0
                        while not os.path.exists(validation_marker):
                            now = time.monotonic()
                            if now - waiting_logged_at >= 60.0:
                                report(
                                    f"[prepared-val] epoch {epoch}: waiting for "
                                    f"validation preparation to complete at "
                                    f"{args.val_prepared_dir}"
                                )
                                waiting_logged_at = now
                            time.sleep(min(args.poll_seconds, 60.0))
                        report(
                            f"[prepared-val] epoch {epoch}: starting MAP/MRR evaluation"
                        )
                        metrics = evaluate_prepared(
                            args.val_prepared_dir,
                            table_model,
                            query_model,
                            scorer,
                            device,
                            metadata,
                            query_batch_size=args.eval_query_batch_size,
                            progress=report,
                        )
                        metric_message = (
                            f"[prepared-val] epoch {epoch}: MAP {metrics['map']:.4f} "
                            f"MRR {metrics['mrr']:.4f} "
                            f"({int(metrics['n_queries'])} queries, "
                            f"{int(metrics['n_tables'])} tables)"
                        )
                        print(metric_message, flush=True)
                        log.write(metric_message + "\n")
                        if metrics["map"] > best_val_map:
                            best_val_map = metrics["map"]
                            best_epoch = epoch
                            epochs_without_improvement = 0
                            save_checkpoint(
                                os.path.join(args.checkpoint_dir, "best_model.pt"),
                                epoch,
                                expected_shards - 1,
                            )
                            best_message = (
                                f"[prepared-val] NEW BEST epoch {epoch}: "
                                f"MAP {best_val_map:.4f}; saved best_model.pt"
                            )
                            print(best_message, flush=True)
                            log.write(best_message + "\n")
                        else:
                            epochs_without_improvement += 1
                            if epochs_without_improvement >= args.patience:
                                stop_training = True
                                stop_message = (
                                    f"[prepared-val] early stopping after "
                                    f"{epochs_without_improvement} epoch(s) without "
                                    f"MAP improvement; best epoch={best_epoch}, "
                                    f"best MAP={best_val_map:.4f}"
                                )
                                print(stop_message, flush=True)
                                log.write(stop_message + "\n")

                    if args.keep_epoch_checkpoints:
                        save_checkpoint(
                            os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch}.pt"),
                            epoch,
                            expected_shards - 1,
                        )
                    # Persist validation/best/patience state by overwriting
                    # the sole rolling checkpoint rather than accumulating
                    # one optimizer-sized file per epoch.
                    save_checkpoint(
                        os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"),
                        epoch,
                        expected_shards - 1,
                    )

            if stop_training:
                break

            producer_done = os.path.exists(
                os.path.join(args.prepared_dir, "PREPARATION_COMPLETE")
            )
            if producer_done:
                # ``paths`` was captured before processing ``new_paths`` and
                # can be hours old for a large backlog. The producer may have
                # published many more shards (and PREPARATION_COMPLETE) while
                # this pass was training. Always rescan before deciding that
                # every shard has been consumed; using the stale snapshot can
                # silently exit with later epochs untouched.
                final_paths = sorted(
                    glob.glob(os.path.join(args.prepared_dir, "epoch_*_shard_*.pkl"))
                )
                remaining = [
                    path
                    for path in final_paths
                    if os.path.basename(path) not in processed
                ]
                if not remaining:
                    break
            if not new_paths:
                time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
