"""Train the 128-D model directly from deterministic prepared pickle streams."""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
from torch.optim import AdamW

from src.data.prepared_batches import iter_prepared_batches, read_prepared_metadata
from src.models.prepared_table_encoder import PreparedQueryEncoder, PreparedTableEncoder
from src.scoring.multi_score import MultiScorer
from src.training.losses import cross_score_queries_tables, query_table_info_nce_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--channel_mix_hidden_dim", type=int, default=512)
    parser.add_argument("--nonlinearity", default="sigmoid")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--table_microbatch_cell_budget", type=int, default=None)
    parser.add_argument("--table_microbatch_max_tables", type=int, default=None)
    parser.add_argument("--poll_seconds", type=float, default=5.0)
    args = parser.parse_args()

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

    table_model = PreparedTableEncoder(
        embed_dim=dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        channel_mix_hidden_dim=args.channel_mix_hidden_dim,
        nonlinearity=args.nonlinearity,
        table_microbatch_cell_budget=args.table_microbatch_cell_budget,
        table_microbatch_max_tables=args.table_microbatch_max_tables,
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
    log_path = os.path.join(args.checkpoint_dir, "train.log")
    with open(log_path, "a", buffering=1) as log:
        global_step = 0
        start_time = time.time()
        processed = set()
        finalized_epochs = set()
        epoch_losses: dict[int, list[float]] = {}
        epoch_steps: dict[int, int] = {}

        def save_checkpoint(path: str, epoch: int, shard: int) -> None:
            losses = epoch_losses.get(epoch, [])
            torch.save(
                {
                    "epoch": epoch,
                    "shard": shard,
                    "global_step": global_step,
                    "metadata": metadata,
                    "table_model_state_dict": table_model.state_dict(),
                    "query_model_state_dict": query_model.state_dict(),
                    "scorer_state_dict": scorer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch_loss": sum(losses) / max(1, len(losses)),
                },
                path,
            )

        while True:
            paths = sorted(
                glob.glob(os.path.join(args.prepared_dir, "epoch_*_shard_*.pkl"))
            )
            new_paths = [path for path in paths if path not in processed]
            for path in new_paths:
                current = read_prepared_metadata(path)
                for key in (
                    "projection_dim", "projection_seed", "model_name", "max_rows", "max_columns"
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
                for shard_step, cpu_batch in enumerate(iter_prepared_batches(path, validate=True)):
                    if step_in_epoch == 0:
                        bq, length = cpu_batch.query_features.shape[:2]
                        bt, n_cols = cpu_batch.col_mask.shape
                        n_rows = cpu_batch.row_mask.shape[1]
                        score_gib = 2 * bq * bt * length * n_cols * n_rows * 4 / 1024**3
                        message = (
                            f"[prepared] epoch {epoch} first batch: Bq={bq}, Bt={bt}, "
                            f"L={length}, N={n_cols}, M={n_rows}; row-match forward "
                            f"temporaries >= {score_gib:.2f} GiB"
                        )
                        print(message, flush=True)
                        log.write(message + "\n")

                    batch = cpu_batch.materialize(device)
                    optimizer.zero_grad(set_to_none=True)
                    q = query_model(batch.query_features, batch.query_mask)
                    x = table_model(
                        batch.cell_features,
                        batch.header_features,
                        batch.row_mask,
                        batch.col_mask,
                    )
                    scores = cross_score_queries_tables(
                        scorer, "row_match", q, x,
                        batch.row_mask.float(), batch.col_mask.float(),
                    )
                    loss = query_table_info_nce_loss(
                        scores,
                        positive_mask=batch.positive_mask,
                        temperature=args.temperature,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm)
                    optimizer.step()

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
                epoch_steps[epoch] = step_in_epoch
                processed.add(path)
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
                    read_prepared_metadata(path).get("epoch") == epoch for path in processed
                )
                if epoch not in finalized_epochs and consumed_shards >= expected_shards:
                    save_checkpoint(
                        os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch}.pt"),
                        epoch,
                        expected_shards - 1,
                    )
                    finalized_epochs.add(epoch)
                    losses = epoch_losses.get(epoch, [])
                    message = (
                        f"[prepared] epoch {epoch} complete: "
                        f"loss {sum(losses)/max(1, len(losses)):.4f}"
                    )
                    print(message, flush=True)
                    log.write(message + "\n")

            producer_done = os.path.exists(
                os.path.join(args.prepared_dir, "PREPARATION_COMPLETE")
            )
            if producer_done and not [path for path in paths if path not in processed]:
                break
            if not new_paths:
                time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
