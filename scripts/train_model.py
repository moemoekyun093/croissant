"""
Unified training script: the SAME ELECTRA pretraining + query-table
finetuning paradigm, run identically for our own TableEncoder or any
baseline encoder (bert/tabbie/strubert/tapas/turl/hytrel), selected via
--encoder. This is what "same training paradigm across all models"
means concretely -- one script, one code path (PretrainTrainer then
FinetuneTrainer, both from src/training/trainer.py, completely
unmodified), only the encoder swapped out via
src/encoding/baseline_encoders/adapter.py::BaselineCellwiseAdapter,
which gives every baseline the same forward_batch_cellwise(tables) ->
(X, col_mask, row_mask, cell_mask) interface our own TableEncoder has.

Runs pretraining then finetuning in one invocation for whichever
--encoder you pick. Every encoder loads the SAME persisted query split
+ fixed corpus (scripts/build_query_splits.py -- run that once first),
so results are directly comparable across models. --embed_dim,
--pretrain_epochs, --finetune_epochs, etc. are shared flags applied
identically regardless of --encoder, for the "consistency in model
parameters (internal dimensions, number of epochs, etc.)" a fair
comparison needs. Checkpoints/logs for each encoder land in their own
subdirectory under --checkpoint_dir so runs for different encoders
never clobber each other.

Finetuning model-selection is early stopping on validation MAP
(--patience epochs without improvement) -- see
FinetuneTrainer.fit/evaluate_map. Final numbers reported: best
validation MAP, and test MAP using that best checkpoint -- nothing else
is used to decide which checkpoint is "best".

Usage:
    # once, shared across every encoder:
    python -m scripts.build_query_splits \\
        --tables_json /path/to/synsql/tables.json \\
        --databases_root /path/to/synsql/databases \\
        --questions_json /path/to/synsql/questions_with_tables.json

    # our own model:
    python -m scripts.train_model --encoder ours \\
        --tables_json /path/to/synsql/tables.json \\
        --databases_root /path/to/synsql/databases \\
        --questions_json /path/to/synsql/questions_with_tables.json \\
        --embed_dim 64 --pretrain_epochs 5 --finetune_epochs 15 --patience 3

    # a baseline, same flags, same everything else:
    python -m scripts.train_model --encoder tabbie \\
        --tables_json /path/to/synsql/tables.json \\
        --databases_root /path/to/synsql/databases \\
        --questions_json /path/to/synsql/questions_with_tables.json \\
        --embed_dim 64 --pretrain_epochs 5 --finetune_epochs 15 --patience 3
"""

import argparse
import datetime
import glob
import json
import os
import random

from src.data.synsql_dataset import SynSQLQueryDataset, SynSQLTableDataset
from src.encoding.baseline_encoders import ENCODER_REGISTRY
from src.encoding.baseline_encoders.adapter import build_baseline_model
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import DiscriminatorHead, TableEncoder, load_pretrained_encoder
from src.training.config import apply_yaml_defaults
from src.training.query_encoder import QueryEncoder
from src.training.trainer import FinetuneTrainer, PretrainTrainer

from scripts.pretrain_electra import make_batches
from scripts.finetune_query_table import cap_columns, resolve_train_batches, to_eval_examples

ENCODER_CHOICES = ["ours"] + sorted(ENCODER_REGISTRY)


def latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Same natural/version-sort convention as scripts/run_pilot.sh's
    shell logic, reimplemented in Python -- picks the highest-epoch
    checkpoint_epoch*.pt in a PretrainTrainer checkpoint_dir."""
    paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch*.pt"))
    if not paths:
        return None

    def epoch_num(path: str) -> int:
        name = os.path.basename(path)
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else -1

    return max(paths, key=epoch_num)


def build_table_model(args):
    if args.encoder == "ours":
        cell_encoder = CellEncoder(
            text_model_name=args.text_model_name,
            output_dim=args.embed_dim,
            text_max_length=args.text_max_length,
            text_trainable=args.text_trainable,
            text_max_batch_size=args.text_max_batch_size,
        )
        return TableEncoder(
            cell_encoder,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            nonlinearity=args.nonlinearity,
            channel_mix_hidden_dim=args.channel_mix_hidden_dim,
        )
    return build_baseline_model(
        args.encoder,
        embed_dim=args.embed_dim,
        model_name=args.model_name,
        num_layers=args.num_layers,
        device=args.device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=ENCODER_CHOICES)
    parser.add_argument("--tables_json", default=None)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--questions_json", required=True)
    parser.add_argument("--split_json", default="configs/splits/query_split.json")
    parser.add_argument("--corpus_json", default="configs/splits/corpus.json")

    # shared architecture knobs -- applied identically across every --encoder
    parser.add_argument("--embed_dim", type=int, help="shared across every encoder, incl. baselines (via projection)")
    parser.add_argument(
        "--num_layers", type=int,
        help="depth of the table-level stack built on top of frozen BERT cell/token "
             "encoding -- applies to ours (RCPE layers), tabbie (row/col transformer "
             "layers), strubert (vertical/horizontal attention layers), turl "
             "(visibility-masked encoder layers), and hytrel (set-attention-pool "
             "layers). Ignored for bert/tapas, which have no comparable on-top stack "
             "(the pretrained backbone IS the whole model for those two papers) -- "
             "see src/encoding/baseline_encoders/adapter.py::_NUM_LAYERS_KWARG.",
    )
    parser.add_argument("--nonlinearity", choices=["sigmoid", "tanh", "relu"], help="only used when --encoder ours")
    parser.add_argument("--channel_mix_hidden_dim", type=int, help="only used when --encoder ours")
    parser.add_argument("--model_name", default=None, help="override the encoder's own default backbone checkpoint (leave unset for TAPAS -- see adapter.py::build_baseline_model)")
    parser.add_argument("--text_model_name")
    parser.add_argument("--text_max_length", type=int)
    parser.add_argument("--text_max_batch_size", type=int)
    parser.add_argument("--text_trainable", action=argparse.BooleanOptionalAction)

    # data scope
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--max_columns", type=int)
    parser.add_argument("--n_dbs", type=int, default=None, help="pretraining pilot cap -- see scripts/pretrain_electra.py")
    parser.add_argument("--n_tables", type=int, default=None)

    # pretraining
    parser.add_argument("--corrupt_frac", type=float)
    parser.add_argument("--pretrain_batch_size", type=int)
    parser.add_argument("--pretrain_epochs", type=int, default=5)
    parser.add_argument("--pretrain_lr", type=float)
    parser.add_argument("--discriminator_hidden_dim", type=int)

    # finetuning
    parser.add_argument("--query_model_name")
    parser.add_argument("--query_trainable", action=argparse.BooleanOptionalAction)
    parser.add_argument("--query_max_length", type=int)
    parser.add_argument("--exclude_special_tokens", action=argparse.BooleanOptionalAction)
    parser.add_argument("--scoring_mode", choices=["global", "row_match", "column_match", "col_deepset", "row_deepset", "mixture"])
    parser.add_argument("--finetune_batch_size", type=int)
    parser.add_argument("--finetune_epochs", type=int, default=15)
    parser.add_argument("--finetune_lr", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--n_hard_negatives", type=int, default=2,
        help="see scripts/finetune_query_table.py's --n_hard_negatives -- same "
             "meaning, applied identically across every encoder for a fair "
             "comparison (consistent hyperparameters across models).",
    )
    parser.add_argument(
        "--val_sample_size", type=int, default=None,
        help="subsample the val query split to this many examples for the "
             "PER-EPOCH early-stopping MAP/MRR check, instead of using every val "
             "query. evaluate_ranking_metrics ranks every val query against the "
             "FULL corpus -- with a real split (hundreds of thousands of val "
             "queries) that's an enormous score matrix recomputed every single "
             "epoch. A fixed random subsample (same subset every epoch, drawn "
             "once via --seed) gives a stable, far cheaper per-epoch signal; a "
             "few thousand queries is already a low-variance MAP estimate. Does "
             "NOT affect the final test-set evaluation, which always uses the "
             "full test split exactly once at the end. Omit (default) to use the "
             "full val split every epoch, as before.",
    )

    # shared optimizer/infra
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--grad_clip_norm", type=float)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint_dir", default="eval/report_runs")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_pretrain", action=argparse.BooleanOptionalAction, default=False,
        help="skip stage 1 (ELECTRA pretraining) entirely and finetune straight from a "
             "freshly-initialized encoder -- no pretrain data sampling, no pretrain "
             "checkpoint. Useful for fast pilot iteration on the finetuning path alone. "
             "NOTE: for the actual model-vs-baseline comparison you're building toward, "
             "the agreed methodology is ELECTRA pretraining + finetuning for every model "
             "-- treat runs made with this flag as finetune-only smoke tests, not "
             "results to report.",
    )
    parser.add_argument(
        "--text_cache_path", default=None,
        help="only meaningful for --encoder ours -- see "
             "scripts/pretrain_electra.py's --text_cache_path. Defaults to "
             "<checkpoint_dir>/<encoder>/text_cache.pt, loaded before "
             "pretraining if it already exists and saved after both stages.",
    )

    # model.yaml has no key collisions with anything else, so the
    # generic multi-file merge is fine for it.
    apply_yaml_defaults(parser, "configs/model.yaml")

    # pretrain.yaml and finetune.yaml both use identically-named keys
    # ("lr", "batch_size", "num_epochs") for what are, in THIS script,
    # deliberately separate --pretrain_*/--finetune_* flags (so both
    # stages' values can coexist in one run) -- apply_yaml_defaults'
    # automatic dest-matching can't disambiguate identically-named keys
    # from two different files (it would just let whichever file is
    # passed last silently win for both stages). Map each stage's yaml
    # value to its own flag by hand instead.
    from src.training.config import load_yaml_defaults

    pretrain_yaml = load_yaml_defaults("configs/pretrain.yaml")
    finetune_yaml = load_yaml_defaults("configs/finetune.yaml")
    stage_defaults = {}
    if "lr" in pretrain_yaml:
        stage_defaults["pretrain_lr"] = pretrain_yaml["lr"]
    if "batch_size" in pretrain_yaml:
        stage_defaults["pretrain_batch_size"] = pretrain_yaml["batch_size"]
    if "num_epochs" in pretrain_yaml:
        stage_defaults["pretrain_epochs"] = pretrain_yaml["num_epochs"]
    if "lr" in finetune_yaml:
        stage_defaults["finetune_lr"] = finetune_yaml["lr"]
    if "batch_size" in finetune_yaml:
        stage_defaults["finetune_batch_size"] = finetune_yaml["batch_size"]
    if "num_epochs" in finetune_yaml:
        stage_defaults["finetune_epochs"] = finetune_yaml["num_epochs"]
    parser.set_defaults(**stage_defaults)

    # remaining shared flags (weight_decay/warmup_ratio/grad_clip_norm/
    # corrupt_frac/discriminator_hidden_dim/max_rows/max_columns/device/
    # log_every) don't meaningfully differ between the two yaml files in
    # practice, so the generic merge (finetune.yaml winning on any
    # collision, applied last) is an acceptable simplification -- ONE
    # shared value across both stages, not per-stage.
    apply_yaml_defaults(parser, "configs/pretrain.yaml", "configs/finetune.yaml")

    # checkpoint_dir is a genuine exception -- both yaml files define it
    # as their OWN stage-specific path ("eval/report_runs/pretrain" /
    # ".../finetune"), which the generic merge above would have just
    # applied to THIS script's single --checkpoint_dir flag (this
    # script derives pretrain_dir/finetune_dir as subdirectories of it
    # itself, per --encoder). Pin it back to this script's own default
    # so a pretrain/finetune yaml edit can't silently redirect it.
    parser.set_defaults(checkpoint_dir="eval/report_runs")

    args = parser.parse_args()

    encoder_checkpoint_dir = os.path.join(args.checkpoint_dir, args.encoder)
    pretrain_dir = os.path.join(encoder_checkpoint_dir, "pretrain")
    finetune_dir = os.path.join(encoder_checkpoint_dir, "finetune")

    if args.text_cache_path is None:
        args.text_cache_path = os.path.join(encoder_checkpoint_dir, "text_cache.pt")

    rng = random.Random(args.seed)

    print(f"=== encoder: {args.encoder} ===")
    print(f"indexing tables from {args.tables_json} / {args.databases_root} ...")
    table_dataset = SynSQLTableDataset(
        tables_json=args.tables_json,
        databases_root=args.databases_root,
        max_rows=args.max_rows,
    )

    query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    print(f"loaded {len(query_dataset)} query -> table example(s)")

    print(f"loading split from {args.split_json} ...")
    resolved = query_dataset.resolve_split(args.split_json)
    train_indices, val_indices, test_indices = resolved["train"], resolved["val"], resolved["test"]
    print(f"split: {len(train_indices)} train / {len(val_indices)} val / {len(test_indices)} test")

    print(f"loading fixed corpus from {args.corpus_json} ...")
    corpus_tables = table_dataset.load_corpus(args.corpus_json)
    print(f"corpus: {len(corpus_tables)} table(s) -- used unsplit for every val/test ranking")

    model = build_table_model(args)

    # Text-embedding cache: only "ours" has one (CellEncoder/TextEmbedder
    # -- see src/encoding/cell_encoder.py). Baselines encode cells inline
    # inside their own forward() with no equivalent caching layer, so
    # model.save_text_cache/load_text_cache would raise AttributeError
    # for them -- guard on --encoder ours accordingly.
    if args.encoder == "ours" and os.path.exists(args.text_cache_path):
        print(f"loading cell/header text cache from {args.text_cache_path} ...")
        model.load_text_cache(args.text_cache_path)
        print(f"text cache warm-started with {model.cell_encoder.text_embedder.cache_size()} entries")

    if args.skip_pretrain:
        print(
            f"\n=== [{args.encoder}] stage 1/2: SKIPPED (--skip_pretrain) -- "
            f"finetuning from a freshly-initialized encoder ==="
        )
    else:
        # -----------------------------------------------------------
        # stage 1: ELECTRA pretraining
        # -----------------------------------------------------------
        db_ids = table_dataset.db_ids()
        if args.n_dbs is not None and args.n_dbs < len(db_ids):
            rng.shuffle(db_ids)
            db_ids = db_ids[: args.n_dbs]
            print(f"pretrain pilot: sampling {len(db_ids)} database(s)")

        table_keys = [(db_id, t) for db_id in db_ids for t in table_dataset.tables_in_db(db_id)]
        rng.shuffle(table_keys)
        if args.n_tables is not None and args.n_tables < len(table_keys):
            table_keys = table_keys[: args.n_tables]
            print(f"pretrain pilot: sampling {len(table_keys)} table(s)")

        pretrain_tables = [table_dataset.get_table(db_id, t) for db_id, t in table_keys]
        print(f"pretrain corpus: {len(pretrain_tables)} table(s)")

        n_val = max(1, int(len(pretrain_tables) * 0.1))
        pretrain_val_tables, pretrain_train_tables = pretrain_tables[:n_val], pretrain_tables[n_val:]

        pretrain_batches = make_batches(pretrain_train_tables, args.pretrain_batch_size, rng, max_columns=args.max_columns)
        pretrain_val_batches = make_batches(pretrain_val_tables, args.pretrain_batch_size, rng, max_columns=args.max_columns)
        print(f"pretrain: {len(pretrain_batches)} train batches/epoch, {len(pretrain_val_batches)} val batches")

        discriminator = DiscriminatorHead(embed_dim=args.embed_dim, hidden_dim=args.discriminator_hidden_dim)

        pretrainer = PretrainTrainer(
            model,
            discriminator,
            lr=args.pretrain_lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            grad_clip_norm=args.grad_clip_norm,
            corrupt_frac=args.corrupt_frac,
            checkpoint_dir=pretrain_dir,
            device=args.device,
            seed=args.seed,
        )

        # Resume from an existing checkpoint if this encoder's pretrain_dir
        # already has one -- e.g. a previous run of this exact command was
        # interrupted partway through pretraining. Without this,
        # restarting train_model.py always retrained from epoch 0 even
        # though PretrainTrainer.save_checkpoint was writing a checkpoint
        # every epoch the whole time -- those files just sat there unused.
        resume_ckpt = latest_checkpoint(pretrain_dir)
        if resume_ckpt is not None:
            print(f"found existing pretrain checkpoint, resuming from {resume_ckpt}")

        print(f"\n=== [{args.encoder}] stage 1/2: ELECTRA pretraining on {args.device} ===")
        pretrainer.fit(
            pretrain_batches,
            num_epochs=args.pretrain_epochs,
            steps_per_epoch=len(pretrain_batches),
            log_every=args.log_every,
            val_batches=pretrain_val_batches,
            resume_from=resume_ckpt,
        )

        if args.encoder == "ours":
            os.makedirs(os.path.dirname(args.text_cache_path) or ".", exist_ok=True)
            model.save_text_cache(args.text_cache_path)
            print(
                f"saved text cache ({model.cell_encoder.text_embedder.cache_size()} entries) "
                f"to {args.text_cache_path} after pretraining"
            )

        ckpt = latest_checkpoint(pretrain_dir)
        if ckpt is None:
            raise RuntimeError(f"pretraining produced no checkpoint in {pretrain_dir}")
        print(f"loading pretrained encoder from {ckpt}")
        load_pretrained_encoder(model, ckpt, device=args.device)

    # ---------------------------------------------------------------
    # stage 2: query-table finetuning (early stopping on val MAP)
    # ---------------------------------------------------------------
    def build_train_batches():
        """Called fresh once per epoch by FinetuneTrainer.fit (see its
        batch_fn docstring) -- see finetune_query_table.py's
        build_train_batches for why this can't just be a static list."""
        return list(
            resolve_train_batches(
                query_dataset,
                table_dataset,
                train_indices,
                args.finetune_batch_size,
                args.max_columns,
                rng,
                n_hard_negatives=args.n_hard_negatives,
            )
        )

    finetune_steps_per_epoch = len(build_train_batches())

    eval_val_indices = val_indices
    if args.val_sample_size is not None and args.val_sample_size < len(val_indices):
        # Fixed once (not resampled per epoch) -- the whole point is a
        # STABLE per-epoch signal so epoch-to-epoch MAP changes reflect
        # the model improving, not a different random subset of queries
        # each time. random.Random(args.seed) here is intentionally a
        # FRESH generator, not `rng` (which keeps advancing for batch
        # construction) -- this selection must be reproducible given
        # just --seed, independent of how many other random draws
        # happened before it.
        eval_val_indices = random.Random(args.seed).sample(val_indices, args.val_sample_size)
        print(
            f"finetune: subsampled val set for per-epoch checks: "
            f"{len(eval_val_indices)}/{len(val_indices)} val quer(ies) "
            f"(--val_sample_size {args.val_sample_size})"
        )

    val_examples = to_eval_examples(query_dataset, eval_val_indices)
    test_examples = to_eval_examples(query_dataset, test_indices)
    print(
        f"finetune: {finetune_steps_per_epoch} train batches/epoch "
        f"(n_hard_negatives={args.n_hard_negatives}), "
        f"{len(val_examples)} val example(s), {len(test_examples)} test example(s)"
    )

    query_encoder = QueryEncoder(
        model_name=args.query_model_name,
        output_dim=args.embed_dim,
        max_length=args.query_max_length,
        trainable=args.query_trainable,
        exclude_special_tokens=args.exclude_special_tokens,
    )

    finetuner = FinetuneTrainer(
        model,
        query_encoder,
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip_norm=args.grad_clip_norm,
        temperature=args.temperature,
        scoring_mode=args.scoring_mode,
        checkpoint_dir=finetune_dir,
        device=args.device,
        seed=args.seed,
    )

    print(f"\n=== [{args.encoder}] stage 2/2: finetuning on {args.device} (scoring_mode={args.scoring_mode}, patience={args.patience}) ===")
    best_val_map = finetuner.fit(
        build_train_batches,
        num_epochs=args.finetune_epochs,
        steps_per_epoch=finetune_steps_per_epoch,
        val_examples=val_examples,
        corpus_tables=corpus_tables,
        patience=args.patience,
        log_every=args.log_every,
    )

    print(f"\n[{args.encoder}] best validation MAP: {best_val_map:.4f}")

    best_ckpt_path = os.path.join(finetune_dir, "best_model.pt")
    test_map = None
    if os.path.exists(best_ckpt_path):
        finetuner.load_checkpoint(best_ckpt_path)
        test_map = finetuner.evaluate_map(test_examples, corpus_tables)
        print(f"[{args.encoder}] test MAP (best-val-MAP checkpoint): {test_map:.4f}")
    else:
        print(f"[{args.encoder}] no best checkpoint found (val MAP never improved) -- skipping test evaluation")

    # Persisted, structured record -- one per encoder, all in the SAME
    # shape, so scripts/run_all_models.sh can aggregate every encoder's
    # results.json into a single cross-model report table (best_model.pt/
    # train.log hold the same numbers but aren't convenient to compare
    # across models). See finetune_query_table.py's matching results.json
    # for the single-model equivalent.
    results = {
        "encoder": args.encoder,
        "best_val_map": best_val_map,
        "test_map": test_map,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_val_used_for_early_stopping": len(eval_val_indices),
        "n_test": len(test_indices),
        "corpus_size": len(corpus_tables),
        "seed": args.seed,
        "scoring_mode": args.scoring_mode,
        "embed_dim": args.embed_dim,
        "num_layers": args.num_layers,
        "n_hard_negatives": args.n_hard_negatives,
        "patience": args.patience,
        "skip_pretrain": args.skip_pretrain,
        "pretrain_epochs_configured": args.pretrain_epochs,
        "finetune_epochs_configured": args.finetune_epochs,
        "split_json": args.split_json,
        "corpus_json": args.corpus_json,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    results_path = os.path.join(encoder_checkpoint_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[{args.encoder}] wrote results to {results_path}")

    if args.encoder == "ours":
        os.makedirs(os.path.dirname(args.text_cache_path) or ".", exist_ok=True)
        model.save_text_cache(args.text_cache_path)
        print(
            f"[{args.encoder}] saved text cache "
            f"({model.cell_encoder.text_embedder.cache_size()} entries) "
            f"to {args.text_cache_path} after finetuning"
        )

    print(f"\n[{args.encoder}] training complete.")
