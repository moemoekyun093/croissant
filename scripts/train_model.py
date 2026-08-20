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


def _int_or_none(value: str) -> int | None:
    """argparse type= for flags that accept either an int or the literal
    string "None" to disable (e.g. --query_batch_size None)."""
    if value.lower() == "none":
        return None
    return int(value)

from scripts.pretrain_electra import make_batches
from scripts.finetune_query_table import cap_columns, count_batches, resolve_train_batches, to_eval_examples

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


def _safe_load_cache(load_fn, path: str, label: str) -> bool:
    """Wraps a *.load_*cache(path) call so a corrupted cache file can't
    crash the whole training launch. Root cause seen in practice: a
    training process killed mid torch.save() (e.g. SIGKILL from a
    relaunch/restart) leaves a truncated .pt file -- torch.load then
    raises a RuntimeError ("PytorchStreamReader failed locating file ...
    internal miniz error") deep inside torch's own unpickler, which
    previously propagated straight out of this script before training
    even started (a corrupted CACHE file was blocking an otherwise-fine
    run). On a load failure: renames the bad file aside to
    '<path>.corrupted' (so it's not silently lost/inspectable later, and
    so it stops being picked up as "exists" on the NEXT launch too) and
    continues with an empty cache -- exactly as if the file had never
    existed, just slower on the first pass while it rebuilds. Returns
    True if the load actually succeeded.
    """
    try:
        load_fn(path)
        return True
    except Exception as e:
        corrupted_path = path + ".corrupted"
        print(
            f"WARNING: failed to load {label} cache from {path!r} "
            f"({type(e).__name__}: {e}) -- treating as if it doesn't "
            f"exist and continuing with an empty cache. Moving the bad "
            f"file to {corrupted_path!r} so it isn't picked up again."
        )
        try:
            os.replace(path, corrupted_path)
        except OSError as move_err:
            print(f"  (also failed to move the corrupted file aside: {move_err})")
        return False


def build_table_model(args):
    if args.encoder == "ours":
        cell_encoder = CellEncoder(
            text_model_name=args.text_model_name,
            output_dim=args.embed_dim,
            text_max_length=args.text_max_length,
            text_trainable=args.text_trainable,
            text_max_batch_size=args.text_max_batch_size,
            header_mode=args.header_mode,
        )
        return TableEncoder(
            cell_encoder,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            nonlinearity=args.nonlinearity,
            channel_mix_hidden_dim=args.channel_mix_hidden_dim,
            num_heads=args.num_heads,
            table_microbatch_cell_budget=getattr(args, "table_microbatch_cell_budget", None),
            table_microbatch_max_tables=getattr(args, "table_microbatch_max_tables", None),
        )
    return build_baseline_model(
        args.encoder,
        embed_dim=args.embed_dim,
        model_name=args.model_name,
        num_layers=args.num_layers,
        tabbie_ffn_hidden_dim=args.tabbie_ffn_hidden_dim,
        strubert_ffn_hidden_dim=getattr(args, "strubert_ffn_hidden_dim", None),
        turl_attention_budget=getattr(args, "turl_attention_budget", None),
        table_microbatch_cell_budget=getattr(args, "table_microbatch_cell_budget", None),
        table_microbatch_max_tables=getattr(args, "table_microbatch_max_tables", None),
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
        "--scoring_embed_dim",
        type=int,
        default=None,
        help="retrieval width after table contextualization (default: embed_dim). "
             "For example, --embed_dim 768 --scoring_embed_dim 64 keeps a "
             "768-wide table network and learns a 768->64 projection before "
             "query-table scoring; the query tower also outputs 64 dimensions.",
    )
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
    parser.add_argument(
        "--header_mode", choices=["concat", "film"], default="concat",
        help="only used when --encoder ours: how each cell's column header is "
             "combined with its content. 'concat' (default) = raw concat of a "
             "cell-text half and a header half (original behavior). 'film' = "
             "the header produces a per-channel (gamma, beta) that modulates "
             "full-width content (FiLM), so content isn't sharing its budget "
             "with a replicated header constant.",
    )
    parser.add_argument(
        "--channel_mix_hidden_dim", type=int,
        help="ours only: hidden width of each pointwise ChannelMix MLP (default: 2 * embed_dim)",
    )
    parser.add_argument(
        "--table_microbatch_cell_budget",
        type=_int_or_none,
        default=None,
        help="all encoders: maximum padded cell slots B*N_max*M_max per "
             "shape-aware candidate-table microbatch; preserves full-pool "
             "InfoNCE and candidate order (default None disables)",
    )
    parser.add_argument(
        "--table_microbatch_max_tables",
        type=_int_or_none,
        default=None,
        help="all baseline encoders: maximum tables per encoder forward "
             "microbatch, useful when token sequence cost is not captured "
             "well by cell count (default None disables)",
    )
    parser.add_argument(
        "--tabbie_ffn_hidden_dim", type=int,
        help="TABBIE only: hidden width of each row/column Transformer FFN (default: 4 * native hidden size)",
    )
    parser.add_argument(
        "--strubert_ffn_hidden_dim",
        type=int,
        help="StruBERT only: hidden width of each vertical/horizontal "
             "Transformer FFN (default: 4 * native hidden size)",
    )
    parser.add_argument(
        "--turl_attention_budget",
        type=int,
        default=2_000_000,
        help="TURL only: dynamic microbatch budget B*S_max^2 for padded "
             "visibility attention (default: 2000000); outlier tables whose "
             "own S^2 exceeds the budget run alone",
    )
    parser.add_argument("--num_heads", type=int, default=8, help="ours cross-column attention heads (embed_dim must be divisible by it); default 8 matches the transformer baselines")
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
    parser.add_argument(
        "--patience", type=int, default=None,
        help="stop early after this many epochs without a val MAP "
             "improvement. Left as None by default so it can auto-scale "
             "with --train_sample_size -- see scripts/finetune_query_table.py's "
             "--patience for the full rationale: without chunking, defaults "
             "to 3 (same as before); with --train_sample_size chunking one "
             "full pass into many short epochs, defaults to ~20% of the "
             "resulting epoch count instead (floor 3). Pass an explicit "
             "value to override either default.",
    )
    parser.add_argument(
        "--query_batch_size", type=_int_or_none, default=2000,
        help="caps how many questions QueryEncoder encodes in ONE forward "
             "pass during MAP/MRR evaluation (per-epoch val AND the final "
             "test-set evaluation) -- see trainer.py's _corpus_scores' "
             "query_batch_size docstring. Left unbounded (None) this "
             "defaults to encoding EVERY query in the split in a single "
             "batched call, which is fine for a --val_sample_size-capped "
             "val set but WILL CUDA-OOM on an unsampled test split at real "
             "dataset scale (confirmed: a ~226k-question test split tried "
             "to allocate 20+ GiB in one scaled_dot_product_attention call "
             "here). Pass --query_batch_size None to disable chunking "
             "entirely if you're sure your query set is small enough.",
    )
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
    parser.add_argument(
        "--train_sample_size", type=int, default=None,
        help="see scripts/finetune_query_table.py's --train_sample_size -- "
             "chunk size (in queries) for one finetuning epoch. The full "
             "train split is shuffled once (fixed via --seed) and sliced "
             "into non-overlapping chunks of this size; each chunk becomes "
             "one epoch, and --finetune_epochs is auto-overridden to the "
             "resulting chunk count so the whole run sweeps every train "
             "query exactly once in total. Omit to keep the old behavior: "
             "one giant epoch over the entire train split, repeated for "
             "--finetune_epochs.",
    )
    parser.add_argument(
        "--val_corpus_sample_size", type=int, default=None,
        help="see scripts/finetune_query_table.py's --val_corpus_sample_size "
             "-- subsample the fixed corpus to this many tables for the "
             "PER-EPOCH early-stopping MAP/MRR check only (forced positives "
             "+ forced same-db hard negatives, see --val_n_hard_negatives, "
             "+ random fill). Final test-set MAP still scores the full "
             "corpus. Omit to score the full corpus every epoch, as before.",
    )
    parser.add_argument(
        "--val_n_hard_negatives", type=int, default=2,
        help="see scripts/finetune_query_table.py's --val_n_hard_negatives "
             "-- only used together with --val_corpus_sample_size. Per "
             "unique db_id among the (possibly subsampled) val queries, "
             "force-include up to this many other same-database tables "
             "into the per-epoch val corpus subsample, so the cheap check "
             "isn't dominated by easy, unrelated-db negatives.",
    )

    # shared optimizer/infra
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--grad_clip_norm", type=float)
    parser.add_argument("--device")
    parser.add_argument(
        "--profile", action="store_true",
        help="log a running average of query-encode/table-encode/scoring "
             "time per training step (see trainer.py's _record_profile), "
             "printed every --profile_every steps",
    )
    parser.add_argument("--profile_every", type=int, default=20)
    parser.add_argument(
        "--sync_cuda_errors",
        action="store_true",
        help="synchronize after every CUDA stage for crash attribution; "
        "slower and unnecessary for normal training",
    )
    parser.add_argument(
        "--score_table_chunk_size",
        type=_int_or_none,
        default=None,
        help="all encoders: score this many candidate tables at a time, "
             "then concatenate the complete score matrix before InfoNCE; "
             "reduces peak similarity-tensor memory without changing the loss",
    )
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
        "--skip_finetune_resume", action=argparse.BooleanOptionalAction, default=False,
        help="ignore an existing <checkpoint_dir>/<encoder>/finetune/best_model.pt "
             "and start finetuning from scratch (fresh model/optimizer state, "
             "epoch 0) instead of auto-resuming from it. Auto-resume is the "
             "default -- pass this to force a clean restart (e.g. after "
             "changing --scoring_mode or other hyperparameters that make the "
             "old checkpoint's optimizer state/epoch count meaningless).",
    )
    parser.add_argument(
        "--text_cache_path", default=None,
        help="only meaningful for --encoder ours -- see "
             "scripts/pretrain_electra.py's --text_cache_path. Defaults to "
             "<checkpoint_dir>/<encoder>/text_cache.pt, loaded before "
             "pretraining if it already exists and saved after both stages.",
    )
    parser.add_argument(
        "--table_cache_path", default=None,
        help="only meaningful for --encoder bert/tapas (the two fully-"
             "frozen baselines, see adapter.py's _FULLY_FROZEN_ENCODERS) "
             "-- persists BaselineCellwiseAdapter._table_cache (every "
             "table's raw cell embeddings, deterministic for a frozen "
             "encoder) to disk, so a table already encoded by ANY past "
             "run never needs its backbone forward pass re-run again. "
             "Defaults to <checkpoint_dir>/<encoder>/table_cache.pt, "
             "loaded before finetuning if it already exists and saved "
             "after finetuning. Delete it if --model_name or the cell "
             "serialization convention changes.",
    )
    parser.add_argument(
        "--frozen_cache_path", default=None,
        help="only meaningful for --encoder tabbie/strubert -- persists "
             "their frozen-BERT sub-step's output (tabbie: per cell/"
             "header string -> [CLS] vector; strubert: per row/column-"
             "sequence string -> (fine, coarse) pooled vectors) to disk. "
             "Unlike --table_cache_path, this does NOT cache the final "
             "table embedding -- tabbie/strubert have trainable layers "
             "(row/col transformer; vertical/horizontal attention + "
             "fusion) on top of the frozen BERT step, so their final "
             "output changes every training step and can't be cached the "
             "way bert/tapas's can (see adapter.py's cacheable "
             "docstring). Only the expensive frozen part is skipped on a "
             "hit; the trainable stack always runs fresh. Defaults to "
             "<checkpoint_dir>/<encoder>/frozen_cache.pt, loaded before "
             "finetuning if it already exists and saved after finetuning.",
    )
    parser.add_argument(
        "--query_cache_path", default=None,
        help="only meaningful when query_trainable=False (the default, "
             "see configs/finetune.yaml) -- persists QueryEncoder's frozen "
             "BERT output per query string to disk (see "
             "query_encoder.py's save_frozen_cache). SHARED across every "
             "--encoder (not nested under <checkpoint_dir>/<encoder>/ "
             "like the table caches), since the query tower and val/test "
             "question text don't depend on which table encoder you're "
             "running -- building this cache once while running 'bert' "
             "means 'tapas'/'tabbie'/etc.'s runs benefit immediately too. "
             "Defaults to <checkpoint_dir>/query_cache.pt, loaded before "
             "finetuning if it exists and saved after finetuning.",
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
    if args.scoring_embed_dim is None:
        args.scoring_embed_dim = args.embed_dim
    if args.scoring_embed_dim <= 0:
        parser.error("--scoring_embed_dim must be positive")

    encoder_checkpoint_dir = os.path.join(args.checkpoint_dir, args.encoder)
    pretrain_dir = os.path.join(encoder_checkpoint_dir, "pretrain")
    finetune_dir = os.path.join(encoder_checkpoint_dir, "finetune")

    if args.text_cache_path is None:
        args.text_cache_path = os.path.join(encoder_checkpoint_dir, "text_cache.pt")
    if args.table_cache_path is None:
        args.table_cache_path = os.path.join(encoder_checkpoint_dir, "table_cache.pt")
    if args.frozen_cache_path is None:
        args.frozen_cache_path = os.path.join(encoder_checkpoint_dir, "frozen_cache.pt")
    if args.query_cache_path is None:
        args.query_cache_path = os.path.join(args.checkpoint_dir, "query_cache.pt")  # SHARED across encoders, not per-encoder

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
        if _safe_load_cache(model.load_text_cache, args.text_cache_path, "text"):
            print(f"text cache warm-started with {model.cell_encoder.text_embedder.cache_size()} entries")

    # Table-embedding cache: only meaningful for the two fully-frozen
    # baselines (bert/tapas -- see adapter.py's _FULLY_FROZEN_ENCODERS /
    # BaselineCellwiseAdapter.cacheable's docstring for why ONLY these
    # two are safe to cache across whole runs, not tabbie/strubert/turl/
    # hytrel). A table's raw cell embeddings are a deterministic
    # function of frozen weights for these two, so a cache built by any
    # PAST run -- pretraining, a previous finetuning attempt, even a
    # different scoring_mode sweep, since table embeddings don't depend
    # on scoring_mode at all -- can be reused here directly.
    if args.encoder in ("bert", "tapas") and os.path.exists(args.table_cache_path):
        print(f"loading table-embedding cache from {args.table_cache_path} ...")
        if _safe_load_cache(model.load_table_cache, args.table_cache_path, "table"):
            print(f"table cache warm-started with {len(model._table_cache)} entries")

    # Frozen-substep cache: tabbie/strubert can't cache their FINAL table
    # embedding (see --frozen_cache_path's help text), but the frozen
    # BERT sub-step underneath their trainable layers is just as
    # deterministic as bert/tapas's whole encoder -- cache that part
    # only. model.baseline_encoder is the actual TabbieTableEncoder/
    # StruBertTableEncoder instance inside the adapter.
    if args.encoder in ("tabbie", "strubert") and os.path.exists(args.frozen_cache_path):
        print(f"loading frozen-substep cache from {args.frozen_cache_path} ...")
        if _safe_load_cache(model.baseline_encoder.load_frozen_cache, args.frozen_cache_path, "frozen-substep"):
            cache_attr = "_cell_cache" if args.encoder == "tabbie" else "_seq_cache"
            print(f"frozen cache warm-started with {len(getattr(model.baseline_encoder, cache_attr))} entries")

    # Resume from an existing checkpoint if this encoder's pretrain_dir
    # already has one -- e.g. a previous run of this exact command was
    # interrupted partway through pretraining. Checked BEFORE gathering
    # any pretrain table data: if the checkpoint already covers every
    # requested --pretrain_epochs, there's nothing left to train, and
    # re-gathering the full pretrain table corpus (a live SQL fetch per
    # table -- up to ~160k of them at real scale) just to build a
    # training loop that immediately no-ops would be pure waste. This is
    # what a prior version of this script actually did -- looked "stuck"
    # after a resume because it silently redid ~168k table fetches for
    # a stage that had nothing left to do.
    resume_ckpt = latest_checkpoint(pretrain_dir)
    pretrain_already_done = False
    if resume_ckpt is not None:
        import torch

        completed_epoch = torch.load(resume_ckpt, map_location="cpu").get("epoch")
        if completed_epoch is not None and completed_epoch + 1 >= args.pretrain_epochs:
            pretrain_already_done = True

    if args.skip_pretrain:
        print(
            f"\n=== [{args.encoder}] stage 1/2: SKIPPED (--skip_pretrain) -- "
            f"finetuning from a freshly-initialized encoder ==="
        )
    elif pretrain_already_done:
        print(
            f"\n=== [{args.encoder}] stage 1/2: already complete -- "
            f"{resume_ckpt} covers all {args.pretrain_epochs} requested "
            f"pretrain epoch(s), skipping straight to finetuning without "
            f"re-gathering the pretrain table corpus ==="
        )
        print(f"loading pretrained encoder from {resume_ckpt}")
        load_pretrained_encoder(model, resume_ckpt, device=args.device)
    else:
        # -----------------------------------------------------------
        # stage 1: ELECTRA pretraining
        # -----------------------------------------------------------
        if resume_ckpt is not None:
            print(f"found existing pretrain checkpoint, resuming from {resume_ckpt}")

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
        if args.encoder in ("bert", "tapas"):
            os.makedirs(os.path.dirname(args.table_cache_path) or ".", exist_ok=True)
            model.save_table_cache(args.table_cache_path)
            print(
                f"saved table cache ({len(model._table_cache)} entries) "
                f"to {args.table_cache_path} after pretraining"
            )
        if args.encoder in ("tabbie", "strubert"):
            os.makedirs(os.path.dirname(args.frozen_cache_path) or ".", exist_ok=True)
            model.baseline_encoder.save_frozen_cache(args.frozen_cache_path)
            cache_attr = "_cell_cache" if args.encoder == "tabbie" else "_seq_cache"
            print(
                f"saved frozen cache ({len(getattr(model.baseline_encoder, cache_attr))} entries) "
                f"to {args.frozen_cache_path} after pretraining"
            )

        ckpt = latest_checkpoint(pretrain_dir)
        if ckpt is None:
            raise RuntimeError(f"pretraining produced no checkpoint in {pretrain_dir}")
        print(f"loading pretrained encoder from {ckpt}")
        load_pretrained_encoder(model, ckpt, device=args.device)

    # ---------------------------------------------------------------
    # stage 2: query-table finetuning (early stopping on val MAP)
    # ---------------------------------------------------------------
    # Chunk the train split into bounded-size epochs -- see
    # scripts/finetune_query_table.py's --train_sample_size for the full
    # rationale (bounding wall-clock time per epoch while still sweeping
    # every train query exactly once across the whole run).
    if args.train_sample_size is not None and args.train_sample_size < len(train_indices):
        n_train_full = len(train_indices)
        shuffled_train = list(train_indices)
        random.Random(args.seed).shuffle(shuffled_train)
        train_chunks = [
            shuffled_train[i : i + args.train_sample_size]
            for i in range(0, n_train_full, args.train_sample_size)
        ]
        args.finetune_epochs = len(train_chunks)
        print(
            f"finetune: chunked {n_train_full} train quer(ies) into "
            f"{len(train_chunks)} epoch(s) of up to {args.train_sample_size} "
            f"quer(ies) each (exactly 1 pass over the full train set) -- "
            f"overriding --finetune_epochs to {args.finetune_epochs}"
        )
    else:
        train_chunks = [train_indices]

    if args.patience is None:
        if len(train_chunks) > 1:
            args.patience = max(3, round(len(train_chunks) * 0.2))
        else:
            args.patience = 3
        print(f"finetune: defaulting --patience to {args.patience} (epochs={len(train_chunks)})")

    # See scripts/finetune_query_table.py's build_train_batches for why
    # fit() calling this exactly once per epoch, strictly in increasing
    # epoch order, is what makes a plain advancing counter safe here.
    _epoch_counter = {"i": 0}

    def build_train_batches():
        """Called fresh once per epoch by FinetuneTrainer.fit (see its
        batch_fn docstring). Consumes the NEXT chunk in train_chunks --
        with --train_sample_size unset, train_chunks has just the one
        full-train-set entry, reused every call, same as before."""
        idx = min(_epoch_counter["i"], len(train_chunks) - 1)
        _epoch_counter["i"] += 1
        return list(
            resolve_train_batches(
                query_dataset,
                table_dataset,
                train_chunks[idx],
                args.finetune_batch_size,
                args.max_columns,
                rng,
                n_hard_negatives=args.n_hard_negatives,
            )
        )

    # Arithmetic count, NOT len(list(build_train_batches())) -- see
    # count_batches' docstring. Uses train_chunks[0]'s size as
    # representative (every chunk is that size except possibly the last
    # remainder) -- feeds only the LR warmup/decay schedule's total_steps
    # estimate, not per-epoch correctness.
    finetune_steps_per_epoch = count_batches(len(train_chunks[0]), args.finetune_batch_size)

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

    # Cheap per-epoch val corpus -- see scripts/finetune_query_table.py's
    # --val_corpus_sample_size/--val_n_hard_negatives for the full
    # rationale (forced positives + forced same-db hard negatives + random
    # fill; final test-set MAP below still scores the FULL corpus_tables,
    # unaffected by this).
    corpus_tables_for_val = corpus_tables
    if args.val_corpus_sample_size is not None and args.val_corpus_sample_size < len(corpus_tables):
        positive_ids = {
            f"{db_id}#sep#{t}" for _q, db_id, table_names in val_examples for t in table_names
        }
        forced = [t for t in corpus_tables if t.table_id in positive_ids]
        forced_ids = {t.table_id for t in forced}

        n_hard = 0
        if args.val_n_hard_negatives > 0:
            db_to_tables = {}
            for t in corpus_tables:
                db_id_of_t = t.table_id.split("#sep#", 1)[0]
                db_to_tables.setdefault(db_id_of_t, []).append(t)

            hard_neg_rng = random.Random(args.seed)
            val_db_ids = {db_id for _q, db_id, _table_names in val_examples}
            for db_id in val_db_ids:
                candidates = [t for t in db_to_tables.get(db_id, []) if t.table_id not in forced_ids]
                if not candidates:
                    continue
                hard_neg_rng.shuffle(candidates)
                for t in candidates[: args.val_n_hard_negatives]:
                    if t.table_id not in forced_ids:
                        forced.append(t)
                        forced_ids.add(t.table_id)
                        n_hard += 1

        remaining_pool = [t for t in corpus_tables if t.table_id not in forced_ids]
        n_fill = max(0, args.val_corpus_sample_size - len(forced))
        filler = random.Random(args.seed).sample(remaining_pool, min(n_fill, len(remaining_pool)))
        corpus_tables_for_val = forced + filler
        print(
            f"finetune: subsampled corpus for per-epoch val checks: "
            f"{len(corpus_tables_for_val)}/{len(corpus_tables)} table(s) "
            f"({len(forced) - n_hard} forced-included positive(s) + "
            f"{n_hard} forced same-db hard negative(s) + {len(filler)} random) "
            f"(--val_corpus_sample_size {args.val_corpus_sample_size}, "
            f"--val_n_hard_negatives {args.val_n_hard_negatives})"
        )

    print(
        f"finetune: {finetune_steps_per_epoch} train batches/epoch "
        f"(n_hard_negatives={args.n_hard_negatives}), "
        f"{len(val_examples)} val example(s) scored against "
        f"{len(corpus_tables_for_val)} corpus table(s) per epoch, "
        f"{len(test_examples)} test example(s)"
    )

    query_encoder = QueryEncoder(
        model_name=args.query_model_name,
        output_dim=args.scoring_embed_dim,
        max_length=args.query_max_length,
        trainable=args.query_trainable,
        exclude_special_tokens=args.exclude_special_tokens,
    )

    # Query cache: only meaningful when frozen (query_trainable=False,
    # the default) -- see --query_cache_path's help text. Shared across
    # every --encoder run, so this can warm-start from a cache built
    # while running a totally different encoder.
    if not args.query_trainable and os.path.exists(args.query_cache_path):
        print(f"loading query cache from {args.query_cache_path} ...")
        if _safe_load_cache(query_encoder.load_frozen_cache, args.query_cache_path, "query"):
            print(f"query cache warm-started with {len(query_encoder._encoder_cache)} entries")

    def save_all_caches() -> None:
        """Called every time val MAP improves (see fit()'s on_checkpoint)
        -- saves whichever cache(s) this --encoder actually has, right
        alongside best_model.pt, so a crash later in this SAME run
        doesn't lose everything accumulated since the last improvement.
        Mirrors the end-of-script cache-saving blocks below exactly, just
        triggered more often and earlier."""
        if args.encoder == "ours":
            os.makedirs(os.path.dirname(args.text_cache_path) or ".", exist_ok=True)
            model.save_text_cache(args.text_cache_path)
        if args.encoder in ("bert", "tapas"):
            os.makedirs(os.path.dirname(args.table_cache_path) or ".", exist_ok=True)
            model.save_table_cache(args.table_cache_path)
        if args.encoder in ("tabbie", "strubert"):
            os.makedirs(os.path.dirname(args.frozen_cache_path) or ".", exist_ok=True)
            model.baseline_encoder.save_frozen_cache(args.frozen_cache_path)
        if not args.query_trainable:
            os.makedirs(os.path.dirname(args.query_cache_path) or ".", exist_ok=True)
            query_encoder.save_frozen_cache(args.query_cache_path)

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
        profile=args.profile,
        profile_every=args.profile_every,
        score_table_chunk_size=args.score_table_chunk_size,
        sync_cuda_errors=args.sync_cuda_errors,
    )

    # Same "auto-resume if a checkpoint already exists" pattern as stage
    # 1's pretrain_dir handling above -- FinetuneTrainer.fit() already
    # accepted a resume_from argument (model/query_encoder/scorer/
    # optimizer state + global_step), this script just never actually
    # looked for one. Unlike pretrain (one file per epoch, needs
    # latest_checkpoint's max-epoch search), finetune only ever writes a
    # single best_model.pt (see save_checkpoint's docstring -- it
    # overwrites, doesn't accumulate), so finding it is just an
    # os.path.exists check. This is what makes "restart tabbie, it
    # already finished an epoch" actually resume from that epoch's
    # trained weights + optimizer state instead of re-initializing the
    # trainable row/col-transformer-on-top from scratch every relaunch.
    finetune_resume_ckpt = os.path.join(finetune_dir, "best_model.pt")
    if not os.path.exists(finetune_resume_ckpt):
        finetune_resume_ckpt = None
    elif args.skip_finetune_resume:
        print(f"found {finetune_resume_ckpt} but --skip_finetune_resume was passed -- starting finetuning fresh")
        finetune_resume_ckpt = None
    else:
        # Validate the file loads BEFORE handing it to finetuner.fit()
        # (which calls load_checkpoint deep inside its own resume logic)
        # -- same corrupted-checkpoint concern as _safe_load_cache above
        # (e.g. a process killed mid torch.save() leaves a truncated
        # file). A failure here just means "start finetuning fresh",
        # same as if the file never existed, instead of crashing the
        # whole launch before any training happens.
        import torch as _torch
        try:
            _torch.load(finetune_resume_ckpt, map_location="cpu")
            print(f"found existing finetune checkpoint, resuming from {finetune_resume_ckpt}")
        except Exception as e:
            corrupted_path = finetune_resume_ckpt + ".corrupted"
            print(
                f"WARNING: failed to load finetune checkpoint from "
                f"{finetune_resume_ckpt!r} ({type(e).__name__}: {e}) -- "
                f"starting finetuning fresh instead of resuming. Moving "
                f"the bad file to {corrupted_path!r}."
            )
            try:
                os.replace(finetune_resume_ckpt, corrupted_path)
            except OSError as move_err:
                print(f"  (also failed to move the corrupted file aside: {move_err})")
            finetune_resume_ckpt = None

    print(f"\n=== [{args.encoder}] stage 2/2: finetuning on {args.device} (scoring_mode={args.scoring_mode}, patience={args.patience}) ===")
    best_val_map = finetuner.fit(
        build_train_batches,
        num_epochs=args.finetune_epochs,
        steps_per_epoch=finetune_steps_per_epoch,
        val_examples=val_examples,
        corpus_tables=corpus_tables_for_val,
        patience=args.patience,
        log_every=args.log_every,
        val_query_batch_size=args.query_batch_size,
        on_checkpoint=save_all_caches,
        resume_from=finetune_resume_ckpt,
    )

    print(f"\n[{args.encoder}] best validation MAP: {best_val_map:.4f}")

    best_ckpt_path = os.path.join(finetune_dir, "best_model.pt")
    test_map = None
    if os.path.exists(best_ckpt_path):
        finetuner.load_checkpoint(best_ckpt_path)
        # test_examples is the FULL, unsampled test split (unlike
        # val_examples, which --val_sample_size may have capped) -- at
        # real SynSQL-2.5M scale this is ~226k questions, so
        # query_batch_size here is NOT optional the way it might appear
        # to be for val (confirmed: omitting it CUDA-OOM'd trying to
        # encode the whole test split in one QueryEncoder call).
        test_map = finetuner.evaluate_map(test_examples, corpus_tables, query_batch_size=args.query_batch_size)
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
        "scoring_embed_dim": args.scoring_embed_dim,
        "num_layers": args.num_layers,
        "table_microbatch_cell_budget": args.table_microbatch_cell_budget,
        "table_microbatch_max_tables": args.table_microbatch_max_tables,
        "score_table_chunk_size": args.score_table_chunk_size,
        "strubert_ffn_hidden_dim": (
            args.strubert_ffn_hidden_dim if args.encoder == "strubert" else None
        ),
        "turl_attention_budget": args.turl_attention_budget if args.encoder == "turl" else None,
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
    if args.encoder in ("bert", "tapas"):
        os.makedirs(os.path.dirname(args.table_cache_path) or ".", exist_ok=True)
        model.save_table_cache(args.table_cache_path)
        print(
            f"[{args.encoder}] saved table cache "
            f"({len(model._table_cache)} entries) "
            f"to {args.table_cache_path} after finetuning"
        )
    if args.encoder in ("tabbie", "strubert"):
        os.makedirs(os.path.dirname(args.frozen_cache_path) or ".", exist_ok=True)
        model.baseline_encoder.save_frozen_cache(args.frozen_cache_path)
        cache_attr = "_cell_cache" if args.encoder == "tabbie" else "_seq_cache"
        print(
            f"[{args.encoder}] saved frozen cache "
            f"({len(getattr(model.baseline_encoder, cache_attr))} entries) "
            f"to {args.frozen_cache_path} after finetuning"
        )
    if not args.query_trainable:
        os.makedirs(os.path.dirname(args.query_cache_path) or ".", exist_ok=True)
        query_encoder.save_frozen_cache(args.query_cache_path)
        print(
            f"[{args.encoder}] saved query cache "
            f"({len(query_encoder._encoder_cache)} entries) "
            f"to {args.query_cache_path} after finetuning -- shared, "
            f"benefits every other --encoder's runs too"
        )

    print(f"\n[{args.encoder}] training complete.")
