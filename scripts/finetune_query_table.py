"""
Finetuning: real query -> positive-table contrastive training, starting
from an ELECTRA-pretrained encoder checkpoint.

Defaults for every flag below come from configs/model.yaml and
configs/finetune.yaml (via src/training/config.py::apply_yaml_defaults)
-- running with no flags at all uses whatever's currently in those
files. Pass a --flag explicitly to override just that one value for a
single run.

Split/corpus: loads the persisted query split and fixed table corpus
built by scripts/build_query_splits.py (--split_json/--corpus_json) --
run that script once first. Training uses the TRAIN query examples
only; validation each epoch computes Mean Average Precision (MAP) of
the VAL query examples ranked against the FULL corpus (never a subset),
per-instruction that only queries are split, never the corpus. Model
selection is early stopping on best validation MAP (--patience epochs
without improvement) -- nothing else is used to decide which checkpoint
is "best". Test-set MAP is reported once at the end, using the
best-validation-MAP checkpoint, not the last epoch trained.

Usage:
    python -m scripts.build_query_splits \\
        --databases_root /path/to/synsql/databases \\
        --questions_json /path/to/synsql/questions_with_tables.json \\
        --tables_json /path/to/synsql/tables.json

    python -m scripts.finetune_query_table \\
        --databases_root /path/to/synsql/databases \\
        --split_json configs/splits/query_split.json \\
        --corpus_json configs/splits/corpus.json \\
        --pretrained_checkpoint eval/report_runs/pretrain/checkpoint_epoch14.pt

Loads the pretrained TableEncoder via load_pretrained_encoder(), which
discards any discriminator-head weights from that checkpoint -- the
discriminator was only ever needed for the pretraining task.
"""

import argparse
import datetime
import json
import os
import random

from src.data.synsql_dataset import SynSQLQueryDataset, SynSQLTableDataset
from src.data.table import Table
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import TableEncoder, load_pretrained_encoder
from src.training.config import apply_yaml_defaults
from src.training.query_encoder import QueryEncoder
from src.training.trainer import FinetuneTrainer


def cap_columns(table: Table, max_columns: int) -> Table:
    """Same outlier-wide-table truncation as the pretraining path's
    make_batches -- a positive table with many more columns than usual
    would otherwise single-handedly set the padding cost for whatever
    batch it lands in."""
    if len(table.columns) <= max_columns:
        return table
    return Table(table_id=table.table_id, table_name=table.table_name, columns=table.columns[:max_columns])


def sample_hard_negatives(
    table_dataset: SynSQLTableDataset,
    db_id: str,
    exclude_table_names: set,
    n: int,
    rng: random.Random,
    max_columns: int,
) -> list[Table]:
    """n tables from the SAME database as a query's positive, excluding
    the query's own gold table(s) -- harder negatives than a random
    other query's positive table (which usually comes from a totally
    different, unrelated database), since same-db tables share schema
    conventions and domain vocabulary and so are much easier for the
    model to confuse.

    Note: these are appended to a batch with no query of their own, so
    they're never anyone's positive -- but it's possible (rare, and no
    worse than the existing in-batch-negative risk) that a same-db table
    picked here happens to be the gold answer for some OTHER query's
    db_id-sharing question elsewhere in the batch; that's an inherent,
    accepted false-negative risk of in-batch/hard-negative training, not
    something new introduced here.
    """
    candidates = [t for t in table_dataset.tables_in_db(db_id) if t not in exclude_table_names]
    if not candidates:
        return []
    rng.shuffle(candidates)
    # Over-sample the shuffled candidate list before materializing --
    # some same-db tables are genuinely empty (0 rows/columns, see
    # SynSQLTableDataset._drop_empty_tables' docstring) and get skipped
    # below, so taking exactly the first n names up front could leave us
    # short of n negatives. get_table() is cheap for anything already
    # cached (e.g. corpus tables), so this costs little in practice.
    chosen: list[Table] = []
    for name in candidates:
        if len(chosen) >= n:
            break
        table = table_dataset.get_table(db_id, name)
        if table.num_rows == 0 or table.num_columns == 0:
            continue
        chosen.append(cap_columns(table, max_columns))
    return chosen


def resolve_train_batches(
    query_dataset: SynSQLQueryDataset,
    table_dataset: SynSQLTableDataset,
    indices: list[int],
    batch_size: int,
    max_columns: int,
    rng: random.Random,
    n_hard_negatives: int = 0,
):
    """Yields (pairs, hard_negatives, gold_table_ids) batches from the given TRAIN
    indices only.

    pairs: exactly one (question, positive_table) per query, randomly
        chosen when a query has more than one valid positive (see
        FinetuneTrainer._score_batch's docstring for why exactly one is
        retained: it guarantees a positive candidate for every query.)
    hard_negatives: n_hard_negatives extra tables per query in this
        batch, sampled from that query's OWN database but excluding its
        gold table(s) -- see sample_hard_negatives. Empty list when
        n_hard_negatives=0 (plain in-batch negatives only).
    gold_table_ids: complete set of valid, database-namespaced table IDs
        for each query, aligned with ``pairs``.  This allows the loss to
        recognize another query's candidate as an additional positive.

    Call this fresh each epoch (see FinetuneTrainer.fit's batch_fn
    docstring) -- it uses `rng` to shuffle query order, pick a positive
    among multiple valid ones, AND pick hard negatives, so a repeat call
    with the same rng state produces a genuinely different batch set,
    not just a reordering of the same one.
    """
    order = list(indices)
    rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        idx_batch = order[i : i + batch_size]
        pairs = []
        hard_negatives = []
        gold_table_ids = []
        for idx in idx_batch:
            question, tables = query_dataset[idx]
            non_empty = [t for t in tables if t.num_rows > 0 and t.num_columns > 0]
            if not non_empty:
                # This query's only positive table(s) are genuinely empty
                # (0 rows/columns) -- see SynSQLTableDataset's
                # _drop_empty_tables docstring. Nothing valid to train on
                # for this query; skip it rather than handing an encoder
                # a table it can't encode (some baselines raise on this).
                continue
            table = cap_columns(rng.choice(non_empty), max_columns)
            pairs.append((question, table))
            ex = query_dataset.examples[idx]
            gold_table_ids.append({f"{ex.db_id}#sep#{name}" for name in ex.table_names})

            if n_hard_negatives > 0:
                hard_negatives.extend(
                    sample_hard_negatives(
                        table_dataset,
                        ex.db_id,
                        set(ex.table_names),
                        n_hard_negatives,
                        rng,
                        max_columns,
                    )
                )
        if len(pairs) >= 2:
            yield pairs, hard_negatives, gold_table_ids


def count_batches(n_examples: int, batch_size: int) -> int:
    """
    How many batches resolve_train_batches will yield for n_examples,
    computed arithmetically -- matches its exact chunking/drop-tail rule
    (full batches of batch_size, final partial batch dropped only if it
    has fewer than 2 examples) WITHOUT actually calling
    resolve_train_batches.

    This matters a lot at real scale: resolve_train_batches doesn't just
    count, it fully MATERIALIZES every batch, including sampling and
    live-SQL-fetching n_hard_negatives extra tables per query. Calling
    it once just to do `len(list(...))` for a steps-per-epoch print
    statement -- which is what this codebase used to do -- means doing
    that full materialization (millions of extra SQLite reads at
    SynSQL-2.5M's real scale) BEFORE training even starts, for a single
    integer. Always use this instead when you just need the count.
    """
    full_batches, remainder = divmod(n_examples, batch_size)
    return full_batches + (1 if remainder >= 2 else 0)


def to_eval_examples(query_dataset: SynSQLQueryDataset, indices: list[int]) -> list[tuple[str, str, list[str]]]:
    """(question, db_id, table_names) tuples for evaluate_map -- see
    FinetuneTrainer.evaluate_map's docstring for why this stays a plain
    tuple rather than SynSQLQueryDataset's own QueryTableExample."""
    examples = []
    for idx in indices:
        ex = query_dataset.examples[idx]
        examples.append((ex.question, ex.db_id, ex.table_names))
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", default=None)
    parser.add_argument("--databases_root", required=True)
    parser.add_argument("--questions_json", required=True)
    parser.add_argument("--split_json", default="configs/splits/query_split.json")
    parser.add_argument("--corpus_json", default="configs/splits/corpus.json")
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--max_columns", type=int)
    parser.add_argument("--embed_dim", type=int, help="must match the pretrained checkpoint")
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--text_model_name")
    parser.add_argument("--text_max_length", type=int)
    parser.add_argument("--text_max_batch_size", type=int)
    parser.add_argument(
        "--text_trainable", action=argparse.BooleanOptionalAction,
        help="unfreeze CellEncoder's BERT -- off by default, see training_config.md",
    )
    parser.add_argument("--nonlinearity", choices=["sigmoid", "tanh", "relu"])
    parser.add_argument(
        "--channel_mix_hidden_dim", type=int,
        help="defaults to embed_dim * 2 when omitted",
    )
    parser.add_argument(
        "--num_heads", type=int, default=8,
        help="ours cross-column attention heads (embed_dim must be divisible by it); "
             "default 8 matches the transformer baselines",
    )
    parser.add_argument("--query_model_name")
    parser.add_argument(
        "--query_trainable", action=argparse.BooleanOptionalAction,
        help="train the query tower's BERT -- on by default, unlike CellEncoder's frozen BERT",
    )
    parser.add_argument("--query_max_length", type=int)
    parser.add_argument("--exclude_special_tokens", action=argparse.BooleanOptionalAction)
    parser.add_argument("--scoring_mode", choices=["global", "row_match", "column_match", "col_deepset", "row_deepset", "mixture"])
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument(
        "--patience", type=int, default=None,
        help="stop early after this many epochs without a val MAP "
             "improvement. Left as None by default so it can auto-scale "
             "with --train_sample_size: without chunking (one epoch = "
             "the whole train set), defaults to 3, same as before. WITH "
             "--train_sample_size chunking a full pass into many short "
             "epochs, 3 epochs of patience is only a few percent of one "
             "pass -- see the auto-scaling logic below, which sets it to "
             "~20% of the total (chunked) epoch count instead, floor 3. "
             "Pass an explicit value to override either default.",
    )
    parser.add_argument(
        "--n_hard_negatives", type=int, default=2,
        help="per query in a training batch, sample this many extra tables from "
             "that query's OWN database (excluding its gold table(s)) as hard "
             "negatives, on top of the ordinary in-batch negatives from other "
             "queries' positive tables. Same-database tables share schema/domain "
             "vocabulary, so they're a harder, more informative negative than a "
             "random other query's positive table. Set to 0 to disable.",
    )
    parser.add_argument(
        "--train_sample_size", type=int, default=None,
        help="chunk size (in queries) for one training epoch, used to "
             "bound wall-clock time per epoch. The FULL train split is "
             "shuffled once (fixed via --seed) and sliced into "
             "non-overlapping chunks of this size; each chunk becomes one "
             "epoch, and --num_epochs is AUTO-OVERRIDDEN to the resulting "
             "number of chunks so the whole run sweeps every train query "
             "EXACTLY ONCE in total -- not the same subset repeated every "
             "epoch. Does not touch val/test queries. Omit to keep the "
             "old behavior: one giant epoch over the entire train split, "
             "using whatever --num_epochs says.",
    )
    parser.add_argument(
        "--val_sample_size", type=int, default=None,
        help="subsample the val query split to this many examples for the "
             "PER-EPOCH early-stopping MAP/MRR check -- see "
             "scripts/train_model.py's --val_sample_size for the full "
             "rationale (a real split's val set can be hundreds of thousands "
             "of queries; ranking all of them against the full corpus every "
             "epoch is enormously expensive for a per-epoch signal). Fixed "
             "subset (drawn once via --seed), not resampled per epoch. Does "
             "NOT affect final test-set evaluation. Omit to use the full val "
             "split every epoch, as before.",
    )
    parser.add_argument(
        "--val_corpus_sample_size", type=int, default=None,
        help="subsample the FIXED corpus to this many tables for the "
             "PER-EPOCH early-stopping MAP/MRR check ONLY -- scoring the "
             "full corpus (often 100k+ tables) every epoch is itself "
             "expensive, on top of whatever --val_sample_size already "
             "trims on the query side; this trims the table side for that "
             "same frequent check, which is what actually makes "
             "validating every short (e.g. --train_sample_size-bounded) "
             "epoch affordable. Every table that's a valid positive for "
             "one of the (possibly --val_sample_size-subsampled) val "
             "queries is force-included first -- a naive uniform sample "
             "would drop most positives entirely for a corpus this size "
             "and silently tank MAP to ~0 for reasons that have nothing "
             "to do with the model -- then filled up to this size with a "
             "random sample of the rest (fixed once via --seed). The "
             "FINAL test-set MAP after training still scores the FULL "
             "corpus, unaffected by this flag. Omit to score the full "
             "corpus every epoch, as before.",
    )
    parser.add_argument(
        "--val_n_hard_negatives", type=int, default=2,
        help="only used together with --val_corpus_sample_size: per "
             "unique db_id among the (possibly subsampled) val queries, "
             "force-include up to this many OTHER same-database tables "
             "(excluding gold positives) into the per-epoch val corpus "
             "subsample, same idea as --n_hard_negatives for training "
             "batches. Without this, a small random corpus subsample is "
             "mostly easy negatives from unrelated databases, and MAP "
             "looks artificially high -- a model can rank the one "
             "obviously-relevant table above a pile of random ones "
             "without actually distinguishing it from its true, harder, "
             "same-schema neighbors. Sampled once per db_id (fixed via "
             "--seed), not per query, so queries sharing a db_id share "
             "the same forced hard negatives. Set to 0 to disable (forced "
             "positives + random fill only, previous behavior). No "
             "effect when --val_corpus_sample_size is unset (full corpus "
             "already contains every same-db table).",
    )
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--grad_clip_norm", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text_cache_path", default=None,
        help="path to CellEncoder's cell/header BERT-embedding cache (see "
             "scripts/pretrain_electra.py's --text_cache_path). Point this at "
             "the SAME file pretraining saved to (e.g. "
             "eval/report_runs/pretrain/text_cache.pt) so cell/header strings "
             "already seen during pretraining don't need BERT re-run here. "
             "Defaults to <checkpoint_dir>/text_cache.pt (this run's OWN "
             "checkpoint_dir, i.e. loads nothing new unless you point it "
             "explicitly at pretraining's).",
    )

    apply_yaml_defaults(parser, "configs/model.yaml", "configs/finetune.yaml")
    args = parser.parse_args()

    if args.text_cache_path is None:
        args.text_cache_path = os.path.join(args.checkpoint_dir, "text_cache.pt")

    rng = random.Random(args.seed)

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

    if args.train_sample_size is not None and args.train_sample_size < len(train_indices):
        # Shuffle the FULL train set once (fixed via --seed, independent
        # of `rng` which stays reserved for per-epoch in-chunk batch
        # order / positive-table choice / hard-negative resampling) and
        # slice it into non-overlapping chunks of --train_sample_size
        # queries each. Each chunk becomes exactly one epoch below, so
        # training for num_epochs = number of chunks sweeps every train
        # query EXACTLY ONCE in total -- not the same subset repeated
        # every epoch (that would never reach the rest of the train set)
        # and not one giant epoch over everything (that's what made a
        # single epoch take hours -- see the 41-min/2900-step log this
        # was sized from).
        n_train_full = len(train_indices)
        shuffled_train = list(train_indices)
        random.Random(args.seed).shuffle(shuffled_train)
        train_chunks = [
            shuffled_train[i : i + args.train_sample_size]
            for i in range(0, n_train_full, args.train_sample_size)
        ]
        args.num_epochs = len(train_chunks)
        print(
            f"chunked {n_train_full} train quer(ies) into {len(train_chunks)} "
            f"epoch(s) of up to {args.train_sample_size} quer(ies) each "
            f"(exactly 1 pass over the full train set) -- overriding "
            f"--num_epochs to {args.num_epochs}"
        )
    else:
        # Previous behavior: one epoch = the entire train split, repeated
        # (reshuffled/resampled fresh each time) for whatever --num_epochs
        # says.
        train_chunks = [train_indices]

    if args.patience is None:
        if len(train_chunks) > 1:
            # See --patience's help text: 3 epochs of patience made sense
            # when one epoch was a full pass over the train set, but here
            # one epoch is only 1/len(train_chunks) of a pass, so 3 would
            # stop the run after a tiny fraction of the data and on a
            # noisier signal (cheap, subsampled val corpus/queries).
            # Scale to ~20% of the chunked epoch count instead, floor 3.
            args.patience = max(3, round(len(train_chunks) * 0.2))
        else:
            args.patience = 3
        print(f"defaulting --patience to {args.patience} (epochs={len(train_chunks)})")

    # fit() calls batch_fn() exactly once per epoch, strictly in
    # increasing epoch order (see FinetuneTrainer.fit) -- so a plain
    # counter that advances on every call correctly hands out
    # train_chunks[0] on epoch 0, train_chunks[1] on epoch 1, etc. Not
    # thread-safe, but fit()'s loop is single-threaded/sequential, so
    # that's not a concern here. (Doesn't correctly resume mid-run via
    # trainer.fit's resume_from, which this script doesn't currently
    # expose as a flag anyway -- start_epoch would be nonzero but this
    # counter always starts at 0; flagging in case resume support is
    # added here later.)
    _epoch_counter = {"i": 0}

    def build_train_batches():
        """Called fresh once per epoch by FinetuneTrainer.fit (see its
        batch_fn docstring). Consumes the NEXT chunk in train_chunks (see
        above) -- with --train_sample_size unset, train_chunks has just
        the one full-train-set entry, reused every call, same as before.
        Still uses `rng` for in-chunk batch order / positive-table choice
        / hard-negative sampling, so each epoch's batches differ even
        when train_chunks has only one entry."""
        idx = min(_epoch_counter["i"], len(train_chunks) - 1)
        _epoch_counter["i"] += 1
        return list(
            resolve_train_batches(
                query_dataset,
                table_dataset,
                train_chunks[idx],
                args.batch_size,
                args.max_columns,
                rng,
                n_hard_negatives=args.n_hard_negatives,
            )
        )

    # Arithmetic count, NOT len(list(build_train_batches())) -- see
    # count_batches' docstring. At real dataset scale (millions of train
    # examples), materializing every batch just to count them means
    # doing millions of extra live SQL hard-negative fetches before
    # training even starts. Uses train_chunks[0]'s size as representative
    # -- every chunk is that size except possibly the last (remainder),
    # so this is exact for all but the final epoch and only feeds the LR
    # warmup/decay schedule's total_steps estimate, not per-epoch
    # correctness.
    steps_per_epoch = count_batches(len(train_chunks[0]), args.batch_size)

    eval_val_indices = val_indices
    if args.val_sample_size is not None and args.val_sample_size < len(val_indices):
        # Fixed once, not resampled per epoch -- see --val_sample_size's
        # help text. Independent random.Random(args.seed), not `rng`
        # (which keeps advancing for batch construction), so this
        # selection is reproducible from --seed alone.
        eval_val_indices = random.Random(args.seed).sample(val_indices, args.val_sample_size)
        print(
            f"subsampled val set for per-epoch checks: "
            f"{len(eval_val_indices)}/{len(val_indices)} val quer(ies) "
            f"(--val_sample_size {args.val_sample_size})"
        )

    val_examples = to_eval_examples(query_dataset, eval_val_indices)
    test_examples = to_eval_examples(query_dataset, test_indices)

    print(f"loading fixed corpus from {args.corpus_json} ...")
    corpus_tables = table_dataset.load_corpus(args.corpus_json)
    print(f"corpus: {len(corpus_tables)} table(s) -- full corpus, used unsplit for final test ranking")

    corpus_tables_for_val = corpus_tables
    if args.val_corpus_sample_size is not None and args.val_corpus_sample_size < len(corpus_tables):
        # Force-include every table that's a valid positive for one of
        # the val_examples above -- see --val_corpus_sample_size's help
        # text for why a naive uniform sample would wreck MAP.
        positive_ids = {
            f"{db_id}#sep#{t}" for _q, db_id, table_names in val_examples for t in table_names
        }
        forced = [t for t in corpus_tables if t.table_id in positive_ids]
        forced_ids = {t.table_id for t in forced}

        # Also force-include same-database hard negatives -- see
        # --val_n_hard_negatives' help text: without these, a small
        # random subsample is nearly all easy, unrelated-db negatives,
        # which inflates val MAP without actually testing whether the
        # model can tell a table apart from its true, harder, same-schema
        # siblings. Grouped by db_id (not per query) so queries sharing a
        # db_id share the same forced negatives instead of each
        # separately inflating the forced set.
        n_hard = 0
        if args.val_n_hard_negatives > 0:
            db_to_tables: dict[str, list[Table]] = {}
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
                chosen = candidates[: args.val_n_hard_negatives]
                for t in chosen:
                    if t.table_id not in forced_ids:
                        forced.append(t)
                        forced_ids.add(t.table_id)
                        n_hard += 1

        # Filled up to val_corpus_sample_size with a random sample of
        # whatever's left, fixed once via --seed (independent of `rng`).
        remaining_pool = [t for t in corpus_tables if t.table_id not in forced_ids]
        n_fill = max(0, args.val_corpus_sample_size - len(forced))
        filler = random.Random(args.seed).sample(remaining_pool, min(n_fill, len(remaining_pool)))
        corpus_tables_for_val = forced + filler
        print(
            f"subsampled corpus for per-epoch val checks: "
            f"{len(corpus_tables_for_val)}/{len(corpus_tables)} table(s) "
            f"({len(forced) - n_hard} forced-included positive(s) + "
            f"{n_hard} forced same-db hard negative(s) + {len(filler)} random) "
            f"(--val_corpus_sample_size {args.val_corpus_sample_size}, "
            f"--val_n_hard_negatives {args.val_n_hard_negatives})"
        )

    print(
        f"{steps_per_epoch} train batches/epoch (n_hard_negatives={args.n_hard_negatives}), "
        f"{len(val_examples)} val example(s) scored against {len(corpus_tables_for_val)} corpus table(s) "
        f"per epoch, {len(test_examples)} test example(s)"
    )

    cell_encoder = CellEncoder(
        text_model_name=args.text_model_name,
        output_dim=args.embed_dim,
        text_max_length=args.text_max_length,
        text_trainable=args.text_trainable,
        text_max_batch_size=args.text_max_batch_size,
    )
    model = TableEncoder(
        cell_encoder,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        nonlinearity=args.nonlinearity,
        channel_mix_hidden_dim=args.channel_mix_hidden_dim,
        num_heads=args.num_heads,
    )
    load_pretrained_encoder(model, args.pretrained_checkpoint, device=args.device)

    # weights and the text cache are independent state -- the checkpoint's
    # state_dict has no cache entries, so this is a separate load, not
    # something load_pretrained_encoder already covers.
    if os.path.exists(args.text_cache_path):
        print(f"loading cell/header text cache from {args.text_cache_path} ...")
        model.load_text_cache(args.text_cache_path)
        print(f"text cache warm-started with {cell_encoder.text_embedder.cache_size()} entries")

    query_encoder = QueryEncoder(
        model_name=args.query_model_name,
        output_dim=args.embed_dim,
        max_length=args.query_max_length,
        trainable=args.query_trainable,
        exclude_special_tokens=args.exclude_special_tokens,
    )

    trainer = FinetuneTrainer(
        model,
        query_encoder,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip_norm=args.grad_clip_norm,
        temperature=args.temperature,
        scoring_mode=args.scoring_mode,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed,
    )

    print(f"starting finetuning on {args.device} (scoring_mode={args.scoring_mode}, patience={args.patience}) ...")
    best_val_map = trainer.fit(
        build_train_batches,
        num_epochs=args.num_epochs,
        steps_per_epoch=steps_per_epoch,
        val_examples=val_examples,
        corpus_tables=corpus_tables_for_val,
        patience=args.patience,
        log_every=args.log_every,
    )

    print(f"\nbest validation MAP: {best_val_map:.4f}")

    # final test-set MAP, using the best-val-MAP checkpoint (not
    # whatever the last epoch trained happened to be -- early stopping
    # means those can differ).
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    test_map = None
    if os.path.exists(best_ckpt_path):
        trainer.load_checkpoint(best_ckpt_path)
        test_map = trainer.evaluate_map(test_examples, corpus_tables)
        print(f"test MAP (best-val-MAP checkpoint): {test_map:.4f}")
    else:
        print("no best checkpoint found (val MAP never improved) -- skipping test evaluation")

    # Persisted, structured record -- best_model.pt/train.log hold the
    # same numbers but aren't convenient to aggregate across runs/models
    # for a report table; this is. See scripts/run_all_models.sh, which
    # collects one of these per encoder into a single combined table.
    results = {
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
        "num_epochs_configured": args.num_epochs,
        "split_json": args.split_json,
        "corpus_json": args.corpus_json,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    results_path = os.path.join(args.checkpoint_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote results to {results_path}")

    os.makedirs(os.path.dirname(args.text_cache_path) or ".", exist_ok=True)
    model.save_text_cache(args.text_cache_path)
    print(
        f"saved text cache ({cell_encoder.text_embedder.cache_size()} entries) "
        f"to {args.text_cache_path}"
    )

    print("\nFinetuning complete.")
