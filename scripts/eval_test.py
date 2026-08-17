"""
Standalone TEST-set evaluation.

Loads one or more finetuned best_model.pt checkpoints and computes
**test MAP + MRR over the FULL corpus** (the complete test query split
ranked against every corpus table), reusing the exact evaluation path
train_model.py runs at the end of finetuning
(FinetuneTrainer.evaluate_ranking_metrics -> _corpus_scores).

The expensive data load (indexing tables, the full corpus, the test
split) happens ONCE and is shared across every model, so evaluating
three models costs one data load + three corpus-scoring passes.

Edit the MODELS list below with each checkpoint's path and the
architecture dims it was TRAINED with (embed_dim / num_layers / num_heads
/ header_mode) -- these must match the checkpoint or load_state_dict will
error on a shape mismatch. For baselines num_heads/header_mode are
ignored; for 'ours' they matter.

Run:
  python -m scripts.eval_test \
    --databases_root ../SynSQL-2.5M/databases \
    --questions_json ../SynSQL-2.5M/questions_with_tables.json \
    --tables_json ../SynSQL-2.5M/tables.json \
    --device cuda:0 --query_batch_size 64

NOTE: scoring the FULL corpus (~166k tables) per model is compute-heavy,
especially for token-level baselines (turl). Expect this to take a while.
"""
from __future__ import annotations

import argparse
import random
import types

import torch

from src.data.synsql_dataset import SynSQLTableDataset, SynSQLQueryDataset
from src.training.query_encoder import QueryEncoder
from src.training.trainer import FinetuneTrainer
from scripts.train_model import build_table_model


# ---- edit these: one entry per model to evaluate ----------------------
# checkpoint = path to best_model.pt
# embed_dim / num_layers MUST match how that checkpoint was trained.
# num_heads / header_mode only matter for encoder == "ours".
MODELS = [
    dict(
        encoder="ours", embed_dim=768, num_layers=3, num_heads=8, header_mode="film",
        checkpoint="/mnt/nas/ayane/croissant/eval/report_runs/ours768mh/finetune/best_model.pt",
    ),
    dict(
        encoder="turl", embed_dim=64, num_layers=3, num_heads=8, header_mode="concat",
        checkpoint="/mnt/nas/ayane/croissant/eval/report_runs/turl/finetune/best_model.pt",
    ),
    dict(
        encoder="tabbie", embed_dim=64, num_layers=3, num_heads=8, header_mode="concat",
        checkpoint="/mnt/nas/ayane/croissant/eval/report_runs/tabbie/finetune/best_model.pt",
    ),
]
# -----------------------------------------------------------------------


def to_eval_examples(query_dataset, indices):
    """(question, db_id, table_names) tuples -- same shape evaluate_*
    expects (see scripts/finetune_query_table.py::to_eval_examples)."""
    out = []
    for i in indices:
        ex = query_dataset.examples[i]
        out.append((ex.question, ex.db_id, ex.table_names))
    return out


def subsample_corpus(corpus_tables, examples, sample_size, n_hard, seed, all_same_db):
    """Build a fixed candidate pool that keeps every query's gold tables
    AND (by default) every table in the same database as any test query
    -- the real hard negatives for table retrieval, since same-schema
    tables are the genuinely confusable ones. Then top up with random
    cross-db filler as easy negatives up to --corpus_sample_size.

    Shared across every model, so the comparison is fair. The absolute
    MAP is over this pool, not the full corpus.

    all_same_db=True  : forced = ALL tables in every db a test query
                        touches (gold positives + all same-db distractors).
    all_same_db=False : forced = gold positives + n_hard random same-db
                        negatives per db (the lighter val-style sampling).
    """
    def db_of(t):
        return t.table_id.split("#sep#", 1)[0]

    positive_ids = {f"{db_id}#sep#{t}" for _q, db_id, tabs in examples for t in tabs}
    query_db_ids = {db_id for _q, db_id, _t in examples}

    if all_same_db:
        forced = [t for t in corpus_tables if db_of(t) in query_db_ids]
        n_pos = sum(1 for t in forced if t.table_id in positive_ids)
        detail = f"{n_pos} gold + {len(forced) - n_pos} same-db distractors (ALL same-db tables)"
    else:
        forced = [t for t in corpus_tables if t.table_id in positive_ids]
        forced_ids = {t.table_id for t in forced}
        db_to_tables = {}
        for t in corpus_tables:
            db_to_tables.setdefault(db_of(t), []).append(t)
        n_hard_added = 0
        rng = random.Random(seed)
        for db_id in query_db_ids:
            cands = [t for t in db_to_tables.get(db_id, []) if t.table_id not in forced_ids]
            rng.shuffle(cands)
            for t in cands[:n_hard]:
                forced.append(t)
                forced_ids.add(t.table_id)
                n_hard_added += 1
        detail = f"{len(forced) - n_hard_added} gold + {n_hard_added} sampled same-db negs"

    forced_ids = {t.table_id for t in forced}
    remaining = [t for t in corpus_tables if t.table_id not in forced_ids]  # other dbs
    n_fill = max(0, sample_size - len(forced))
    filler = random.Random(seed).sample(remaining, min(n_fill, len(remaining)))
    pool = forced + filler
    print(f"[eval] corpus pool: {len(pool)}/{len(corpus_tables)} tables "
          f"({detail} + {len(filler)} cross-db filler)")
    if len(forced) >= sample_size:
        print(f"  note: forced set ({len(forced)}) >= --corpus_sample_size ({sample_size}); "
              f"no cross-db filler added (pool = all same-db candidates).")
    return pool


def _infer_dims(model_sd):
    """Best-effort read of embed_dim / num_layers straight from the
    checkpoint, so we can WARN if the configured dims disagree (a
    mismatch would otherwise surface as an opaque load_state_dict error).
    """
    import re
    embed_dim = None
    for k, v in model_sd.items():
        if k.endswith("projection.weight") and v.ndim == 2:   # baselines
            embed_dim = v.shape[0]
    layers = set()
    for k in model_sd:
        m = re.search(r"layers?\.(\d+)\.", k)
        if m:
            layers.add(int(m.group(1)))
    num_layers = (max(layers) + 1) if layers else None
    return embed_dim, num_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--databases_root", required=True)
    ap.add_argument("--questions_json", required=True)
    ap.add_argument("--tables_json", default=None)
    ap.add_argument("--split_json", default="configs/splits/query_split.json")
    ap.add_argument("--corpus_json", default="configs/splits/corpus.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_rows", type=int, default=50)
    ap.add_argument("--query_batch_size", type=int, default=64)
    ap.add_argument("--table_batch_size", type=int, default=32)
    ap.add_argument("--scoring_mode", default="row_match")
    # Corpus subsampling: rank ALL test queries against a fixed pool of
    # this many tables (gold positives always kept) instead of the full
    # ~166k corpus. 0 = use the full corpus (slow, the "real" number).
    # Any positive value mirrors the val eval, so results land on the
    # same scale as the val MAPs -- and is far faster.
    ap.add_argument("--corpus_sample_size", type=int, default=0)
    ap.add_argument("--n_hard_negatives", type=int, default=2,
                    help="only used when --no-all_same_db: random same-db negs per db")
    ap.add_argument("--all_same_db", action=argparse.BooleanOptionalAction, default=True,
                    help="include ALL tables in each query's database as hard negatives "
                         "(default). --no-all_same_db falls back to n_hard_negatives random ones.")
    ap.add_argument("--seed", type=int, default=42)
    # query tower -- must match training so the state_dict loads and
    # scoring behaves identically
    # Defaults MUST match configs/finetune.yaml so the query tower matches
    # the checkpoints and scoring is identical to training.
    ap.add_argument("--query_model_name", default="bert-base-uncased")
    ap.add_argument("--query_max_length", type=int, default=32)
    ap.add_argument("--query_trainable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--exclude_special_tokens", action=argparse.BooleanOptionalAction, default=False)
    # 'ours' cell encoder -- match training defaults
    ap.add_argument("--text_model_name", default="bert-base-uncased")
    ap.add_argument("--text_max_length", type=int, default=32)
    ap.add_argument("--text_max_batch_size", type=int, default=2048)
    ap.add_argument("--text_trainable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--nonlinearity", default="sigmoid")
    ap.add_argument("--channel_mix_hidden_dim", type=int, default=None)
    ap.add_argument("--model_name", default=None)
    args = ap.parse_args()

    # ---- shared data load (once) --------------------------------------
    print(f"indexing tables from {args.tables_json} / {args.databases_root} ...")
    table_dataset = SynSQLTableDataset(
        tables_json=args.tables_json,
        databases_root=args.databases_root,
        max_rows=args.max_rows,
    )
    query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    resolved = query_dataset.resolve_split(args.split_json)
    test_examples = to_eval_examples(query_dataset, resolved["test"])
    corpus_tables = table_dataset.load_corpus(args.corpus_json)
    print(f"[eval] {len(test_examples)} test queries; full corpus {len(corpus_tables)} tables")

    # One fixed pool, shared across every model so the comparison is fair.
    # Default: all same-db tables per query (+ optional cross-db filler via
    # --corpus_sample_size). Full corpus only when explicitly disabled.
    use_full = (not args.all_same_db) and not (args.corpus_sample_size and args.corpus_sample_size > 0)
    if use_full:
        eval_corpus = corpus_tables
        print("[eval] using FULL corpus (slow; the true full-corpus number)")
    else:
        eval_corpus = subsample_corpus(
            corpus_tables, test_examples, args.corpus_sample_size,
            args.n_hard_negatives, args.seed, args.all_same_db,
        )
    print()

    results = []
    for spec in MODELS:
        enc, ckpt = spec["encoder"], spec["checkpoint"]
        print(f"=== {enc} :: {ckpt} ===")
        try:
            raw = torch.load(ckpt, map_location="cpu")
        except Exception as e:
            print(f"  !! could not load checkpoint ({type(e).__name__}: {e}) -- skipping\n")
            results.append((enc, None, None, None))
            continue

        inf_dim, inf_layers = _infer_dims(raw.get("model_state_dict", {}))
        if inf_dim is not None and inf_dim != spec["embed_dim"]:
            print(f"  WARNING: checkpoint projection dim={inf_dim} but config embed_dim={spec['embed_dim']}")
        if inf_layers is not None and inf_layers != spec["num_layers"]:
            print(f"  WARNING: checkpoint has {inf_layers} layers but config num_layers={spec['num_layers']}")

        margs = types.SimpleNamespace(
            encoder=enc,
            embed_dim=spec["embed_dim"],
            num_layers=spec["num_layers"],
            num_heads=spec.get("num_heads", 8),
            header_mode=spec.get("header_mode", "concat"),
            nonlinearity=args.nonlinearity,
            channel_mix_hidden_dim=args.channel_mix_hidden_dim,
            text_model_name=args.text_model_name,
            text_max_length=args.text_max_length,
            text_trainable=args.text_trainable,
            text_max_batch_size=args.text_max_batch_size,
            model_name=args.model_name,
            device=args.device,
        )
        model = build_table_model(margs)
        query_encoder = QueryEncoder(
            model_name=args.query_model_name,
            output_dim=spec["embed_dim"],
            max_length=args.query_max_length,
            trainable=args.query_trainable,
            exclude_special_tokens=args.exclude_special_tokens,
        )
        trainer = FinetuneTrainer(
            model,
            query_encoder,
            scoring_mode=args.scoring_mode,
            device=args.device,
            checkpoint_dir=f"/tmp/eval_scratch_{enc}",
        )
        trainer.load_checkpoint(ckpt)  # restores model + query_encoder + scorer (+ global_step)

        m = trainer.evaluate_ranking_metrics(
            test_examples, eval_corpus,
            query_batch_size=args.query_batch_size,
            table_batch_size=args.table_batch_size,
        )
        vm = raw.get("val_map")
        print(f"[{enc}] test MAP {m['map']:.4f} | test MRR {m['mrr']:.4f}  "
              f"(ckpt epoch={raw.get('epoch')}, stored val_map={vm})\n")
        results.append((enc, m["map"], m["mrr"], vm))

        # free GPU memory before the next model
        del model, query_encoder, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("==================  TEST RESULTS  ==================")
    print(f"{'model':12} {'test_MAP':>9} {'test_MRR':>9} {'val_MAP':>9}")
    for enc, mp, mr, vm in results:
        mp_s = f"{mp:9.4f}" if mp is not None else "     n/a "
        mr_s = f"{mr:9.4f}" if mr is not None else "     n/a "
        vm_s = f"{vm:9.4f}" if isinstance(vm, (int, float)) else "     n/a "
        print(f"{enc:12} {mp_s} {mr_s} {vm_s}")
    print("===================================================")


if __name__ == "__main__":
    main()
