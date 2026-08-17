"""
Standalone TEST-set evaluation -- STREAMING, memory-bounded, all queries.

Instead of scoring every query against one giant in-memory pool (which
materializes an [n_queries x n_corpus] score matrix and every corpus
table's 768-dim embedding at once -> hundreds of GB), this processes
queries in chunks: for each chunk it loads only the tables that chunk
actually needs (all tables in the databases those queries touch -- the
real same-db hard negatives -- plus an optional fixed cross-db filler),
scores that chunk, records per-query AP/MRR, frees, and moves on. So it
covers ALL test queries with flat memory.

Metric is exact: overall MAP/MRR is the mean of per-query AP/RR over every
query that has a positive in its pool (same rule as the training eval).

Edit MODELS below (checkpoint paths + the dims each was TRAINED with).
For baselines num_heads/header_mode are ignored; for 'ours' they matter.

Run:
  python -u -m scripts.eval_test \
    --databases_root ../SynSQL-2.5M/databases \
    --questions_json ../SynSQL-2.5M/questions_with_tables.json \
    --tables_json ../SynSQL-2.5M/tables.json \
    --device cuda:0 --query_batch_size 64 \
    --query_chunk_size 2000            # queries per streamed chunk
    # --corpus_sample_size 2000        # optional fixed cross-db filler
"""
from __future__ import annotations

import argparse
import os
import random
import types

import torch

from src.data.synsql_dataset import SynSQLTableDataset, SynSQLQueryDataset
from src.training.query_encoder import QueryEncoder
from src.training.trainer import FinetuneTrainer
from scripts.train_model import build_table_model


MODELS = [
    # Optional cache warm-start (skipped if the file doesn't exist):
    #   ours   -> text_cache  (per-cell BERT features; big win, cells repeat)
    #   tabbie/strubert -> frozen_cache
    #   any    -> query_cache (only helps if queries were seen; test queries
    #                          mostly weren't, so minor)
    dict(encoder="ours", embed_dim=768, num_layers=3, num_heads=8, header_mode="film",
         checkpoint="/mnt/nas/ayane/croissant/eval/report_runs/ours768mh/finetune/best_model.pt",
         text_cache="/tmp/ayane/caches768mh/ours768mh_text_cache.pt",
         query_cache="/tmp/ayane/caches768mh/ours768mh_query_cache.pt"),
    dict(encoder="turl", embed_dim=64, num_layers=3, num_heads=8, header_mode="concat",
         checkpoint="/mnt/nas/ayane/croissant/eval/report_runs/turl/finetune/best_model.pt"),
    dict(encoder="tabbie", embed_dim=64, num_layers=3, num_heads=8, header_mode="concat",
         checkpoint="/mnt/nas/ayane/croissant/eval/report_runs/tabbie/finetune/best_model.pt"),
]


def to_eval_examples(query_dataset, indices):
    out = []
    for i in indices:
        ex = query_dataset.examples[i]
        out.append((ex.question, ex.db_id, ex.table_names))
    return out


def per_query_metrics(scores, positive_mask):
    """Per-query AP and RR for every query that has >=1 positive, from a
    [n_queries, n_pool] score matrix + positive mask. Same vectorized
    ranking as src/eval/retrieval_metrics.compute_ranking_metrics, but
    returns the per-query lists so they can be accumulated across chunks."""
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores)
    if not isinstance(positive_mask, torch.Tensor):
        positive_mask = torch.as_tensor(positive_mask)
    scores = scores.float()
    positive_mask = positive_mask.float()
    nq, nc = scores.shape
    order = torch.argsort(scores, dim=1, descending=True)
    ranked = torch.gather(positive_mask, 1, order)
    ranks = torch.arange(1, nc + 1, dtype=scores.dtype).unsqueeze(0)
    prec = ranked.cumsum(dim=1) / ranks
    n_rel = positive_mask.sum(dim=1)
    has = n_rel > 0
    ap = (prec * ranked).sum(dim=1)
    ap = (ap[has] / n_rel[has]).tolist()
    first = torch.argmax(ranked, dim=1)
    rr = (1.0 / (first[has].to(scores.dtype) + 1.0)).tolist()
    return ap, rr


def streaming_eval(trainer, examples, corpus_tables, chunk_size, n_distract,
                   seed, query_batch_size, table_batch_size):
    def db_of(t):
        return t.table_id.split("#sep#", 1)[0]

    db_to_tables = {}
    for t in corpus_tables:
        db_to_tables.setdefault(db_of(t), []).append(t)

    # sort queries by db so each database's tables are encoded in as few
    # chunks as possible
    ex = sorted(examples, key=lambda e: e[1])
    total = len(ex)
    ap_all, rr_all = [], []

    for ci in range(0, total, chunk_size):
        chunk = ex[ci:ci + chunk_size]
        chunk_dbs = {e[1] for e in chunk}
        # every chunk query's own-db tables ...
        pool = [t for d in chunk_dbs for t in db_to_tables.get(d, [])]
        pool_ids = {t.table_id for t in pool}
        # ... plus a random reservoir (per chunk) so each query has at least
        # n_distract non-own-db tables available to draw distractors from.
        if n_distract > 0:
            res_rng = random.Random(seed * 7919 + ci)
            reservoir = res_rng.sample(corpus_tables, min(len(corpus_tables), 2 * n_distract))
            pool += [t for t in reservoir if t.table_id not in pool_ids]

        scores, pos = trainer._corpus_scores(
            chunk, pool, query_batch_size=query_batch_size, table_batch_size=table_batch_size,
        )

        # Each query sees ITS OWN db's tables + a random n_distract sample of
        # the OTHER pool tables. Drawn PER QUERY (so distractors vary across
        # queries; sourced from the chunk's other dbs + the reservoir), but
        # seeded by stream position so every model sees the identical
        # distractors for a given query -> fair comparison.
        pool_dbs = [db_of(t) for t in pool]
        uniq = {d: i for i, d in enumerate(sorted(set(pool_dbs) | chunk_dbs))}
        pool_db_t = torch.tensor([uniq[d] for d in pool_dbs])  # [npool]
        masked = torch.full_like(scores, float("-inf"))
        for i, e in enumerate(chunk):
            own = pool_db_t == uniq[e[1]]
            masked[i, own] = scores[i, own]
            if n_distract > 0:
                others = (~own).nonzero(as_tuple=True)[0]
                if others.numel() > n_distract:
                    g = torch.Generator().manual_seed(seed * 1000003 + ci + i)
                    others = others[torch.randperm(others.numel(), generator=g)[:n_distract]]
                masked[i, others] = scores[i, others]

        ap, rr = per_query_metrics(masked, pos)
        ap_all.extend(ap)
        rr_all.extend(rr)

        done = min(ci + chunk_size, total)
        run_map = sum(ap_all) / max(1, len(ap_all))
        run_mrr = sum(rr_all) / max(1, len(rr_all))
        print(f"  [{done}/{total} q] each query vs own-db + {n_distract} random distractors | "
              f"running MAP {run_map:.4f}  MRR {run_mrr:.4f}", flush=True)

    return {
        "map": sum(ap_all) / max(1, len(ap_all)),
        "mrr": sum(rr_all) / max(1, len(rr_all)),
        "n_scored": len(ap_all),
    }


def _infer_dims(model_sd):
    import re
    embed_dim = None
    for k, v in model_sd.items():
        if k.endswith("projection.weight") and v.ndim == 2:
            embed_dim = v.shape[0]
    layers = set()
    for k in model_sd:
        m = re.search(r"layers?\.(\d+)\.", k)
        if m:
            layers.add(int(m.group(1)))
    return embed_dim, (max(layers) + 1 if layers else None)


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
    # streaming
    ap.add_argument("--query_chunk_size", type=int, default=2000,
                    help="test queries processed per streamed chunk (memory knob)")
    ap.add_argument("--corpus_sample_size", type=int, default=2000,
                    help="number of random distractor tables sampled PER QUERY (varying across "
                         "queries), on top of the query's own-db tables (0 = own-db only)")
    ap.add_argument("--query_sample_size", type=int, default=0,
                    help="optionally cap total test queries (0 = all) for a quick check")
    ap.add_argument("--seed", type=int, default=42)
    # query tower -- MUST match configs/finetune.yaml
    ap.add_argument("--query_model_name", default="bert-base-uncased")
    ap.add_argument("--query_max_length", type=int, default=32)
    ap.add_argument("--query_trainable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--exclude_special_tokens", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--query_cache", default="/tmp/ayane/caches768mh/ours768mh_query_cache.pt",
                    help="shared frozen query cache (model-independent); loaded for every model")
    # 'ours' cell encoder
    ap.add_argument("--text_model_name", default="bert-base-uncased")
    ap.add_argument("--text_max_length", type=int, default=32)
    ap.add_argument("--text_max_batch_size", type=int, default=2048)
    ap.add_argument("--text_trainable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--nonlinearity", default="sigmoid")
    ap.add_argument("--channel_mix_hidden_dim", type=int, default=None)
    ap.add_argument("--model_name", default=None)
    args = ap.parse_args()

    print(f"indexing tables from {args.tables_json} / {args.databases_root} ...", flush=True)
    table_dataset = SynSQLTableDataset(
        tables_json=args.tables_json, databases_root=args.databases_root, max_rows=args.max_rows,
    )
    query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    resolved = query_dataset.resolve_split(args.split_json)
    test_examples = to_eval_examples(query_dataset, resolved["test"])
    if args.query_sample_size and 0 < args.query_sample_size < len(test_examples):
        test_examples = random.Random(args.seed).sample(test_examples, args.query_sample_size)
        print(f"[eval] sampled {len(test_examples)} test queries (seed {args.seed})", flush=True)
    corpus_tables = table_dataset.load_corpus(args.corpus_json)
    print(f"[eval] {len(test_examples)} test queries; corpus {len(corpus_tables)} tables; "
          f"chunk={args.query_chunk_size}, filler={args.corpus_sample_size}\n", flush=True)

    results = []
    for spec in MODELS:
        enc, ckpt = spec["encoder"], spec["checkpoint"]
        print(f"=== {enc} :: {ckpt} ===", flush=True)
        try:
            raw = torch.load(ckpt, map_location="cpu")
        except Exception as e:
            print(f"  !! could not load checkpoint ({type(e).__name__}: {e}) -- skipping\n")
            results.append((enc, None, None, None))
            continue

        inf_dim, inf_layers = _infer_dims(raw.get("model_state_dict", {}))
        if inf_dim is not None and inf_dim != spec["embed_dim"]:
            print(f"  WARNING: checkpoint proj dim={inf_dim} but config embed_dim={spec['embed_dim']}")
        if inf_layers is not None and inf_layers != spec["num_layers"]:
            print(f"  WARNING: checkpoint has {inf_layers} layers but config num_layers={spec['num_layers']}")

        margs = types.SimpleNamespace(
            encoder=enc, embed_dim=spec["embed_dim"], num_layers=spec["num_layers"],
            num_heads=spec.get("num_heads", 8), header_mode=spec.get("header_mode", "concat"),
            nonlinearity=args.nonlinearity, channel_mix_hidden_dim=args.channel_mix_hidden_dim,
            text_model_name=args.text_model_name, text_max_length=args.text_max_length,
            text_trainable=args.text_trainable, text_max_batch_size=args.text_max_batch_size,
            model_name=args.model_name, device=args.device,
        )
        model = build_table_model(margs)
        query_encoder = QueryEncoder(
            model_name=args.query_model_name, output_dim=spec["embed_dim"],
            max_length=args.query_max_length, trainable=args.query_trainable,
            exclude_special_tokens=args.exclude_special_tokens,
        )
        trainer = FinetuneTrainer(
            model, query_encoder, scoring_mode=args.scoring_mode, device=args.device,
            checkpoint_dir=f"/tmp/eval_scratch_{enc}",
        )
        trainer.load_checkpoint(ckpt)

        # warm-start caches so the corpus encode reuses precomputed features
        def _try_cache(fn, path, label):
            if fn is not None and path and os.path.exists(path):
                try:
                    fn(path)
                    print(f"  loaded {label} cache: {path}", flush=True)
                except Exception as ex:
                    print(f"  {label} cache load skipped ({type(ex).__name__}: {ex})", flush=True)
        if enc == "ours":
            _try_cache(model.load_text_cache, spec.get("text_cache"), "text")
        if enc in ("tabbie", "strubert"):
            _try_cache(getattr(getattr(model, "baseline_encoder", None), "load_frozen_cache", None),
                       spec.get("frozen_cache"), "frozen")
        if not args.query_trainable:
            _try_cache(query_encoder.load_frozen_cache, args.query_cache, "query")

        m = streaming_eval(
            trainer, test_examples, corpus_tables,
            chunk_size=args.query_chunk_size, n_distract=args.corpus_sample_size,
            seed=args.seed, query_batch_size=args.query_batch_size,
            table_batch_size=args.table_batch_size,
        )
        vm = raw.get("val_map")
        print(f"[{enc}] test MAP {m['map']:.4f} | test MRR {m['mrr']:.4f}  "
              f"(over {m['n_scored']} scored queries; ckpt epoch={raw.get('epoch')}, val_map={vm})\n", flush=True)
        results.append((enc, m["map"], m["mrr"], vm))

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
