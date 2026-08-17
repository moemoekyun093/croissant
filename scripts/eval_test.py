"""
Standalone TEST-set evaluation -- STREAMING, all queries, all models scored
against ONE shared candidate set per query.

Structure: the outer loop is over query CHUNKS, the inner loop is over
models. For each chunk we build the candidate pool + each query's distractor
set exactly ONCE, then score every model against that identical set. So:

  * the candidate set is provably identical across models (same tables, same
    per-query distractors) -> a clean apples-to-apples comparison,
  * the query BERT features are encoded once and reused (frozen query tower),
  * memory stays flat -- only one chunk's pool is materialized at a time.

Per query: its own database's tables (ALL of them) + `--corpus_sample_size`
random distractor tables, sampled PER QUERY (varying across queries, drawn
from the chunk's other dbs + a random reservoir). Metric = mean per-query
AP/RR over queries that have a positive in their set.

Edit MODELS below (checkpoint paths + trained dims; optional cache paths).
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
    dict(encoder="ours", embed_dim=768, num_layers=3, num_heads=8, header_mode="film",
         checkpoint="/mnt/nas/ayane/croissant/eval/report_runs/ours768mh/finetune/best_model.pt",
         text_cache="/tmp/ayane/caches768mh/ours768mh_text_cache.pt"),
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
    """Per-query AP + RR for every query with >=1 positive, from an
    [n_queries, n_pool] score matrix + positive mask (vectorized ranking,
    same as retrieval_metrics.compute_ranking_metrics)."""
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


def build_pool_and_mask(chunk, db_to_tables, corpus_tables, n_distract, seed, ci):
    """Build the chunk's candidate pool and the per-query visibility mask
    ONCE (shared by every model). Returns (pool, visible[bool nq x npool])."""
    def db_of(t):
        return t.table_id.split("#sep#", 1)[0]

    chunk_dbs = {e[1] for e in chunk}
    pool = [t for d in chunk_dbs for t in db_to_tables.get(d, [])]
    pool_ids = {t.table_id for t in pool}
    if n_distract > 0:
        res_rng = random.Random(seed * 7919 + ci)
        reservoir = res_rng.sample(corpus_tables, min(len(corpus_tables), 2 * n_distract))
        pool += [t for t in reservoir if t.table_id not in pool_ids]

    pool_dbs = [db_of(t) for t in pool]
    uniq = {d: i for i, d in enumerate(sorted(set(pool_dbs) | chunk_dbs))}
    pool_db_t = torch.tensor([uniq[d] for d in pool_dbs])  # [npool]

    nq, npool = len(chunk), len(pool)
    visible = torch.zeros(nq, npool, dtype=torch.bool)
    for i, e in enumerate(chunk):
        own = pool_db_t == uniq[e[1]]
        visible[i] |= own
        if n_distract > 0:
            others = (~own).nonzero(as_tuple=True)[0]
            if others.numel() > n_distract:
                g = torch.Generator().manual_seed(seed * 1000003 + ci + i)
                others = others[torch.randperm(others.numel(), generator=g)[:n_distract]]
            visible[i, others] = True
    return pool, visible


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
    ap.add_argument("--query_chunk_size", type=int, default=2000)
    ap.add_argument("--corpus_sample_size", type=int, default=2000,
                    help="random distractor tables per query (varying), on top of its own db")
    ap.add_argument("--query_sample_size", type=int, default=0, help="cap total test queries (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    # query tower -- match configs/finetune.yaml
    ap.add_argument("--query_model_name", default="bert-base-uncased")
    ap.add_argument("--query_max_length", type=int, default=32)
    ap.add_argument("--query_trainable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--exclude_special_tokens", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--query_cache", default="/tmp/ayane/caches768mh/ours768mh_query_cache.pt")
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
        tables_json=args.tables_json, databases_root=args.databases_root, max_rows=args.max_rows)
    query_dataset = SynSQLQueryDataset(args.questions_json, table_dataset)
    resolved = query_dataset.resolve_split(args.split_json)
    test_examples = to_eval_examples(query_dataset, resolved["test"])
    if args.query_sample_size and 0 < args.query_sample_size < len(test_examples):
        test_examples = random.Random(args.seed).sample(test_examples, args.query_sample_size)
        print(f"[eval] sampled {len(test_examples)} test queries (seed {args.seed})", flush=True)
    corpus_tables = table_dataset.load_corpus(args.corpus_json)
    print(f"[eval] {len(test_examples)} test queries; corpus {len(corpus_tables)} tables; "
          f"chunk={args.query_chunk_size}, distractors/query={args.corpus_sample_size}\n", flush=True)

    def db_of(t):
        return t.table_id.split("#sep#", 1)[0]
    db_to_tables = {}
    for t in corpus_tables:
        db_to_tables.setdefault(db_of(t), []).append(t)

    def _try_cache(fn, path, label):
        if fn is not None and path and os.path.exists(path):
            try:
                fn(path)
                print(f"  loaded {label} cache: {path}", flush=True)
            except Exception as ex:
                print(f"  {label} cache skipped ({type(ex).__name__}: {ex})", flush=True)

    # ---- build every model up front (kept resident) --------------------
    shared_query_cache = {}
    runners = []
    for spec in MODELS:
        enc, ckpt = spec["encoder"], spec["checkpoint"]
        print(f"=== loading {enc} :: {ckpt} ===", flush=True)
        try:
            raw = torch.load(ckpt, map_location="cpu")
        except Exception as e:
            print(f"  !! could not load checkpoint ({type(e).__name__}: {e}) -- skipping\n")
            continue
        margs = types.SimpleNamespace(
            encoder=enc, embed_dim=spec["embed_dim"], num_layers=spec["num_layers"],
            num_heads=spec.get("num_heads", 8), header_mode=spec.get("header_mode", "concat"),
            nonlinearity=args.nonlinearity, channel_mix_hidden_dim=args.channel_mix_hidden_dim,
            text_model_name=args.text_model_name, text_max_length=args.text_max_length,
            text_trainable=args.text_trainable, text_max_batch_size=args.text_max_batch_size,
            model_name=args.model_name, device=args.device)
        model = build_table_model(margs)
        query_encoder = QueryEncoder(
            model_name=args.query_model_name, output_dim=spec["embed_dim"],
            max_length=args.query_max_length, trainable=args.query_trainable,
            exclude_special_tokens=args.exclude_special_tokens)
        if not args.query_trainable:
            query_encoder._encoder_cache = shared_query_cache
        trainer = FinetuneTrainer(
            model, query_encoder, scoring_mode=args.scoring_mode, device=args.device,
            checkpoint_dir=f"/tmp/eval_scratch_{enc}")
        trainer.load_checkpoint(ckpt)
        if enc == "ours":
            _try_cache(model.load_text_cache, spec.get("text_cache"), "text")
        if enc in ("tabbie", "strubert"):
            _try_cache(getattr(getattr(model, "baseline_encoder", None), "load_frozen_cache", None),
                       spec.get("frozen_cache"), "frozen")
        if not args.query_trainable:
            _try_cache(query_encoder.load_frozen_cache, args.query_cache, "query")
        runners.append({"enc": enc, "trainer": trainer, "ap": [], "rr": [],
                        "val_map": raw.get("val_map"), "epoch": raw.get("epoch")})

    # ---- stream chunks; score EVERY model against the SAME set ----------
    ex = sorted(test_examples, key=lambda e: e[1])
    total = len(ex)
    print(f"\n[eval] streaming {total} queries in chunks of {args.query_chunk_size}\n", flush=True)
    for ci in range(0, total, args.query_chunk_size):
        chunk = ex[ci:ci + args.query_chunk_size]
        pool, visible = build_pool_and_mask(
            chunk, db_to_tables, corpus_tables, args.corpus_sample_size, args.seed, ci)
        for r in runners:
            scores, pos = r["trainer"]._corpus_scores(
                chunk, pool, query_batch_size=args.query_batch_size,
                table_batch_size=args.table_batch_size)
            masked = scores.masked_fill(~visible, float("-inf"))
            a, rr = per_query_metrics(masked, pos)
            r["ap"].extend(a)
            r["rr"].extend(rr)
        done = min(ci + args.query_chunk_size, total)
        summary = "  ".join(
            f"{r['enc']} MAP {sum(r['ap'])/max(1,len(r['ap'])):.4f}" for r in runners)
        print(f"  [{done}/{total} q | pool {len(pool)}] {summary}", flush=True)

    print("\n==================  TEST RESULTS  ==================")
    print(f"{'model':12} {'test_MAP':>9} {'test_MRR':>9} {'val_MAP':>9}")
    for r in runners:
        mp = sum(r["ap"]) / max(1, len(r["ap"]))
        mr = sum(r["rr"]) / max(1, len(r["rr"]))
        vm = r["val_map"]
        vm_s = f"{vm:9.4f}" if isinstance(vm, (int, float)) else "     n/a "
        print(f"{r['enc']:12} {mp:9.4f} {mr:9.4f} {vm_s}")
    print("===================================================")


if __name__ == "__main__":
    main()
