"""
Evaluates the trained model's MaxSim similarity specifically on the BIRD
tables. No labeled query/match pairs exist, so this uses a weak but real
proxy: BIRD tables sharing the same source database -- encoded in
table_name as "{db_id}#sep#{table}", per build_bird_jsonl.py -- are more
likely to be genuinely related than tables from different databases.

For each BIRD table, ranks every other BIRD table by MaxSim and checks
how often the top-k matches come from the SAME database, compared
against the random-chance baseline (what you'd expect if the scores
carried no signal at all).

Also saves the full ranked list of closest tables per table to a JSON
file -- the aggregate lift numbers tell you WHETHER it's working, this
file lets you actually inspect WHAT it's finding, table by table.

Loads BIRD tables directly from the corpus JSONL (via
table_from_corpus_record, same as webtables) rather than re-querying
the raw SQLite databases -- the corpus now has properly structured,
already-imputed BIRD records (imputation happens once, at corpus-build
time, inside build_bird_jsonl.py), so reading the corpus directly
reflects whatever data-quality fixes have actually been applied there.

Usage:
    python -m scripts.bird_similarity_eval \
        --corpus_jsonl /mnt/nas/ayane/tables/big_corpus.jsonl \
        --checkpoint eval/report_runs/run_full/checkpoint_epoch2.pt \
        --embed_dim 64 --device cuda:2
"""

import argparse
import json
import os
from collections import Counter

import torch
import torch.nn.functional as F

from src.data.corpus_loader import table_from_corpus_record
from src.data.table import Table
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import TableEncoder
from src.scoring.maxsim import batched_maxsim_matrix


def compute_column_alignment(
    query_repr: torch.Tensor,
    query_table: Table,
    doc_repr: torch.Tensor,
    doc_table: Table,
) -> list[dict]:
    """
    For each column in query_table, finds its single best-matching column
    in doc_table -- exactly the argmax MaxSim already computes internally
    per query column, just surfaced here instead of thrown away after
    the max() call.

    query_repr: [n_q, k], doc_repr: [n_d, k]
    returns: list of n_q dicts, one per query column, in query column order
    """

    q = F.normalize(query_repr, dim=-1)
    d = F.normalize(doc_repr, dim=-1)

    sim = q @ d.transpose(0, 1)  # [n_q, n_d]
    best_scores, best_idx = sim.max(dim=-1)

    alignment = []
    for qi in range(query_repr.shape[0]):
        di = best_idx[qi].item()
        alignment.append(
            {
                "query_column": query_table.columns[qi].header,
                "matched_column": doc_table.columns[di].header,
                "cosine_sim": best_scores[qi].item(),
            }
        )

    return alignment


def single_best_column_pair(reprs_i: torch.Tensor, reprs_j: torch.Tensor):
    """
    For two tables' column reprs [n_i,k] and [n_j,k], finds the SINGLE
    highest-cosine-similarity column pair between them -- a different,
    complementary signal to aggregate MaxSim (which sums each query
    column's best match, rather than surfacing one standout pair).

    Useful for join-key/foreign-key-style discovery, where ONE strongly
    matching column pair can be meaningful even if a table's other
    columns are unrelated -- but more vulnerable to spurious
    single-column collisions than the aggregate score, since there's no
    averaging against a table's other columns to dilute a coincidence.
    Report this ALONGSIDE the aggregate ranking, not as a replacement.

    returns: (best_score, best_query_col_idx, best_doc_col_idx)
    """

    qi = F.normalize(reprs_i, dim=-1)
    dj = F.normalize(reprs_j, dim=-1)
    sim_matrix = qi @ dj.transpose(0, 1)  # [n_i, n_j]

    flat_idx = sim_matrix.argmax()
    best_qi = (flat_idx // sim_matrix.shape[1]).item()
    best_dj = (flat_idx % sim_matrix.shape[1]).item()
    best_score = sim_matrix[best_qi, best_dj].item()

    return best_score, best_qi, best_dj


def get_db_id(table: Table) -> str:
    """table_name is "{db_id}#sep#{table_name_original}"."""
    return table.table_name.split("#sep#")[0]


def cap_columns(table: Table, max_columns: int = 20) -> Table:
    if len(table.columns) <= max_columns:
        return table
    return Table(
        table_id=table.table_id,
        table_name=table.table_name,
        columns=table.columns[:max_columns],
    )


def encode_all(
    model: TableEncoder,
    tables: list[Table],
    batch_size: int = 64,
    ablation: str | None = None,
) -> list[torch.Tensor]:
    """
    Encodes in size-bucketed chunks. Returns a ragged list of [n_i, k]
    representations, in the SAME order as `tables`.

    ablation: None (normal, both headers+content), "headers_only"
        (cell content zeroed -- retrieval using ONLY column names), or
        "content_only" (headers zeroed -- retrieval using ONLY cell
        values). Lets you see how BIRD retrieval quality changes when
        one signal is unavailable, not just the aggregate InfoNCE loss
        from ablation_check.py.
    """

    order = sorted(
        range(len(tables)),
        key=lambda i: (tables[i].num_columns, tables[i].num_rows),
    )
    reprs: list[torch.Tensor | None] = [None] * len(tables)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx_chunk = order[start : start + batch_size]
            chunk = [tables[i] for i in idx_chunk]

            X, _col_mask = model.forward_batch(chunk, ablation=ablation)

            for local_i, global_i in enumerate(idx_chunk):
                n = chunk[local_i].num_columns
                reprs[global_i] = X[local_i, :n].cpu()

    return reprs  # type: ignore[return-value]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1, help="must match the checkpoint being loaded")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_columns", type=int, default=20)
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--ablation",
        default=None,
        choices=[None, "headers_only", "content_only"],
        help="None (normal), 'headers_only' (zero cell content, "
        "retrieval using ONLY column names), or 'content_only' (zero "
        "headers, retrieval using ONLY cell values)",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="path to save the full ranked list of closest tables per "
        "table (JSON). Defaults to bird_closest_matches.json next to "
        "the checkpoint.",
    )
    parser.add_argument(
        "--output_top_k",
        type=int,
        default=10,
        help="number of closest matches to save per table in --output_file",
    )
    args = parser.parse_args()

    if args.output_file is None:
        suffix = f"_{args.ablation}" if args.ablation else ""
        args.output_file = os.path.join(
            os.path.dirname(args.checkpoint) or ".",
            f"bird_closest_matches{suffix}.json",
        )

    print("loading BIRD tables directly from corpus...")
    # reads whatever is ACTUALLY in the corpus -- including any
    # imputation applied by build_bird_jsonl.py at corpus-build time.
    # No longer re-queries the raw SQLite databases directly (that
    # bypassed imputation entirely, since imputation only happens
    # inside build_bird_jsonl.py's own corpus-construction step).
    tables = []
    with open(args.corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("source") != "bird":
                continue
            table = table_from_corpus_record(record)
            if table.num_columns > 0:
                tables.append(table)

    if len(tables) == 0:
        raise SystemExit(
            f"No usable BIRD tables found in {args.corpus_jsonl}.\n"
            f"Check with: grep -c '\"source\": \"bird\"' {args.corpus_jsonl}"
        )
    print(f"loaded {len(tables)} BIRD tables from corpus")

    tables = [cap_columns(t, args.max_columns) for t in tables]

    db_ids = [get_db_id(t) for t in tables]
    n_dbs = len(set(db_ids))
    print(f"across {n_dbs} distinct source databases")

    cell_encoder = CellEncoder(text_model_name="bert-base-uncased", output_dim=args.embed_dim)
    model = TableEncoder(cell_encoder, embed_dim=args.embed_dim, num_layers=args.num_layers)

    print(f"loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)

    print("encoding all BIRD tables...")
    reprs = encode_all(model, tables, batch_size=args.batch_size, ablation=args.ablation)

    print("computing full pairwise MaxSim matrix...")
    reprs_device = [r.to(args.device) for r in reprs]
    sims = batched_maxsim_matrix(reprs_device).cpu()  # [N, N]
    sims.fill_diagonal_(float("-inf"))  # exclude self-match

    N = len(tables)
    db_counts = Counter(db_ids)

    results = {k: [] for k in args.top_k}
    baseline_precisions = []
    reciprocal_ranks = []

    output_top_k = min(args.output_top_k, N - 1)
    closest_matches = []

    for i in range(N):
        same_db_total = db_counts[db_ids[i]] - 1  # exclude itself
        baseline_precisions.append(same_db_total / max(1, N - 1))

        order_i = torch.argsort(sims[i], descending=True)

        rank = None
        for r, j in enumerate(order_i.tolist(), start=1):
            if db_ids[j] == db_ids[i]:
                rank = r
                break
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        for k in args.top_k:
            top_k_idx = order_i[:k].tolist()
            hits = sum(1 for j in top_k_idx if db_ids[j] == db_ids[i])
            results[k].append(hits / k)

        # save the full ranked list for this table, with per-column alignment
        matches = []
        for rank_pos, j in enumerate(order_i[:output_top_k].tolist()):
            alignment = compute_column_alignment(
                reprs_device[i], tables[i], reprs_device[j], tables[j]
            )
            matches.append(
                {
                    "table_id": tables[j].table_id,
                    "table_name": tables[j].table_name,
                    "db_id": db_ids[j],
                    "score": sims[i, j].item(),
                    "same_db": db_ids[j] == db_ids[i],
                    "column_alignment": alignment,
                }
            )

        # complementary signal: the SINGLE best column pair against ANY
        # other table (not just the aggregate-MaxSim top match) -- may
        # point to a DIFFERENT table than the aggregate ranking above.
        # Reported alongside, not blended in -- see single_best_column_pair
        # docstring for why this is more collision-prone.
        best_single_score = float("-inf")
        best_single_j = None
        best_single_qi = None
        best_single_dj = None
        for j in range(N):
            if j == i:
                continue
            score, qi_idx, dj_idx = single_best_column_pair(
                reprs_device[i], reprs_device[j]
            )
            if score > best_single_score:
                best_single_score = score
                best_single_j = j
                best_single_qi = qi_idx
                best_single_dj = dj_idx

        single_best_match = {
            "table_id": tables[best_single_j].table_id,
            "table_name": tables[best_single_j].table_name,
            "db_id": db_ids[best_single_j],
            "same_db": db_ids[best_single_j] == db_ids[i],
            "score": best_single_score,
            "query_column": tables[i].columns[best_single_qi].header,
            "matched_column": tables[best_single_j].columns[best_single_dj].header,
        }

        closest_matches.append(
            {
                "query_table_id": tables[i].table_id,
                "query_table_name": tables[i].table_name,
                "query_db_id": db_ids[i],
                "top_matches": matches,
                "single_best_column_pair_match": single_best_match,
            }
        )

        # print the top-1 match's column alignment to console, so it's
        # visible immediately without opening the saved JSON file
        if matches:
            top = matches[0]
            print(
                f"\n[{tables[i].table_name}] best match: "
                f"{top['table_name']} (score={top['score']:.3f}, "
                f"same_db={top['same_db']})"
            )
            for a in top["column_alignment"]:
                print(
                    f"    {a['query_column']!r} -> {a['matched_column']!r} "
                    f"(cos_sim={a['cosine_sim']:.3f})"
                )
            sb = single_best_match
            print(
                f"    [single-best-column-pair]: {sb['query_column']!r} -> "
                f"{sb['matched_column']!r} in {sb['table_name']} "
                f"(cos_sim={sb['score']:.3f}, same_db={sb['same_db']})"
            )

    avg_baseline = sum(baseline_precisions) / N

    print(f"\n== BIRD same-database retrieval eval ({N} tables, {n_dbs} databases) ==")
    print(f"random-chance baseline precision: {avg_baseline:.4f}")
    print(f"MRR (rank of first same-db match): {sum(reciprocal_ranks) / N:.4f}")
    for k in args.top_k:
        avg_p = sum(results[k]) / N
        lift = avg_p / max(avg_baseline, 1e-9)
        print(f"Precision@{k}: {avg_p:.4f}  (lift over baseline: {lift:.2f}x)")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(closest_matches, f, indent=2, ensure_ascii=False)
    print(f"\nsaved closest-table matches ({output_top_k} per table) to: {args.output_file}")