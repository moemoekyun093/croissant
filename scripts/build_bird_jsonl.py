import json
import sqlite3
import argparse
import random
import datetime
from collections import defaultdict, deque

from tqdm import tqdm


# ==========================================================
# CONFIG
# ==========================================================

START_ID = 90266199


# ==========================================================
# CLEAN
# ==========================================================

def clean_text(x):

    if x is None:
        return ""

    x = str(x)

    x = " ".join(x.split())

    return x.strip()


# ==========================================================
# SQLITE PATH
# ==========================================================

def get_sqlite_path(
    db_root,
    db_id
):

    return (
        f"{db_root}/"
        f"{db_id}/"
        f"{db_id}.sqlite"
    )


# ==========================================================
# NULL / MISSING-VALUE HANDLING
# ==========================================================

def random_value_for_type(declared_type):
    """
    Generates a random placeholder value appropriate to a column's
    DECLARED SQL type (from PRAGMA table_info), for the rare case where
    an entire column is null within the sampled rows -- there's nothing
    real anywhere in the sample to impute from, so this falls back to
    the column's schema type instead of its (nonexistent) observed data.

    NOTE: for TEXT-typed columns, "a random value of the right type" is
    genuinely underspecified -- there's no real content anywhere to
    infer format/vocabulary from, and random characters would likely be
    meaningless gibberish, arguably WORSE than leaving it null (masking
    says "no signal, ignore this"; gibberish text asserts a false
    signal the model has no way to recognize as meaningless). TEXT
    columns are deliberately left as None here, still handled by
    CellEncoder's own null-masking (cell_mask).
    """

    t = (declared_type or "").upper()

    if "INT" in t:
        return random.randint(0, 100000)

    if any(x in t for x in ["REAL", "FLOA", "DOUB", "NUMERIC", "DECIMAL"]):
        return round(random.uniform(0, 10000), 2)

    if "DATE" in t or "TIME" in t:
        start = datetime.date(2000, 1, 1)
        end = datetime.date(2024, 12, 31)
        delta_days = (end - start).days
        return (start + datetime.timedelta(days=random.randint(0, delta_days))).isoformat()

    return None  # TEXT/CHAR/BLOB/unknown -- see docstring


def _is_missing(value):
    """
    Treats both true SQL NULL and empty/whitespace-only strings as
    "missing" -- SQL's IS NOT NULL / COUNT(col) only catch the former;
    an empty string '' is a legitimate, present, non-NULL value in SQL,
    but semantically identical to NULL for our purposes (both produce
    no real content). This matches how the rest of the pipeline already
    treats them (CellEncoder classifies "" as an empty cell, masked out
    the same way a null would be via cell_mask) -- so imputation needs
    the same definition, or these cells silently slip through untouched.
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def fetch_nonnull_pools(cur, table_name, column_names, pool_size=2000):
    """
    For each column, queries up to pool_size of that column's own
    REAL (non-missing) values from the WHOLE table -- not just whatever
    ended up in the max_rows sample used for the corpus's displayed
    content, and not just SQL-NULL-excluded (empty strings are also
    excluded -- see _is_missing).

    Decouples "how many rows do we show" from "where do we draw
    imputation values from": a column that's merely sparse (rather than
    genuinely all-missing) can easily have ZERO real values within an
    unlucky sample window even though the full table has plenty
    elsewhere -- this queries the real, table-wide pool instead.
    """

    pools = []
    for col in column_names:
        cur.execute(
            f'SELECT "{col}" FROM "{table_name}" '
            f'WHERE "{col}" IS NOT NULL AND TRIM("{col}") != \'\' '
            f'LIMIT {pool_size}'
        )
        pools.append([row[0] for row in cur.fetchall()])
    return pools


def impute_nulls_deterministic(raw_rows, declared_types, nonnull_pools):
    """
    For each column, cycles through that column's own real-value pool
    -- drawn from the WHOLE table via fetch_nonnull_pools, not just this
    sample -- to fill missing cells (NULL or empty string -- see
    _is_missing), deterministically (no randomness/seed needed for this
    part). Falls back to a random value matching the column's declared
    SQL type only if the pool itself is empty (genuinely no real values
    anywhere in the table for that column) -- a DIFFERENT random draw
    per row, reusing one fixed value for every row would just recreate
    a single-constant-value collapse problem under a different guise.
    """

    if not raw_rows:
        return raw_rows

    n_cols = len(raw_rows[0])
    fill_counters = [0] * n_cols
    imputed_rows = []

    for row in raw_rows:
        new_row = list(row)
        for c in range(n_cols):
            if _is_missing(new_row[c]):
                pool = nonnull_pools[c]
                if pool:
                    new_row[c] = pool[fill_counters[c] % len(pool)]
                    fill_counters[c] += 1
                else:
                    new_row[c] = random_value_for_type(declared_types[c])
        imputed_rows.append(new_row)

    return imputed_rows


# ==========================================================
# FOREIGN KEYS / ALIGNED SAMPLING
# ==========================================================
# Rows sampled independently per table can't show any real cross-table
# relationship: a fact table's random 50 rows reference one set of
# entities, another table's own independent random 50 rows reference a
# totally different set -- essentially zero overlap. To fix this
# WITHOUT ever computing a full join (which would be as expensive as
# the base table, even with ORDER BY RANDOM() LIMIT -- SQLite still has
# to evaluate every joined row before it can pick the top N), sampling
# happens in two safe, bounded steps per database:
#   1. Every table gets its own random base sample first (cheap).
#   2. Any table that's REFERENCED by another table's foreign key gets
#      its sample REPLACED with a lookup of exactly the entities the
#      referencing table's base sample pointed at (bounded: at most
#      max_rows distinct values per referencing table, regardless of
#      how large the referenced table actually is).
# This keeps each table's own columns intact (no column merging) while
# making the ENTITIES shown in each table's sample genuinely overlap.

def get_foreign_keys(cur, table_name):
    """
    Returns this table's OUTGOING foreign keys as (from_col, to_table,
    to_col) tuples, auto-discovered from the schema's declared FOREIGN
    KEY constraints -- no hardcoded per-table logic needed.
    """
    cur.execute(f'PRAGMA foreign_key_list("{table_name}")')
    fks = []
    for row in cur.fetchall():
        # columns: id, seq, table, from, to, on_update, on_delete, match
        to_table, from_col, to_col = row[2], row[3], row[4]
        fks.append((from_col, to_table, to_col))
    return fks


def sample_base_rows(cur, table_name, max_rows):
    """
    Random sample of a table's own rows -- genuinely deterministic
    (given random.seed() was called), unlike SQL's own RANDOM(): SQLite
    seeds ORDER BY RANDOM() from system entropy internally, completely
    unaffected by Python's random.seed() -- confirmed directly (two
    "same seed" runs produced different sampled rows before this fix).

    Instead: reads just the table's rowids (a cheap single-column scan,
    trivial even for a huge table -- e.g. ~3MB of integers for 383K
    rows), uses Python's OWN seeded random.sample() to pick which rows
    to use, THEN fetches only those specific rows via a bounded
    WHERE rowid IN (...) lookup. Same cost profile as before, genuinely
    reproducible now.
    """
    cur.execute(f'SELECT rowid FROM "{table_name}"')
    all_rowids = [r[0] for r in cur.fetchall()]

    if not all_rowids:
        return []

    if len(all_rowids) <= max_rows:
        chosen_rowids = all_rowids
    else:
        chosen_rowids = random.sample(all_rowids, max_rows)

    placeholders = ",".join("?" * len(chosen_rowids))
    cur.execute(
        f'SELECT * FROM "{table_name}" WHERE rowid IN ({placeholders})', chosen_rowids
    )
    return cur.fetchall()


def topological_sort_tables(table_names, table_fks):
    """
    Orders tables so that for every FK relationship (child references
    parent), the CHILD is processed -- and its sample fully finalized
    -- BEFORE the parent. Required for chains (A->B->C): B's aligned
    sample (derived from A) must be FINALIZED before it's used to
    derive C's alignment -- otherwise C ends up aligned against B's
    original, pre-alignment sample, which may share nothing with what
    B's corpus entry actually shows (confirmed directly: without this,
    a 3-table chain's 2-hop alignment came back at 0/50 overlap, versus
    50/50 for the direct 1-hop relationship).

    Uses Kahn's algorithm. Self-references are ignored. If a cycle is
    detected (FKs forming a loop -- unusual but not forbidden by
    SQLite), the involved tables are appended in arbitrary order at the
    end with a warning, rather than crashing; they fall back to
    whatever alignment is available from already-processed tables.
    """

    edges = defaultdict(set)  # child -> set of parents (deduplicated)
    for child, fks in table_fks.items():
        if child not in table_names:
            continue
        for from_col, to_table, to_col in fks:
            if to_table in table_names and to_table != child:
                edges[child].add(to_table)

    in_degree = {t: 0 for t in table_names}
    for child, parents in edges.items():
        for parent in parents:
            in_degree[parent] += 1

    queue = deque([t for t in table_names if in_degree[t] == 0])
    order = []
    visited = set()

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for parent in edges.get(node, []):
            in_degree[parent] -= 1
            if in_degree[parent] == 0:
                queue.append(parent)

    remaining = [t for t in table_names if t not in visited]
    if remaining:
        print(f"  [WARN] cyclic foreign-key relationship among: {remaining} -- processing in arbitrary order")
        order.extend(remaining)

    return order


def compute_aligned_rows(cur, parent_table, parent_pk_col, referenced_values, max_rows):
    """
    Given a specific set of values (collected from OTHER tables' base
    samples), fetches up to max_rows of parent_table's rows matching
    those values -- a bounded WHERE...IN lookup, never a full scan or
    join, regardless of parent_table's actual size.
    """
    if not referenced_values:
        return []

    values_to_use = list(referenced_values)[:max_rows]
    placeholders = ",".join("?" * len(values_to_use))

    try:
        cur.execute(
            f'SELECT * FROM "{parent_table}" WHERE "{parent_pk_col}" IN ({placeholders})',
            values_to_use,
        )
        return cur.fetchall()
    except Exception as e:
        print(f"  [WARN] aligned lookup failed for {parent_table}.{parent_pk_col}: {e}")
        return []


# ==========================================================
# BUILD CONTENTS (matches build_row_major_corpus.py's format)
# ==========================================================

def build_contents(
    table_name,
    column_names,
    rows
):
    """
    Same structured format as the webtable corpus's create_document():
    "[TABLE]\\n{name}\\n\\n[SCHEMA]\\n{col1} | {col2}\\n\\n[ROWS]\\n{row1}\\n{row2}..."
    so table_from_corpus_record() can parse BIRD tables the same way it
    parses webtables -- no separate SQLite re-query needed downstream.
    """

    parts = [
        "[TABLE]",
        table_name,
        "\n[SCHEMA]",
        " | ".join(column_names),
        "\n[ROWS]",
    ]

    for row in rows:

        parts.append(
            " | ".join(row)
        )

    return "\n".join(parts)


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--bird_json", required=True)
    parser.add_argument("--bird_db_root", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_rows", type=int, default=50)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for the all-null-column random-value fallback -- "
        "same seed gives byte-identical corpus output across regenerations"
    )
    parser.add_argument(
        "--align_related_tables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="for tables referenced by another table's foreign key, "
        "replace their random sample with a lookup of exactly the "
        "entities the referencing table's sample points at -- so the "
        "SAME entities appear in both tables' corpus entries, without "
        "merging columns or computing a full join (always bounded, "
        "regardless of table size). ON by default -- pass "
        "--no-align_related_tables to fall back to independent random "
        "sampling per table."
    )

    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.bird_json, "r", encoding="utf-8") as f:
        bird_tables = json.load(f)

    print(f"Loaded {len(bird_tables)} BIRD tables")

    # group by database -- alignment only makes sense within one db
    tables_by_db = defaultdict(list)
    for key, table_info in bird_tables.items():
        tables_by_db[table_info["db_id"]].append(table_info)

    next_id = START_ID

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:

        for db_id, table_infos in tqdm(tables_by_db.items(), desc="Processing databases"):

            sqlite_path = get_sqlite_path(args.bird_db_root, db_id)

            try:
                conn = sqlite3.connect(sqlite_path)
                cur = conn.cursor()
            except Exception as e:
                print(f"[ERROR] could not open {sqlite_path}: {e}")
                continue

            # -- pass 1: schema + foreign keys for every table (no
            #    sampling yet -- sampling happens in topological order
            #    below, so alignment can use FINALIZED parent/child
            #    samples rather than raw pre-alignment ones) --
            table_schema = {}       # table_name_original -> (declared_types, schema_column_names)
            table_fks = {}          # table_name_original -> [(from_col, to_table, to_col), ...]

            for table_info in table_infos:
                t_name = table_info["table_name_original"]
                try:
                    cur.execute(f'PRAGMA table_info("{t_name}")')
                    schema_info = cur.fetchall()
                    declared_types = [c[2] for c in schema_info]
                    schema_column_names = [c[1] for c in schema_info]
                    table_schema[t_name] = (declared_types, schema_column_names)
                    table_fks[t_name] = get_foreign_keys(cur, t_name)
                except Exception as e:
                    print(f"[ERROR] {t_name}: {e}")
                    table_schema[t_name] = ([], [])
                    table_fks[t_name] = []

            table_names = list(table_schema.keys())

            incoming_refs = defaultdict(list)  # to_table -> [(from_table, from_col, to_col), ...]
            for t_name, fks in table_fks.items():
                for from_col, to_table, to_col in fks:
                    if to_table in table_schema:
                        incoming_refs[to_table].append((t_name, from_col, to_col))

            # -- pass 2: sample EVERY table, in topological order, so
            #    a parent's alignment always uses its children's
            #    ALREADY-FINALIZED samples (guaranteed available -- the
            #    topological order processes children strictly before
            #    the parents that depend on them). Tables with no
            #    incoming references just get an ordinary base sample.
            final_rows = {}

            if args.align_related_tables:
                process_order = topological_sort_tables(table_names, table_fks)
            else:
                process_order = table_names  # order doesn't matter, no alignment happening

            for t_name in process_order:
                referencing = incoming_refs.get(t_name) if args.align_related_tables else None

                if not referencing:
                    final_rows[t_name] = sample_base_rows(cur, t_name, args.max_rows)
                    continue

                # aligned sample: use REFERENCING tables' FINAL rows
                # (already computed -- they're children, processed earlier
                # in topological order)
                referenced_values = set()
                to_col = None
                for from_table, from_col, this_to_col in referencing:
                    to_col = this_to_col
                    _, from_schema_cols = table_schema.get(from_table, ([], []))
                    if from_col not in from_schema_cols:
                        continue
                    from_idx = from_schema_cols.index(from_col)
                    for row in final_rows.get(from_table, []):
                        if row[from_idx] is not None:
                            referenced_values.add(row[from_idx])

                if not referenced_values or to_col is None:
                    final_rows[t_name] = sample_base_rows(cur, t_name, args.max_rows)
                    continue

                aligned = compute_aligned_rows(
                    cur, t_name, to_col, referenced_values, args.max_rows
                )
                final_rows[t_name] = aligned if aligned else sample_base_rows(
                    cur, t_name, args.max_rows
                )

            # -- pass 3: impute + write each table's document --
            for table_info in table_infos:
                t_name = table_info["table_name_original"]

                table_name = (
                    clean_text(table_info["db_id"])
                    + "#sep#"
                    + clean_text(t_name)
                )

                column_names = [clean_text(col) for col in table_info["column_names"]]

                declared_types, schema_column_names = table_schema.get(t_name, ([], []))
                raw_rows = final_rows.get(t_name, [])

                nonnull_pools = fetch_nonnull_pools(
                    cur, t_name, schema_column_names, pool_size=2000
                ) if schema_column_names else []

                imputed_rows = impute_nulls_deterministic(
                    raw_rows, declared_types, nonnull_pools
                ) if nonnull_pools else raw_rows

                rows = [[clean_text(v) for v in row] for row in imputed_rows]

                contents = build_contents(table_name, column_names, rows)

                doc = {
                    "id": str(next_id),
                    "table_name": table_name,
                    "column_names": column_names,
                    "contents": contents,
                    "source": "bird"
                }

                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                next_id += 1

            conn.close()

    print("\nSaved to:")
    print(args.output_jsonl)


if __name__ == "__main__":
    main()