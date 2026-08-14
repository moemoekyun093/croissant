import json
import sqlite3
import random
import argparse

from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher


# ==========================================================
# CONFIG
# ==========================================================

MAX_ROWS_PER_BIRD_TABLE = 50

random.seed(42)


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
# SAMPLE VALUES FROM BIRD TABLE
# ==========================================================

def sample_table_values(
    sqlite_path,
    table_name,
    max_rows=50
):

    values = []

    try:

        conn = sqlite3.connect(
            sqlite_path
        )

        cur = conn.cursor()

        query = (
            f'SELECT * FROM "{table_name}" '
            f'LIMIT {max_rows}'
        )

        cur.execute(query)

        rows = cur.fetchall()

        for row in rows:

            for value in row:

                if value is None:
                    continue

                value = clean_text(
                    value
                )

                if value:
                    values.append(
                        value
                    )

        conn.close()

    except Exception:
        pass

    # MAX_QUERY_VALUES = 100
    # random.shuffle(values)

    # values = values[:MAX_QUERY_VALUES]

    return values


# ==========================================================
# BUILD QUERY
# ==========================================================

def build_query(
    table_info,
    db_root
):

    query_parts = []

    table_name = clean_text(
        table_info[
            "table_name"
        ]
    )

    query_parts.append(
        table_name
    )

    query_parts.extend([

        clean_text(col)

        for col in table_info[
            "column_names"
        ]
    ])

    sqlite_path = get_sqlite_path(

        db_root,

        table_info[
            "db_id"
        ]
    )

    values = sample_table_values(

        sqlite_path,

        table_info[
            "table_name_original"
        ],

        MAX_ROWS_PER_BIRD_TABLE
    )

    query_parts.extend(
        values
    )

    return " ".join(
        query_parts
    )


# ==========================================================
# RETRIEVE TABLE IDS
# ==========================================================

def retrieve_tables(

    bird_tables,

    searcher,

    db_root,

    top_per_table
):

    selected = set()

    for key, table_info in tqdm(

        bird_tables.items(),

        desc="Retrieving"
    ):

        MAX_QUERY_TERMS = 1000

        query = build_query(
            table_info,
            db_root
        )

        terms = query.split()

        if len(terms) > MAX_QUERY_TERMS:
            terms = terms[:MAX_QUERY_TERMS]

        while True:

            try:

                query = " ".join(terms)

                hits = searcher.search(
                    query,
                    k=top_per_table
                )

                break

            except Exception as e:

                print(
                    f"[WARN] Query too large "
                    f"({len(terms)} terms)"
                )

                if len(terms) < 50:
                    raise

                terms = terms[:len(terms)//2]

        for hit in hits:

            raw_doc = json.loads(

                searcher.doc(
                    hit.docid
                ).raw()
            )

            selected.add(

                raw_doc[
                    "table_id"
                ]
            )

    return selected


# ==========================================================
# STREAM CORPUS
# ==========================================================

def write_selected_tables(

    corpus_jsonl,

    selected_ids,

    output_jsonl
):

    written = 0

    with open(

        corpus_jsonl,

        "r",

        encoding="utf-8"

    ) as fin, open(

        output_jsonl,

        "w",

        encoding="utf-8"

    ) as fout:

        for line in tqdm(

            fin,

            desc="Writing output"
        ):

            doc = json.loads(
                line
            )

            if doc[
                "table_id"
            ] not in selected_ids:

                continue

            out = {

            "id":
                doc["table_id"],

            "table_name":
                doc["page_title"],

            "column_names":
                doc["column_names"],

            "contents":
                doc["contents"],

            "source":
                "webtable"
            }

            fout.write(

                json.dumps(
                    out,
                    ensure_ascii=False
                )

                + "\n"
            )

            written += 1

    return written


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--bird_json",

        required=True
    )

    parser.add_argument(

        "--bird_db_root",

        required=True
    )

    parser.add_argument(

        "--index_dir",

        required=True
    )

    parser.add_argument(

        "--corpus_jsonl",

        required=True
    )

    parser.add_argument(

        "--output_jsonl",

        required=True
    )

    parser.add_argument(

        "--top_per_table",

        type=int,

        default=10000
    )

    parser.add_argument(

        "--target_size",

        type=int,

        default=1000000
    )

    args = parser.parse_args()

    # ------------------------------------------------------
    # Load BIRD
    # ------------------------------------------------------

    with open(

        args.bird_json,

        "r",

        encoding="utf-8"

    ) as f:

        bird_tables = json.load(
            f
        )

    print(
        f"Loaded "
        f"{len(bird_tables)} "
        f"BIRD tables"
    )

    # ------------------------------------------------------
    # Lucene
    # ------------------------------------------------------

    searcher = LuceneSearcher(
        args.index_dir
    )

    # ------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------

    selected = retrieve_tables(

        bird_tables,

        searcher,

        args.bird_db_root,

        args.top_per_table
    )

    print(
        "\nRetrieved "
        f"{len(selected)} "
        "unique tables"
    )

    # ------------------------------------------------------
    # Random fill
    # ------------------------------------------------------

    if args.target_size > len(selected):

        print(
            "\nCollecting all IDs..."
        )

        all_ids = []

        with open(
            args.corpus_jsonl,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                doc = json.loads(
                    line
                )

                all_ids.append(
                    doc[
                        "table_id"
                    ]
                )

        remaining = list(
            set(all_ids)
            - selected
        )

        need = min(

            args.target_size
            - len(selected),

            len(remaining)
        )

        print(
            f"Adding "
            f"{need} "
            f"random tables"
        )

        selected.update(

            random.sample(
                remaining,
                need
            )
        )

    # ------------------------------------------------------
    # Write output
    # ------------------------------------------------------

    written = write_selected_tables(

        args.corpus_jsonl,

        selected,

        args.output_jsonl
    )

    print(
        "\nWrote "
        f"{written} "
        "tables"
    )

    print(
        "\nSaved:"
    )

    print(
        args.output_jsonl
    )


if __name__ == "__main__":
    main()