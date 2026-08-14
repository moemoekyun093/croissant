import os
import json
import gzip
import random
import argparse

from pathlib import Path
from tqdm import tqdm


# ==========================================================
# CONFIG
# ==========================================================

MAX_ROWS = 50

random.seed(42)


# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_text(x):

    if x is None:
        return ""

    x = str(x)

    x = " ".join(x.split())

    return x.strip()


# ==========================================================
# EXTRACT COLUMNS WITH CONSISTENT ROW SAMPLING
# ==========================================================

def extract_columns(table_json):

    relation = table_json.get(
        "relation",
        []
    )

    if not relation:
        return []

    max_rows = 0

    for col in relation:

        if isinstance(col, list):

            max_rows = max(
                max_rows,
                max(
                    0,
                    len(col) - 1
                )
            )

    if max_rows <= MAX_ROWS:

        selected_rows = list(
            range(max_rows)
        )

    else:

        selected_rows = sorted(
            random.sample(
                range(max_rows),
                MAX_ROWS
            )
        )

    columns = []

    for col in relation:

        if not isinstance(col, list):
            continue

        if len(col) == 0:
            continue

        header = clean_text(
            col[0]
        )

        values = []

        for row_idx in selected_rows:

            value_idx = row_idx + 1

            if value_idx >= len(col):
                continue

            value = clean_text(
                col[value_idx]
            )

            values.append(
                value
            )

        columns.append(
            {
                "header": header,
                "values": values
            }
        )

    return columns


# ==========================================================
# CREATE TABLE DOCUMENT
# ==========================================================

def create_document(
    table_json,
    table_id
):

    page_title = clean_text(
        table_json.get(
            "pageTitle",
            ""
        )
    )

    columns = extract_columns(
        table_json
    )

    if len(columns) == 0:

        return {

            "id": str(table_id),

            "table_id":
                str(table_id),

            "page_title":
                page_title,

            "column_names": [],

            "num_columns": 0,

            "contents":
                page_title
        }

    column_names = [

        col["header"]

        for col in columns
    ]

    n_rows = min(

        len(col["values"])

        for col in columns
    )

    contents_parts = []

    contents_parts.append(
        "[TABLE]"
    )

    contents_parts.append(
        page_title
    )

    contents_parts.append(
        "\n[SCHEMA]"
    )

    contents_parts.append(
        " | ".join(
            column_names
        )
    )

    contents_parts.append(
        "\n[ROWS]"
    )

    for r in range(n_rows):

        row = []

        for col in columns:

            row.append(
                col["values"][r]
            )

        contents_parts.append(
            " | ".join(row)
        )

    contents = "\n".join(
        contents_parts
    )

    doc = {

        "id":
            str(table_id),

        "table_id":
            str(table_id),

        "page_title":
            page_title,

        "column_names":
            column_names,

        "num_columns":
            len(column_names),

        "contents":
            contents
    }

    return doc


# ==========================================================
# PROCESS ONE JSON.GZ FILE
# ==========================================================

def process_json_gz(
    json_gz_path,
    writer,
    start_table_id
):

    table_id = start_table_id

    try:

        with gzip.open(
            json_gz_path,
            "rt",
            encoding="utf-8",
            errors="replace"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:

                    table_json = json.loads(
                        line
                    )

                    doc = create_document(
                        table_json,
                        table_id
                    )

                    writer.write(
                        json.dumps(
                            doc,
                            ensure_ascii=False
                        )
                        + "\n"
                    )

                    table_id += 1

                except Exception as e:

                    print(
                        f"[LINE ERROR] "
                        f"{json_gz_path}"
                    )

                    print(e)

    except Exception as e:

        print(
            f"[FILE ERROR] "
            f"{json_gz_path}"
        )

        print(e)

    return table_id


# ==========================================================
# BUILD CORPUS
# ==========================================================

def build_corpus(
    input_dir,
    corpus_dir
):

    os.makedirs(
        corpus_dir,
        exist_ok=True
    )

    corpus_path = os.path.join(
        corpus_dir,
        "corpus.jsonl"
    )

    json_gz_files = sorted(
        Path(input_dir).rglob(
            "*.json.gz"
        )
    )

    print(
        f"\nFound "
        f"{len(json_gz_files)} "
        f"json.gz files"
    )

    table_id = 0

    with open(
        corpus_path,
        "w",
        encoding="utf-8"
    ) as writer:

        for path in tqdm(
            json_gz_files,
            desc="Processing"
        ):

            table_id = process_json_gz(
                path,
                writer,
                table_id
            )

    print("\nCorpus complete")

    print(
        f"Total tables: "
        f"{table_id}"
    )

    print(
        f"Saved to: "
        f"{corpus_path}"
    )

    return corpus_path


# ==========================================================
# BUILD LUCENE INDEX
# ==========================================================

def build_lucene_index(
    corpus_dir,
    index_dir
):

    os.makedirs(
        index_dir,
        exist_ok=True
    )

    cmd = f"""
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input {corpus_dir} \
  --index {index_dir} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions \
  --storeDocvectors \
  --storeRaw
"""

    print(
        "\nBuilding Lucene index...\n"
    )

    os.system(cmd)

    print(
        "\nLucene indexing complete"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        required=True
    )

    parser.add_argument(
        "--corpus_dir",
        default="processed_corpus"
    )

    parser.add_argument(
        "--index_dir",
        default="lucene_index"
    )

    args = parser.parse_args()

    build_corpus(
        args.input_dir,
        args.corpus_dir
    )

    build_lucene_index(
        args.corpus_dir,
        args.index_dir
    )

    print("\nDONE")


if __name__ == "__main__":
    main()

