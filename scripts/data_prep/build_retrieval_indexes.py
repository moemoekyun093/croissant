import os
import json
import argparse
import subprocess

from tqdm import tqdm


# ==========================================================
# BUILD LUCENE INDEX
# ==========================================================

def build_lucene_index(
    corpus_dir,
    index_dir,
    threads=16
):

    os.makedirs(
        index_dir,
        exist_ok=True
    )

    cmd = [
        "python",
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        corpus_dir,
        "--index",
        index_dir,
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw"
    ]

    print("\nBuilding Lucene index...\n")

    subprocess.run(
        cmd,
        check=True
    )

    print(
        "\nLucene indexing complete"
    )


# ==========================================================
# LOAD CORPUS
# ==========================================================

def load_corpus(corpus_jsonl):

    ids = []
    texts = []

    with open(
        corpus_jsonl,
        "r",
        encoding="utf-8"
    ) as f:

        for line in tqdm(
            f,
            desc="Loading corpus"
        ):

            doc = json.loads(
                line
            )

            ids.append(
                str(doc["id"])
            )

            texts.append(
                doc["contents"]
            )

    return ids, texts


# ==========================================================
# BUILD COLBERT INDEX
# ==========================================================

def build_colbert_index(
    corpus_jsonl,
    index_dir,
    model_name,
    batch_size
):

    from pylate import indexes
    from pylate import models

    os.makedirs(
        index_dir,
        exist_ok=True
    )

    print(
        "\nLoading corpus..."
    )

    document_ids, documents = load_corpus(
        corpus_jsonl
    )

    print(
        f"\nLoaded {len(documents)} documents"
    )

    print(
        "\nLoading ColBERT model..."
    )

    model = models.ColBERT(
        model_name_or_path=model_name
    )

    print(
        "\nEncoding documents..."
    )

    document_embeddings = model.encode(
        documents,
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=True
    )

    print(
        "\nCreating PLAID index..."
    )

    index = indexes.PLAID(
        index_folder=index_dir,
        index_name="index",
        override=True
    )

    index.add_documents(
        documents_ids=document_ids,
        documents_embeddings=document_embeddings
    )

    print(
        "\nColBERT indexing complete"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus_jsonl",
        required=True
    )

    parser.add_argument(
        "--lucene_dir",
        required=True
    )

    parser.add_argument(
        "--colbert_dir",
        required=True
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=16
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--model_name",
        default="lightonai/GTE-ModernColBERT-v1"
    )

    args = parser.parse_args()

    # ------------------------------------------------------
    # Lucene
    # ------------------------------------------------------

    corpus_dir = os.path.dirname(
        os.path.abspath(
            args.corpus_jsonl
        )
    )

    build_lucene_index(
        corpus_dir=corpus_dir,
        index_dir=args.lucene_dir,
        threads=args.threads
    )

    # ------------------------------------------------------
    # ColBERT
    # ------------------------------------------------------

    build_colbert_index(
        corpus_jsonl=args.corpus_jsonl,
        index_dir=args.colbert_dir,
        model_name=args.model_name,
        batch_size=args.batch_size
    )

    print("\nDONE")


if __name__ == "__main__":
    main()