"""
Task 5: Retrieval

Converts a question into an embedding and searches the
FAISS store that belongs to one document.

Each document has its own index, so a search reads that
document's index directly. When several documents are
requested, each store is searched on its own and the
results are merged by similarity.
"""

import argparse
from pathlib import Path

import numpy as np

from src.embeddings import (
    load_embedding_model,
    embed_query
)

from src.vector_store import (
    VECTORSTORE_DIR,
    document_ids,
    get_document_store,
    resolve_document_id,
    display_documents
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TOP_K = 5


# ---------------------------------------------------------
# Embed the question
# ---------------------------------------------------------

def encode_query(query):
    """Turn the question into a 2D float32 array."""

    model = load_embedding_model()

    query_embedding = embed_query(
        query,
        model=model
    )

    return np.asarray(
        query_embedding,
        dtype="float32"
    ).reshape(1, -1)


# ---------------------------------------------------------
# Retrieve from a single loaded store
# ---------------------------------------------------------

def retrieve(
    query,
    index,
    metadata,
    k=DEFAULT_TOP_K,
    doc_id=None
):
    """
    Search one document's FAISS index using the
    question embedding.
    """

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if k < 1:
        raise ValueError(
            "k must be at least 1."
        )

    if index.ntotal == 0:
        return []

    query_embedding = encode_query(query)

    # Search FAISS
    scores, positions = index.search(
        query_embedding,
        min(k, index.ntotal)
    )

    results = []

    for rank, (score, position) in enumerate(
        zip(scores[0], positions[0]),
        start=1
    ):

        if position == -1:
            continue

        chunk = metadata[int(position)]

        results.append({
            "rank": rank,

            "similarity":
                float(score),

            "document_id":
                doc_id,

            "source_file":
                chunk["source_file"],

            "page_number":
                chunk["page_number"],

            "chunk_index":
                chunk["chunk_index"],

            "text":
                chunk["text"]
        })

    return results


# ---------------------------------------------------------
# Retrieve from a document by id
# ---------------------------------------------------------

def retrieve_from_document(
    query,
    doc_id,
    k=DEFAULT_TOP_K,
    store_dir=VECTORSTORE_DIR
):
    """
    Open the store belonging to one document and search
    only that index.
    """

    index, metadata = get_document_store(
        doc_id,
        store_dir
    )

    return retrieve(
        query,
        index,
        metadata,
        k,
        doc_id=doc_id
    )


# ---------------------------------------------------------
# Retrieve across several documents
# ---------------------------------------------------------

def retrieve_from_documents(
    query,
    documents=None,
    k=DEFAULT_TOP_K,
    store_dir=VECTORSTORE_DIR
):
    """
    Search one store per document and merge the hits by
    similarity. The indexes stay separate; only the
    result lists are combined.

    documents:
        None  -> every indexed document
        str   -> a single document id or filename
        list  -> those documents
    """

    if documents is None:
        selected = document_ids(store_dir)

    elif isinstance(documents, str):
        selected = [
            resolve_document_id(documents, store_dir)
        ]

    else:
        selected = [
            resolve_document_id(name, store_dir)
            for name in documents
        ]

    if not selected:
        raise ValueError(
            "No documents have been indexed yet."
        )

    merged = []

    for doc_id in selected:

        merged.extend(
            retrieve_from_document(
                query,
                doc_id,
                k,
                store_dir
            )
        )

    # Best matches first, whichever store they came from
    merged.sort(
        key=lambda result: result["similarity"],
        reverse=True
    )

    merged = merged[:k]

    for rank, result in enumerate(
        merged,
        start=1
    ):
        result["rank"] = rank

    return merged


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

def display_results(
    query,
    results
):

    print("\n" + "=" * 70)

    print(
        f"TOP {len(results)} RESULTS"
    )

    print(
        f"Question: {query}"
    )

    print("=" * 70)

    if not results:

        print(
            "No matching chunks found."
        )

        return

    for result in results:

        print(
            f"\n[{result['rank']}] "
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Document: "
            f"{result['document_id']}"
        )

        print(
            f"Source: "
            f"{result['source_file']}"
        )

        print(
            f"Page: "
            f"{result['page_number']}"
        )

        print(
            f"Chunk: "
            f"{result['chunk_index']}"
        )

        print("\nText:")

        text = result["text"]

        print(
            text[:500]
        )

        if len(text) > 500:
            print("...")

        print("-" * 70)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Retrieve relevant legal document chunks."
    )

    parser.add_argument(
        "query",
        help="Question to search for."
    )

    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results."
    )

    parser.add_argument(
        "-d",
        "--document",
        default=None,
        help=(
            "Document id or PDF filename to search. "
            "Leave empty to search every document."
        )
    )

    parser.add_argument(
        "--store-dir",
        type=Path,
        default=VECTORSTORE_DIR
    )

    args = parser.parse_args()

    # Retrieve chunks
    try:

        results = retrieve_from_documents(
            query=args.query,
            documents=args.document,
            k=args.top_k,
            store_dir=args.store_dir
        )

    except Exception as error:

        print(
            f"Error retrieving chunks: {error}"
        )

        display_documents(args.store_dir)

        return 1

    # Display results
    display_results(
        args.query,
        results
    )

    return 0


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
