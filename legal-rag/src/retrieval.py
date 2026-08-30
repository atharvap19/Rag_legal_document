"""
Task 5: Retrieval

Converts a question into an embedding and searches
the FAISS vector store for the most relevant chunks.
"""

import argparse
from pathlib import Path

import numpy as np

from src.embeddings import (
    load_embedding_model,
    embed_query
)

from src.vector_store import load_vector_store


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

DEFAULT_TOP_K = 5


# ---------------------------------------------------------
# Retrieve relevant chunks
# ---------------------------------------------------------

def retrieve(
    query,
    index,
    metadata,
    k=DEFAULT_TOP_K
):
    """
    Search FAISS using the question embedding.
    """

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if k < 1:
        raise ValueError(
            "k must be at least 1."
        )

    # Load embedding model
    model = load_embedding_model()

    # Convert question to embedding
    query_embedding = embed_query(
        query,
        model=model
    )

    # FAISS needs a 2D NumPy array
    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    ).reshape(1, -1)

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
        "--store-dir",
        type=Path,
        default=VECTORSTORE_DIR
    )

    args = parser.parse_args()

    # Load FAISS index and metadata
    try:

        index, metadata = load_vector_store(
            args.store_dir
        )

    except Exception as error:

        print(
            f"Error loading vector store: {error}"
        )

        return 1

    # Retrieve chunks
    results = retrieve(
        query=args.query,
        index=index,
        metadata=metadata,
        k=args.top_k
    )

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