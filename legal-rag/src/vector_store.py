
"""
Task 4: FAISS Vector Store

Creates a FAISS index from document chunks and their embeddings.
The index and chunk metadata are saved inside vectorstore/.
"""

from pathlib import Path
import argparse
import json
from datetime import datetime, timezone

import faiss
import numpy as np

from src.ingestion import load_documents
from src.chunking import create_chunks
from src.embeddings import (
    load_embedding_model,
    generate_embeddings,
    embed_query
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"


# ---------------------------------------------------------
# Create FAISS vector store
# ---------------------------------------------------------

def create_vector_store(chunks):

    if not chunks:
        raise ValueError("No chunks available.")

    # Load embedding model
    model = load_embedding_model()

    # Extract text
    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    print(
        f"\nCreating embeddings for "
        f"{len(texts)} chunks..."
    )

    # Generate embeddings
    embeddings = generate_embeddings(
        texts,
        model=model
    )

    # Get vector dimension
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatIP(
        dimension
    )

    # Add embeddings to FAISS
    index.add(embeddings)

    print(
        f"FAISS index created with "
        f"{index.ntotal} vectors."
    )

    # Save metadata
    metadata = []

    for i, chunk in enumerate(chunks):

        metadata.append({
            "chunk_index": i,

            "source_file": chunk.metadata.get(
                "source_file",
                "unknown.pdf"
            ),

            "page_number": chunk.metadata.get(
                "page_number",
                0
            ),

            "chunk_index_in_page": chunk.metadata.get(
                "chunk_index_in_page",
                0
            ),

            "text": chunk.page_content
        })

    return index, metadata, dimension


# ---------------------------------------------------------
# Save vector store
# ---------------------------------------------------------

def save_vector_store(
    index,
    metadata,
    dimension,
    directory=VECTORSTORE_DIR
):

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save FAISS index
    index_path = directory / INDEX_FILE

    faiss.write_index(
        index,
        str(index_path)
    )

    # Save metadata
    metadata_path = directory / METADATA_FILE

    data = {
        "embedding_model":
            "sentence-transformers/all-MiniLM-L6-v2",

        "embedding_dimension":
            dimension,

        "chunk_count":
            len(metadata),

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "records":
            metadata
    }

    metadata_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        f"\nSaved FAISS index    : "
        f"{index_path}"
    )

    print(
        f"Saved metadata       : "
        f"{metadata_path}"
    )


# ---------------------------------------------------------
# Load vector store
# ---------------------------------------------------------

def load_vector_store(
    directory=VECTORSTORE_DIR
):

    index_path = directory / INDEX_FILE
    metadata_path = directory / METADATA_FILE

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    # Load FAISS
    index = faiss.read_index(
        str(index_path)
    )

    # Load metadata
    data = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    metadata = data["records"]

    return index, metadata


# ---------------------------------------------------------
# Search vector store
# ---------------------------------------------------------

def search_vector_store(
    query,
    index,
    metadata,
    top_k=5
):

    # Load embedding model
    model = load_embedding_model()

    # Convert query into embedding
    query_embedding = embed_query(
        query,
        model=model
    )

    # FAISS expects a 2D array
    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    ).reshape(1, -1)

    # Search FAISS
    scores, positions = index.search(
        query_embedding,
        min(top_k, len(metadata))
    )

    results = []

    for score, position in zip(
        scores[0],
        positions[0]
    ):

        if position == -1:
            continue

        results.append({
            "score": float(score),
            "chunk": metadata[int(position)]
        })

    return results


# ---------------------------------------------------------
# Build complete vector store
# ---------------------------------------------------------

def build_vector_store(
    data_dir=DATA_DIR,
    store_dir=VECTORSTORE_DIR,
    chunk_size=256,
    chunk_overlap=50
):

    # Load PDFs
    documents = load_documents(
        data_dir
    )

    if not documents:
        raise ValueError(
            "No PDF documents found."
        )

    print(
        f"\nLoaded {len(documents)} page documents."
    )

    # Create chunks
    chunks = create_chunks(
        documents,
        chunk_size,
        chunk_overlap
    )

    if not chunks:
        raise ValueError(
            "No chunks were created."
        )

    print(
        f"Created {len(chunks)} chunks."
    )

    # Create FAISS index
    index, metadata, dimension = (
        create_vector_store(chunks)
    )

    # Save everything
    save_vector_store(
        index,
        metadata,
        dimension,
        store_dir
    )

    return index, metadata


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Build a FAISS vector store."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR
    )

    parser.add_argument(
        "--store-dir",
        type=Path,
        default=VECTORSTORE_DIR
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50
    )

    parser.add_argument(
        "--rebuild",
        action="store_true"
    )

    args = parser.parse_args()

    index_path = (
        args.store_dir / INDEX_FILE
    )

    metadata_path = (
        args.store_dir / METADATA_FILE
    )

    # Check existing store
    if (
        index_path.exists()
        and metadata_path.exists()
        and not args.rebuild
    ):

        index, metadata = load_vector_store(
            args.store_dir
        )

        print(
            f"\nFAISS store already exists."
        )

        print(
            f"Vectors: {index.ntotal}"
        )

        print(
            "Use --rebuild to create it again."
        )

        return

    # Build store
    index, metadata = build_vector_store(
        args.data_dir,
        args.store_dir,
        args.chunk_size,
        args.chunk_overlap
    )

    # Final information
    print("\n" + "=" * 60)
    print("             VECTOR STORE READY")
    print("=" * 60)

    print(
        f"Vectors indexed : "
        f"{index.ntotal}"
    )

    print(
        f"Embedding size  : "
        f"{index.d}"
    )

    print(
        f"Chunks stored   : "
        f"{len(metadata)}"
    )

    print(
        f"Location        : "
        f"{args.store_dir}"
    )

    print("=" * 60)


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
