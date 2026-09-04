
"""
Task 4: FAISS Vector Store (one store per document)

Every PDF gets its own FAISS index and its own chunks file:

    vectorstore/
        registry.json
        <document_id>/
            index.faiss
            chunks.json

Documents are never merged into a shared index. Retrieval
opens the store of the document being asked about and
searches only that index.
"""

from pathlib import Path
import argparse
import json
import re
import shutil
from datetime import datetime, timezone

import faiss
import numpy as np

from src.ingestion import find_pdfs, load_pdf
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
CHUNKS_FILE = "chunks.json"
REGISTRY_FILE = "registry.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Document identity
# ---------------------------------------------------------

def document_id(source_file):
    """
    Turn a PDF filename into a safe folder name.

    "Master Services Agreement.pdf"
        becomes
    "master_services_agreement"
    """

    stem = Path(str(source_file)).stem

    slug = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        stem
    ).strip("_").lower()

    return slug or "document"


def document_dir(
    doc_id,
    store_dir=VECTORSTORE_DIR
):
    """Folder holding one document's index and chunks."""

    return Path(store_dir) / doc_id


def store_exists(
    doc_id,
    store_dir=VECTORSTORE_DIR
):
    """True when the document already has a built store."""

    directory = document_dir(doc_id, store_dir)

    return (
        (directory / INDEX_FILE).exists()
        and (directory / CHUNKS_FILE).exists()
    )


# ---------------------------------------------------------
# Build the FAISS index for one document
# ---------------------------------------------------------

def create_vector_store(chunks):
    """
    Embed the chunks of a single document and build
    its FAISS index.
    """

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

    # Chunk records. chunk_index is the position inside
    # this document's own index, so metadata[position]
    # always lines up with a FAISS search result.
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
# Save one document's store
# ---------------------------------------------------------

def save_vector_store(
    index,
    metadata,
    dimension,
    directory,
    doc_id=None,
    source_file=None
):
    """
    Write index.faiss and chunks.json into the document's
    own folder.
    """

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    if doc_id is None:
        doc_id = directory.name

    if source_file is None and metadata:
        source_file = metadata[0]["source_file"]

    # Save FAISS index
    index_path = directory / INDEX_FILE

    faiss.write_index(
        index,
        str(index_path)
    )

    # Save chunks
    chunks_path = directory / CHUNKS_FILE

    data = {
        "document_id":
            doc_id,

        "source_file":
            source_file,

        "embedding_model":
            EMBEDDING_MODEL,

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

    chunks_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        f"Saved FAISS index    : {index_path}"
    )

    print(
        f"Saved chunks         : {chunks_path}"
    )

    return data


# ---------------------------------------------------------
# Load one document's store
# ---------------------------------------------------------

def load_vector_store(directory):
    """
    Load index.faiss and chunks.json from a single
    document folder.
    """

    directory = Path(directory)

    index_path = directory / INDEX_FILE
    chunks_path = directory / CHUNKS_FILE

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}"
        )

    # Load FAISS
    index = faiss.read_index(
        str(index_path)
    )

    # Load chunks
    data = json.loads(
        chunks_path.read_text(
            encoding="utf-8"
        )
    )

    metadata = data["records"]

    return index, metadata


def load_document_store(
    doc_id,
    store_dir=VECTORSTORE_DIR
):
    """Load a document's store by its id."""

    directory = document_dir(doc_id, store_dir)

    if not directory.exists():
        raise FileNotFoundError(
            f"No vector store for document: {doc_id}"
        )

    return load_vector_store(directory)


# ---------------------------------------------------------
# Cached access
# ---------------------------------------------------------

_STORE_CACHE = {}


def get_document_store(
    doc_id,
    store_dir=VECTORSTORE_DIR
):
    """
    Load a document's store, reusing the copy already in
    memory unless the files on disk have changed.
    """

    directory = document_dir(doc_id, store_dir)

    index_path = directory / INDEX_FILE
    chunks_path = directory / CHUNKS_FILE

    if not (
        index_path.exists()
        and chunks_path.exists()
    ):
        raise FileNotFoundError(
            f"No vector store for document: {doc_id}"
        )

    key = (str(Path(store_dir).resolve()), doc_id)

    stamp = (
        index_path.stat().st_mtime_ns,
        chunks_path.stat().st_mtime_ns
    )

    cached = _STORE_CACHE.get(key)

    if cached and cached[0] == stamp:
        return cached[1], cached[2]

    index, metadata = load_vector_store(directory)

    _STORE_CACHE[key] = (stamp, index, metadata)

    return index, metadata


def clear_store_cache():
    """Forget every store held in memory."""

    _STORE_CACHE.clear()


# ---------------------------------------------------------
# Registry
# ---------------------------------------------------------

def registry_path(store_dir=VECTORSTORE_DIR):

    return Path(store_dir) / REGISTRY_FILE


def load_registry(store_dir=VECTORSTORE_DIR):
    """
    Read registry.json, the list of documents that have
    their own store.
    """

    path = registry_path(store_dir)

    if not path.exists():
        return {"documents": []}

    try:

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

    except json.JSONDecodeError:

        return {"documents": []}

    if "documents" not in data:
        data["documents"] = []

    return data


def save_registry(
    registry,
    store_dir=VECTORSTORE_DIR
):

    store_dir = Path(store_dir)

    store_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    registry["embedding_model"] = EMBEDDING_MODEL

    registry["updated_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    registry_path(store_dir).write_text(
        json.dumps(
            registry,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


def update_registry(
    entry,
    store_dir=VECTORSTORE_DIR
):
    """Insert or replace one document entry."""

    registry = load_registry(store_dir)

    documents = [
        item
        for item in registry["documents"]
        if item["document_id"] != entry["document_id"]
    ]

    documents.append(entry)

    registry["documents"] = sorted(
        documents,
        key=lambda item: item["document_id"]
    )

    save_registry(registry, store_dir)

    return registry


def list_documents(store_dir=VECTORSTORE_DIR):
    """
    Every document that currently has a usable store.

    The registry is filtered against what is really on
    disk, so a removed folder disappears from the list.
    """

    registry = load_registry(store_dir)

    return [
        entry
        for entry in registry["documents"]
        if store_exists(
            entry["document_id"],
            store_dir
        )
    ]


def document_ids(store_dir=VECTORSTORE_DIR):

    return [
        entry["document_id"]
        for entry in list_documents(store_dir)
    ]


def resolve_document_id(
    name,
    store_dir=VECTORSTORE_DIR
):
    """
    Accept a document id or a source filename and return
    the matching document id.
    """

    if not name:
        raise ValueError("No document given.")

    entries = list_documents(store_dir)

    for entry in entries:

        if entry["document_id"] == name:
            return entry["document_id"]

    for entry in entries:

        if entry["source_file"] == name:
            return entry["document_id"]

    # Fall back to slugifying whatever was passed in
    doc_id = document_id(name)

    if store_exists(doc_id, store_dir):
        return doc_id

    available = ", ".join(
        entry["document_id"]
        for entry in entries
    ) or "none"

    raise ValueError(
        f"Unknown document: {name}. "
        f"Available: {available}"
    )


def delete_document_store(
    doc_id,
    store_dir=VECTORSTORE_DIR
):
    """Remove one document's folder and registry entry."""

    directory = document_dir(doc_id, store_dir)

    if directory.exists():
        shutil.rmtree(directory)

    registry = load_registry(store_dir)

    registry["documents"] = [
        item
        for item in registry["documents"]
        if item["document_id"] != doc_id
    ]

    save_registry(registry, store_dir)

    clear_store_cache()


# ---------------------------------------------------------
# Build the store for one PDF
# ---------------------------------------------------------

def build_document_store(
    pdf_path,
    store_dir=VECTORSTORE_DIR,
    chunk_size=256,
    chunk_overlap=50
):
    """
    Chunk, embed and index a single PDF into its own
    FAISS store. Other documents are left untouched.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    doc_id = document_id(pdf_path.name)

    print(
        f"\nIndexing {pdf_path.name} as '{doc_id}'"
    )

    # Load only this PDF
    pages = load_pdf(pdf_path)

    if not pages:
        raise ValueError(
            f"No readable pages in {pdf_path.name}"
        )

    # Create chunks
    chunks = create_chunks(
        pages,
        chunk_size,
        chunk_overlap
    )

    if not chunks:
        raise ValueError(
            f"No chunks were created for {pdf_path.name}"
        )

    print(
        f"Loaded {len(pages)} pages, "
        f"created {len(chunks)} chunks."
    )

    # Build this document's own index
    index, metadata, dimension = (
        create_vector_store(chunks)
    )

    # Save into the document's own folder
    data = save_vector_store(
        index,
        metadata,
        dimension,
        document_dir(doc_id, store_dir),
        doc_id=doc_id,
        source_file=pdf_path.name
    )

    entry = {
        "document_id":
            doc_id,

        "source_file":
            pdf_path.name,

        "page_count":
            len(pages),

        "chunk_count":
            len(metadata),

        "embedding_dimension":
            dimension,

        "created_at":
            data["created_at"]
    }

    update_registry(entry, store_dir)

    clear_store_cache()

    return entry, index, metadata


# ---------------------------------------------------------
# Build a store for every PDF
# ---------------------------------------------------------

def build_all_stores(
    data_dir=DATA_DIR,
    store_dir=VECTORSTORE_DIR,
    chunk_size=256,
    chunk_overlap=50,
    rebuild=False
):
    """
    Give every PDF in data_dir its own vector store.

    Documents that already have one are skipped unless
    a rebuild is requested.
    """

    pdfs = find_pdfs(Path(data_dir))

    if not pdfs:
        raise ValueError(
            "No PDF documents found."
        )

    entries = []

    for pdf in pdfs:

        doc_id = document_id(pdf.name)

        if (
            not rebuild
            and store_exists(doc_id, store_dir)
        ):

            registry = load_registry(store_dir)

            existing = next(
                (
                    item
                    for item in registry["documents"]
                    if item["document_id"] == doc_id
                ),
                None
            )

            if existing:

                print(
                    f"\nSkipping {pdf.name} "
                    f"(store already exists)."
                )

                entries.append(existing)

                continue

        entry, _, _ = build_document_store(
            pdf,
            store_dir,
            chunk_size,
            chunk_overlap
        )

        entries.append(entry)

    return entries


# ---------------------------------------------------------
# Search one document's store
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
# Display the registry
# ---------------------------------------------------------

def display_documents(store_dir=VECTORSTORE_DIR):

    entries = list_documents(store_dir)

    print("\n" + "=" * 70)
    print("                 INDEXED DOCUMENTS")
    print("=" * 70)

    if not entries:

        print("No documents indexed yet.")

        print("=" * 70)

        return

    header = (
        "DOCUMENT ID".ljust(32)
        + "PAGES".rjust(8)
        + "CHUNKS".rjust(10)
    )

    print(header)

    print("-" * 70)

    for entry in entries:

        pages = str(entry.get("page_count", 0))

        chunks = str(entry.get("chunk_count", 0))

        print(
            entry["document_id"].ljust(32)
            + pages.rjust(8)
            + chunks.rjust(10)
        )

    print("-" * 70)

    print(
        f"{len(entries)} separate vector stores in "
        f"{store_dir}"
    )

    print("=" * 70)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build one FAISS vector store per document."
        )
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
        "--pdf",
        type=Path,
        default=None,
        help="Index a single PDF instead of the folder."
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
        action="store_true",
        help="Rebuild stores that already exist."
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Show the indexed documents and exit."
    )

    args = parser.parse_args()

    # Just show what is indexed
    if args.list:

        display_documents(args.store_dir)

        return

    # Single PDF
    if args.pdf:

        build_document_store(
            args.pdf,
            args.store_dir,
            args.chunk_size,
            args.chunk_overlap
        )

        display_documents(args.store_dir)

        return

    # Whole folder, one store per PDF
    build_all_stores(
        args.data_dir,
        args.store_dir,
        args.chunk_size,
        args.chunk_overlap,
        rebuild=args.rebuild
    )

    display_documents(args.store_dir)


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
