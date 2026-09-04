"""Legal Document RAG system.

This package contains the individual building blocks of the pipeline:

    ingestion    -> load PDFs from disk into LangChain Documents
    chunking     -> split those Documents into retrieval-sized chunks
    embeddings   -> turn text into vectors with a sentence-transformers model
    vector_store -> build / save / load one FAISS store per document
    retrieval    -> find the chunks most similar to a question inside a
                    single document's store
    rag          -> ask Gemini to answer using only the retrieved chunks
    evaluation   -> score the pipeline against reference answers

Path constants live here so that every module resolves files relative to the
project root rather than to the current working directory. This keeps the
project runnable from anywhere without hard-coding absolute paths.
"""

from pathlib import Path

# src/__init__.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

__all__ = [
    "PROJECT_ROOT",
    "DATA_RAW_DIR",
    "VECTORSTORE_DIR",
    "EVALUATION_DIR",
    "FRONTEND_DIR",
]
