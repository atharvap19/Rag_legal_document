"""
Main application for the Legal Document RAG system.

Run from the project root:

    python main.py

Every uploaded PDF gets its own FAISS index and its own
chunks file inside vectorstore/<document_id>/. Uploading a
new contract only builds that contract's store; nothing
else is re-indexed. A question is answered from the store
of the document it names.

The application will:

1. Give every PDF in data/raw/ its own vector store.
2. Start the FastAPI server.

Open:
    http://127.0.0.1:8001/docs   (API documentation)
    http://127.0.0.1:8001/ui     (simple frontend)
"""

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.vector_store import (
    build_all_stores,
    build_document_store,
    delete_document_store,
    display_documents,
    list_documents,
    resolve_document_id
)
from src.rag import answer_question
from src.evaluation import (
    load_questions,
    evaluate_question,
    calculate_average
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

FRONTEND_DIR = PROJECT_ROOT / "frontend"

CHUNK_SIZE = 256

CHUNK_OVERLAP = 50

# Port 8000 is left to the sop-checker app, which mounts its
# UI at the root path and would otherwise clash here.
HOST = "127.0.0.1"

PORT = 8001


# ---------------------------------------------------------
# Vector stores
# ---------------------------------------------------------

def setup_vector_stores():
    """
    Make sure every PDF in data/raw/ has its own store.

    Documents that are already indexed are left alone, so
    startup only does work for new files.
    """

    entries = build_all_stores(
        data_dir=DATA_DIR,
        store_dir=VECTORSTORE_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    display_documents(VECTORSTORE_DIR)

    return entries


def indexed_documents():
    """The documents that currently have a store."""

    return list_documents(VECTORSTORE_DIR)


def require_documents():
    """Fail cleanly when nothing has been indexed yet."""

    documents = indexed_documents()

    if not documents:

        raise HTTPException(
            status_code=409,
            detail=(
                "No documents indexed yet. "
                "Upload a PDF first."
            )
        )

    return documents


def resolve_requested_document(name):
    """
    Turn the document sent by the client into a document
    id, or None to use every document.
    """

    if not name:
        return None

    try:

        return resolve_document_id(
            name,
            VECTORSTORE_DIR
        )

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app):

    try:

        setup_vector_stores()

    except Exception as error:

        print(
            f"No vector stores at startup: {error}"
        )

        print(
            "Upload a PDF at /ui to build one."
        )

    yield


# ---------------------------------------------------------
# Create FastAPI app
# ---------------------------------------------------------

app = FastAPI(
    title="Legal Document RAG",
    description=(
        "Question answering over legal PDF documents.\n\n"
        "Each document is stored in its own FAISS index "
        "with its own chunks file. Upload a contract, "
        "answer the 12 benchmark questions, and view the "
        "RAG evaluation scores."
    ),
    version="2.0",
    lifespan=lifespan
)


# ---------------------------------------------------------
# Simple frontend
# ---------------------------------------------------------

if FRONTEND_DIR.exists():

    app.mount(
        "/ui",
        StaticFiles(
            directory=FRONTEND_DIR,
            html=True
        ),
        name="ui"
    )


# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------

class QuestionRequest(BaseModel):

    question: str

    top_k: int = 5

    document: str | None = Field(
        None,
        description=(
            "Document id or PDF filename to search. "
            "Leave empty to search every document."
        )
    )


class EvaluateRequest(BaseModel):

    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Chunks retrieved per question."
    )

    limit: int | None = Field(
        None,
        ge=1,
        description=(
            "Only evaluate the first N questions. "
            "Leave empty to run all of them."
        )
    )

    use_judge: bool = Field(
        True,
        description=(
            "Score faithfulness and answer relevance "
            "with the Gemini judge. Turning this off "
            "roughly halves the run time and returns "
            "ROUGE-L only."
        )
    )

    document: str | None = Field(
        None,
        description=(
            "Document id or PDF filename to evaluate. "
            "Leave empty to use every document."
        )
    )


# ---------------------------------------------------------
# Home
# ---------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Legal Document RAG API is running"
    }


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health():

    documents = indexed_documents()

    return {
        "status": "running",

        "documents_indexed":
            len(documents),

        "vectors":
            sum(
                entry["chunk_count"]
                for entry in documents
            )
    }


# ---------------------------------------------------------
# Indexed documents
# ---------------------------------------------------------

@app.get("/documents")
def get_documents():
    """
    Every indexed document, each with its own FAISS index
    and chunks file.
    """

    documents = indexed_documents()

    return {
        "count": len(documents),
        "documents": documents
    }


# ---------------------------------------------------------
# Benchmark questions
# ---------------------------------------------------------

@app.get("/questions")
def get_questions():
    """
    The 12 benchmark questions and their reference
    answers, as used by the evaluation run.
    """

    try:

        questions = load_questions()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    return {
        "count": len(questions),
        "questions": questions
    }


# ---------------------------------------------------------
# Upload a document
# ---------------------------------------------------------

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(
        ...,
        description="A PDF contract to index."
    )
):
    """
    Save a PDF into data/raw/ and build a vector store
    just for that file.

    Uploading a file that already exists replaces it and
    rebuilds only its store. Every other document keeps
    its own index untouched.
    """

    filename = Path(file.filename or "").name

    if not filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = DATA_DIR / filename

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    destination.write_bytes(contents)

    # Index this document on its own
    try:

        entry, index, metadata = build_document_store(
            destination,
            store_dir=VECTORSTORE_DIR,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {error}"
        )

    return {
        "uploaded": filename,

        "document": entry,

        "chunks": entry["chunk_count"],

        "vectors": index.ntotal,

        "documents_indexed": [
            item["document_id"]
            for item in indexed_documents()
        ]
    }


# ---------------------------------------------------------
# Remove a document
# ---------------------------------------------------------

@app.delete("/documents/{document}")
def remove_document(document: str):
    """
    Delete one document's vector store and chunks file.

    Because stores are separate, nothing else has to be
    rebuilt.
    """

    doc_id = resolve_requested_document(document)

    delete_document_store(
        doc_id,
        VECTORSTORE_DIR
    )

    return {
        "deleted": doc_id,

        "documents_indexed": [
            item["document_id"]
            for item in indexed_documents()
        ]
    }


# ---------------------------------------------------------
# Ask question
# ---------------------------------------------------------

@app.post("/ask")
def ask(request: QuestionRequest):

    require_documents()

    if not request.question.strip():

        return {
            "error": "Question cannot be empty."
        }

    doc_id = resolve_requested_document(
        request.document
    )

    try:

        # Run RAG against that document's own store
        answer, results = answer_question(
            question=request.question,
            documents=doc_id,
            top_k=request.top_k,
            store_dir=VECTORSTORE_DIR
        )

        # Format sources
        sources = []

        for result in results:

            sources.append({

                "document_id":
                    result["document_id"],

                "source_file":
                    result["source_file"],

                "page":
                    result["page_number"],

                "chunk":
                    result["chunk_index"],

                "similarity":
                    round(
                        result["similarity"],
                        4
                    )
            })

        return {

            "question":
                request.question,

            "document":
                doc_id,

            "answer":
                answer,

            "sources":
                sources
        }

    except Exception as error:

        return {
            "error": str(error)
        }


# ---------------------------------------------------------
# Run the evaluation
# ---------------------------------------------------------

@app.post("/evaluate")
def evaluate(request: EvaluateRequest):
    """
    Answer every benchmark question against the chosen
    document and score the answers with ROUGE-L plus the
    Gemini judge.

    A full 12-question run with the judge enabled makes
    24 Gemini calls and takes a couple of minutes.
    """

    documents = require_documents()

    doc_id = resolve_requested_document(
        request.document
    )

    try:

        questions = load_questions()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    if request.limit:

        questions = questions[:request.limit]

    results = []

    for item in questions:

        results.append(
            evaluate_question(
                item["question"],
                item["reference_answer"],
                doc_id,
                request.top_k,
                request.use_judge,
                store_dir=VECTORSTORE_DIR
            )
        )

    if doc_id:

        evaluated = [doc_id]

    else:

        evaluated = [
            entry["document_id"]
            for entry in documents
        ]

    return {

        "documents": evaluated,

        "question_count": len(results),

        "top_k": request.top_k,

        "judge_enabled": request.use_judge,

        "averages": {

            "rouge_l":
                calculate_average(results, "rouge_l"),

            "faithfulness":
                calculate_average(results, "faithfulness"),

            "answer_relevance":
                calculate_average(
                    results,
                    "answer_relevance"
                )
        },

        "results": results
    }


# ---------------------------------------------------------
# Start application
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)

    print(
        "       LEGAL DOCUMENT RAG SYSTEM"
    )

    print("=" * 60)

    print("\nStarting FastAPI server...")

    print(
        f"Frontend: http://{HOST}:{PORT}/ui"
    )

    print(
        f"API docs: http://{HOST}:{PORT}/docs"
    )

    print("=" * 60)

    uvicorn.run(
        app,
        host=HOST,
        port=PORT
    )
