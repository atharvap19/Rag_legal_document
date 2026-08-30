
"""
Main application for the Legal Document RAG system.

Run from the project root:

    python main.py

The application will:

1. Check whether the FAISS vector store exists.
2. Build it automatically if it does not exist.
3. Load the vector store.
4. Start the FastAPI server.

Open:
    http://127.0.0.1:8000/docs   (API documentation)
    http://127.0.0.1:8000/ui     (simple frontend)
"""

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.vector_store import (
    build_vector_store,
    load_vector_store
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


# ---------------------------------------------------------
# Vector store
# ---------------------------------------------------------

index = None
metadata = None


def setup_vector_store():

    global index
    global metadata

    index_file = VECTORSTORE_DIR / "index.faiss"
    metadata_file = VECTORSTORE_DIR / "metadata.json"

    # -----------------------------------------------------
    # If vector store already exists
    # -----------------------------------------------------

    if index_file.exists() and metadata_file.exists():

        print("\nVector store already exists.")

        index, metadata = load_vector_store(
            VECTORSTORE_DIR
        )

        print(
            f"Loaded {index.ntotal} vectors."
        )

        return


    # -----------------------------------------------------
    # Otherwise build it
    # -----------------------------------------------------

    print("\nVector store not found.")

    print("Building vector store...")

    index, metadata = build_vector_store(
        data_dir=DATA_DIR,
        store_dir=VECTORSTORE_DIR,
        chunk_size=256,
        chunk_overlap=50
    )

    print("\nVector store created successfully.")

    print(
        f"Vectors indexed: {index.ntotal}"
    )


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app):

    # Only load if it was not already prepared by __main__
    if index is None:

        try:

            setup_vector_store()

        except Exception as error:

            print(
                f"Vector store unavailable at startup: "
                f"{error}"
            )

    yield


# ---------------------------------------------------------
# Create FastAPI app
# ---------------------------------------------------------

app = FastAPI(
    title="Legal Document RAG",
    description=(
        "Question answering over legal PDF documents.\n\n"
        "Upload a contract, answer the 12 benchmark "
        "questions, and view the RAG evaluation scores."
    ),
    version="1.0",
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

    return {
        "status": "running",

        "vector_store_loaded":
            index is not None,

        "vectors":
            index.ntotal if index else 0
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
    Save a PDF into data/raw/ and rebuild the FAISS
    index from every PDF in that folder.

    Uploading a file that already exists replaces it.
    Nothing else in data/raw/ is deleted.
    """

    global index
    global metadata

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

    # Rebuild the index over the whole corpus
    try:

        index, metadata = build_vector_store(
            data_dir=DATA_DIR,
            store_dir=VECTORSTORE_DIR,
            chunk_size=256,
            chunk_overlap=50
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {error}"
        )

    documents = sorted({
        record["source_file"]
        for record in metadata
    })

    return {
        "uploaded": filename,

        "documents_indexed": documents,

        "chunks": len(metadata),

        "vectors": index.ntotal
    }


# ---------------------------------------------------------
# Ask question
# ---------------------------------------------------------

@app.post("/ask")
def ask(request: QuestionRequest):

    if index is None:

        return {
            "error": "Vector store is not loaded."
        }


    if not request.question.strip():

        return {
            "error": "Question cannot be empty."
        }


    try:

        # Run RAG
        answer, results = answer_question(
            question=request.question,
            index=index,
            metadata=metadata,
            top_k=request.top_k
        )


        # Format sources
        sources = []

        for result in results:

            sources.append({

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
    Answer every benchmark question against the
    currently indexed document and score the answers
    with ROUGE-L plus the Gemini judge.

    A full 12-question run with the judge enabled makes
    24 Gemini calls and takes a couple of minutes.
    """

    if index is None:

        raise HTTPException(
            status_code=409,
            detail="Vector store is not loaded."
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
                index,
                metadata,
                request.top_k,
                request.use_judge
            )
        )

    documents = sorted({
        record["source_file"]
        for record in metadata
    })

    return {

        "documents": documents,

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


    # Build/load FAISS
    try:

        setup_vector_store()

    except Exception as error:

        # An empty corpus is a valid starting point now
        # that documents can be uploaded through the API.
        print(
            f"\nNo vector store yet: {error}"
        )

        print(
            "Upload a PDF at /ui to build one."
        )


    print("\nStarting FastAPI server...")

    print(
        "Frontend: http://127.0.0.1:8000/ui"
    )

    print(
        "API docs: http://127.0.0.1:8000/docs"
    )

    print("=" * 60)


    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
