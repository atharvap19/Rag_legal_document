
"""
FastAPI backend for the Legal RAG application.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag import answer_question
from src.vector_store import load_vector_store


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------

app = FastAPI(
    title="Legal RAG API",
    description="Legal Document Analysis using RAG",
    version="1.0"
)


# ---------------------------------------------------------
# Allow frontend requests
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ---------------------------------------------------------
# Load FAISS
# ---------------------------------------------------------

try:

    index, metadata = load_vector_store(
        VECTORSTORE_DIR
    )

    print(
        f"Loaded FAISS vector store "
        f"with {index.ntotal} vectors."
    )

except Exception as error:

    index = None
    metadata = None

    print(
        f"Could not load vector store: {error}"
    )


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class QuestionRequest(BaseModel):

    question: str

    top_k: int = 5


# ---------------------------------------------------------
# Home
# ---------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Legal RAG API is running"
    }


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "running",
        "vector_store_loaded": index is not None
    }


# ---------------------------------------------------------
# Ask question
# ---------------------------------------------------------

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    if index is None:

        return {
            "error": "Vector store is not loaded."
        }

    if not request.question.strip():

        return {
            "error": "Question cannot be empty."
        }

    try:

        answer, results = answer_question(
            question=request.question,
            index=index,
            metadata=metadata,
            top_k=request.top_k
        )

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

