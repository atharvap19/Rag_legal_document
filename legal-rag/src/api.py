
"""
FastAPI backend for the Legal RAG application.

Each document has its own FAISS index and chunks file
inside vectorstore/<document_id>/, so a question is
answered from the store of the document it names.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag import answer_question
from src.vector_store import (
    VECTORSTORE_DIR,
    list_documents,
    resolve_document_id
)


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------

app = FastAPI(
    title="Legal RAG API",
    description="Legal Document Analysis using RAG",
    version="2.0"
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
# Indexed documents
# ---------------------------------------------------------

def indexed_documents():

    return list_documents(VECTORSTORE_DIR)


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
# Request model
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

    documents = indexed_documents()

    return {
        "status": "running",

        "documents_indexed": len(documents),

        "vectors": sum(
            entry["chunk_count"]
            for entry in documents
        )
    }


# ---------------------------------------------------------
# Indexed documents
# ---------------------------------------------------------

@app.get("/documents")
def get_documents():

    documents = indexed_documents()

    return {
        "count": len(documents),
        "documents": documents
    }


# ---------------------------------------------------------
# Ask question
# ---------------------------------------------------------

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    if not indexed_documents():

        return {
            "error": "No documents indexed yet."
        }

    if not request.question.strip():

        return {
            "error": "Question cannot be empty."
        }

    doc_id = resolve_requested_document(
        request.document
    )

    try:

        answer, results = answer_question(
            question=request.question,
            documents=doc_id,
            top_k=request.top_k,
            store_dir=VECTORSTORE_DIR
        )

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
