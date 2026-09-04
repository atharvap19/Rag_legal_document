
"""
Task 6: Retrieval-Augmented Generation with Gemini

Takes a user question, retrieves relevant legal chunks
from the vector store of the chosen document, and asks
Gemini to generate a grounded answer.

Every document has its own FAISS index, so a question
about one contract only ever reads that contract's store.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.retrieval import (
    retrieve,
    retrieve_from_documents
)
from src.vector_store import (
    VECTORSTORE_DIR,
    display_documents
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GEMINI_MODEL = "gemini-3.5-flash-lite"

DEFAULT_TOP_K = 5


# ---------------------------------------------------------
# Load Gemini API key
# ---------------------------------------------------------

def load_api_key():

    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env"
        )

    return api_key


# ---------------------------------------------------------
# Resolve Gemini model name
# ---------------------------------------------------------

def get_model_name():

    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    return os.getenv(
        "GEMINI_MODEL",
        GEMINI_MODEL
    )


# ---------------------------------------------------------
# Build context
# ---------------------------------------------------------

def build_context(results):

    context = []

    for result in results:

        text = result["text"]

        source = result["source_file"]

        page = result["page_number"]

        chunk = result["chunk_index"]

        context.append(
            f"[{source}, page {page}, chunk {chunk}]\n"
            f"{text}"
        )

    return "\n\n---\n\n".join(context)


# ---------------------------------------------------------
# Build Gemini prompt
# ---------------------------------------------------------

def build_prompt(question, context):
    return f"""
You are a legal document question-answering assistant.

Answer the question using ONLY the information contained in the
RETRIEVED CONTEXT below.

Rules:
- Use only information explicitly supported by the retrieved context.
- Do not use outside knowledge or invent information.
- Examine ALL retrieved chunks and use every chunk that is relevant.
- You may combine information from multiple relevant chunks to form
  one complete answer.
- Ignore chunks that are unrelated to the question.
- If the answer is only partially supported, provide only the supported
  information and clearly state what cannot be determined.
- Do not reject the question simply because the information is spread
  across multiple chunks.
- Cite each important statement using:
  [source_file, page X, chunk Y]
- Only cite chunks that actually support the statement.
- Do not mention similarity scores, embeddings, retrieval, or ranking.

If the retrieved context genuinely does not contain enough information
to answer the question, respond exactly with:

"I could not find sufficient information in the provided documents to answer this question."

Question:
{question}

Retrieved Context:
{context}

Answer:
"""


# ---------------------------------------------------------
# Generate Gemini answer
# ---------------------------------------------------------

def generate_answer(
    question,
    context
):

    api_key = load_api_key()

    client = genai.Client(
        api_key=api_key
    )

    prompt = build_prompt(
        question,
        context
    )

    response = client.models.generate_content(
        model=get_model_name(),
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024
        )
    )

    answer = (
        response.text or ""
    ).strip()

    if not answer:

        raise ValueError(
            "Gemini returned an empty response."
        )

    return answer


# ---------------------------------------------------------
# Turn retrieved chunks into an answer
# ---------------------------------------------------------

NO_ANSWER = (
    "I could not find sufficient information "
    "in the provided documents to answer "
    "this question."
)


def answer_from_results(
    question,
    results
):
    """Build the context and ask Gemini."""

    if not results:
        return NO_ANSWER, []

    context = build_context(
        results
    )

    answer = generate_answer(
        question,
        context
    )

    return answer, results


# ---------------------------------------------------------
# Complete RAG pipeline
# ---------------------------------------------------------

def answer_question(
    question,
    documents=None,
    top_k=DEFAULT_TOP_K,
    store_dir=VECTORSTORE_DIR
):
    """
    Answer a question against the vector store of one
    document.

    documents:
        None  -> every indexed document, each searched
                 in its own store
        str   -> a single document id or PDF filename
        list  -> those documents
    """

    results = retrieve_from_documents(
        query=question,
        documents=documents,
        k=top_k,
        store_dir=store_dir
    )

    return answer_from_results(
        question,
        results
    )


def answer_with_store(
    question,
    index,
    metadata,
    top_k=DEFAULT_TOP_K,
    doc_id=None
):
    """
    Answer a question using a store that is already
    loaded in memory.
    """

    results = retrieve(
        query=question,
        index=index,
        metadata=metadata,
        k=top_k,
        doc_id=doc_id
    )

    return answer_from_results(
        question,
        results
    )


# ---------------------------------------------------------
# Display answer
# ---------------------------------------------------------

def display_answer(
    question,
    answer,
    results
):

    print("\n" + "=" * 70)

    print(
        "                 RAG ANSWER"
    )

    print("=" * 70)

    print(
        f"\nQuestion:\n{question}"
    )

    print(
        f"\nAnswer:\n{answer}"
    )

    # ---------------------------------------------
    # Sources
    # ---------------------------------------------

    if results:

        print(
            "\n" + "-" * 70
        )

        print(
            "SOURCES"
        )

        print(
            "-" * 70
        )

        for result in results:

            print(
                f"[{result['rank']}] "
                f"{result['source_file']}, "
                f"page {result['page_number']}, "
                f"chunk {result['chunk_index']} "
                f"(similarity: "
                f"{result['similarity']:.4f})"
            )

    print(
        "\n" + "=" * 70
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Ask questions about legal documents."
    )

    parser.add_argument(
        "question",
        help="Question to ask."
    )

    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve."
    )

    parser.add_argument(
        "-d",
        "--document",
        default=None,
        help=(
            "Document id or PDF filename to ask about. "
            "Leave empty to use every document."
        )
    )

    parser.add_argument(
        "--store-dir",
        type=Path,
        default=VECTORSTORE_DIR,
        help="Folder holding the per-document stores."
    )

    args = parser.parse_args()

    # ---------------------------------------------
    # Run RAG
    # ---------------------------------------------

    try:

        answer, results = answer_question(
            question=args.question,
            documents=args.document,
            top_k=args.top_k,
            store_dir=args.store_dir
        )

    except Exception as error:

        print(
            f"Error running RAG: {error}"
        )

        display_documents(args.store_dir)

        return 1

    # ---------------------------------------------
    # Display
    # ---------------------------------------------

    display_answer(
        args.question,
        answer,
        results
    )

    return 0


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
