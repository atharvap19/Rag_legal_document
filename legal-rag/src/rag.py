
"""
Task 6: Retrieval-Augmented Generation with Gemini

Takes a user question, retrieves relevant legal chunks
from FAISS, and asks Gemini to generate a grounded answer.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.retrieval import retrieve
from src.vector_store import load_vector_store


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

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

def build_prompt(
    question,
    context
):

    return f"""
You are a legal document analysis assistant.

Answer the question ONLY using the provided
legal document context.

Do not invent information.

If the answer is not available in the context,
say:

"I could not find sufficient information in the
provided documents to answer this question."

Include the source of the information using
the format:

[source_file, page X, chunk Y]

Question:
{question}

Retrieved Context:
{context}
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
# Complete RAG pipeline
# ---------------------------------------------------------

def answer_question(
    question,
    index,
    metadata,
    top_k=DEFAULT_TOP_K
):

    # ---------------------------------------------
    # Retrieve relevant chunks
    # ---------------------------------------------

    results = retrieve(
        query=question,
        index=index,
        metadata=metadata,
        k=top_k
    )

    if not results:

        return (
            "I could not find sufficient information "
            "in the provided documents to answer "
            "this question."
        ), []

    # ---------------------------------------------
    # Build context
    # ---------------------------------------------

    context = build_context(
        results
    )

    # ---------------------------------------------
    # Ask Gemini
    # ---------------------------------------------

    answer = generate_answer(
        question,
        context
    )

    return answer, results


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
        "--store-dir",
        type=Path,
        default=VECTORSTORE_DIR,
        help="FAISS vector store location."
    )

    args = parser.parse_args()

    # ---------------------------------------------
    # Load FAISS
    # ---------------------------------------------

    try:

        index, metadata = load_vector_store(
            args.store_dir
        )

    except Exception as error:

        print(
            f"Error loading vector store: {error}"
        )

        return 1

    # ---------------------------------------------
    # Run RAG
    # ---------------------------------------------

    try:

        answer, results = answer_question(
            question=args.question,
            index=index,
            metadata=metadata,
            top_k=args.top_k
        )

    except Exception as error:

        print(
            f"Error running RAG: {error}"
        )

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
