
"""
Task 1: Data Ingestion and Understanding

Loads PDF files from data/raw/ using LangChain's PyPDFLoader
and calculates basic statistics about the legal document corpus.
"""

from pathlib import Path
import argparse
import math

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOKENS_PER_WORD = 1.33


# ---------------------------------------------------------
# Basic functions
# ---------------------------------------------------------

def count_words(text):
    """Return the number of words in the text."""
    return len(text.split())


def count_approx_tokens(text):
    """Estimate tokens using approximately 1.33 tokens per word."""
    return math.ceil(count_words(text) * TOKENS_PER_WORD)


def count_exact_tokens(texts):
    """
    Count tokens using the tokenizer of all-MiniLM-L6-v2.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    encoded = tokenizer(
        texts,
        add_special_tokens=False
    )["input_ids"]

    return [len(tokens) for tokens in encoded]


# ---------------------------------------------------------
# Find PDFs
# ---------------------------------------------------------

def find_pdfs(data_dir):
    """Find all PDF files inside the data directory."""

    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return []

    pdfs = sorted(data_dir.glob("*.pdf"))

    if not pdfs:
        print(f"No PDF files found in: {data_dir}")

    return pdfs


# ---------------------------------------------------------
# Load PDFs
# ---------------------------------------------------------

def load_documents(data_dir):
    """
    Load all PDFs using LangChain's PyPDFLoader.

    PyPDFLoader creates one Document object per page.
    """

    documents = []

    for pdf in find_pdfs(data_dir):

        try:
            pages = PyPDFLoader(str(pdf)).load()

            for page in pages:

                # Store useful metadata
                page.metadata["source_file"] = pdf.name

                # Convert LangChain's 0-based page number
                # to a human-friendly 1-based page number
                page_number = page.metadata.get("page", 0)

                page.metadata["page_number"] = (
                    int(page_number) + 1
                )

            documents.extend(pages)

            print(
                f"Loaded {pdf.name} "
                f"({len(pages)} pages)"
            )

        except Exception as error:
            print(
                f"Could not read {pdf.name}: {error}"
            )

    return documents


# ---------------------------------------------------------
# Calculate statistics
# ---------------------------------------------------------

def calculate_statistics(documents, exact_tokens=False):

    if not documents:
        return None

    # Get text from every page
    texts = [
        document.page_content
        for document in documents
    ]

    # Token counts
    if exact_tokens:
        token_counts = count_exact_tokens(texts)
        token_method = "Exact MiniLM tokenizer"
    else:
        token_counts = [
            count_approx_tokens(text)
            for text in texts
        ]
        token_method = "Approximation (words × 1.33)"

    # Group statistics by PDF
    document_stats = {}

    for document, tokens in zip(
        documents,
        token_counts
    ):

        filename = document.metadata["source_file"]

        if filename not in document_stats:
            document_stats[filename] = {
                "pages": 0,
                "words": 0,
                "characters": 0,
                "tokens": 0
            }

        document_stats[filename]["pages"] += 1

        document_stats[filename]["words"] += (
            count_words(document.page_content)
        )

        document_stats[filename]["characters"] += (
            len(document.page_content)
        )

        document_stats[filename]["tokens"] += tokens

    # Total number of PDFs
    total_documents = len(document_stats)

    # Total number of pages
    total_pages = len(documents)

    # Total words
    total_words = sum(
        stats["words"]
        for stats in document_stats.values()
    )

    # Total characters
    total_characters = sum(
        stats["characters"]
        for stats in document_stats.values()
    )

    # Total tokens
    total_tokens = sum(
        stats["tokens"]
        for stats in document_stats.values()
    )

    # Average values
    average_words = (
        total_words / total_documents
    )

    average_tokens = (
        total_tokens / total_documents
    )

    average_pages = (
        total_pages / total_documents
    )

    # Shortest and longest document
    shortest = min(
        document_stats.items(),
        key=lambda item: item[1]["words"]
    )

    longest = max(
        document_stats.items(),
        key=lambda item: item[1]["words"]
    )

    return {
        "total_documents": total_documents,
        "total_pages": total_pages,
        "total_words": total_words,
        "total_characters": total_characters,
        "total_tokens": total_tokens,
        "average_words": average_words,
        "average_tokens": average_tokens,
        "average_pages": average_pages,
        "shortest": shortest,
        "longest": longest,
        "token_method": token_method,
        "document_stats": document_stats
    }


# ---------------------------------------------------------
# Display statistics
# ---------------------------------------------------------

def display_statistics(stats):

    if stats is None:
        print("No documents were loaded.")
        return

    print("\n" + "=" * 60)
    print("             LEGAL CORPUS STATISTICS")
    print("=" * 60)

    print(
        f"Original PDF documents : "
        f"{stats['total_documents']}"
    )

    print(
        f"Loaded page documents  : "
        f"{stats['total_pages']}"
    )

    print(
        f"Total words            : "
        f"{stats['total_words']:,}"
    )

    print(
        f"Total characters       : "
        f"{stats['total_characters']:,}"
    )

    print(
        f"Total tokens           : "
        f"{stats['total_tokens']:,}"
    )

    print(
        f"Token method           : "
        f"{stats['token_method']}"
    )

    print()

    print(
        f"Average pages/document : "
        f"{stats['average_pages']:.2f}"
    )

    print(
        f"Average words/document : "
        f"{stats['average_words']:,.2f}"
    )

    print(
        f"Average tokens/document: "
        f"{stats['average_tokens']:,.2f}"
    )

    print()

    shortest_name, shortest_data = stats["shortest"]
    longest_name, longest_data = stats["longest"]

    print(
        f"Shortest document      : "
        f"{shortest_name} "
        f"({shortest_data['words']:,} words)"
    )

    print(
        f"Longest document       : "
        f"{longest_name} "
        f"({longest_data['words']:,} words)"
    )

    print("\n" + "-" * 60)
    print(
        f"{'DOCUMENT':<30}"
        f"{'PAGES':>8}"
        f"{'WORDS':>12}"
        f"{'TOKENS':>12}"
    )
    print("-" * 60)

    for filename, data in stats["document_stats"].items():

        name = filename

        if len(name) > 28:
            name = name[:25] + "..."

        print(
            f"{name:<30}"
            f"{data['pages']:>8}"
            f"{data['words']:>12,}"
            f"{data['tokens']:>12,}"
        )

    print("-" * 60)
    print("=" * 60)


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Load and analyze legal PDF documents."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Folder containing legal PDFs."
    )

    parser.add_argument(
        "--tokenizer",
        action="store_true",
        help="Use the actual MiniLM tokenizer."
    )

    args = parser.parse_args()

    print(
        f"\nReading PDFs from: {args.data_dir}\n"
    )

    # Load documents
    documents = load_documents(args.data_dir)

    if not documents:
        print(
            "\nNo documents loaded. "
            "Place PDF files inside data/raw/"
        )
        return

    # Calculate statistics
    statistics = calculate_statistics(
        documents,
        exact_tokens=args.tokenizer
    )

    # Display statistics
    display_statistics(statistics)

    # Show sample document
    sample = documents[0]

    print("\n" + "=" * 60)
    print("                 SAMPLE PAGE")
    print("=" * 60)

    print(
        f"Source : "
        f"{sample.metadata.get('source_file')}"
    )

    print(
        f"Page   : "
        f"{sample.metadata.get('page_number')}"
    )

    preview = " ".join(
        sample.page_content.split()
    )[:500]

    print(f"\nText:\n{preview}...")


if __name__ == "__main__":
    main()

