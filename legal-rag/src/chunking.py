
"""
Task 2: Document Chunking

Splits legal documents into smaller overlapping chunks
for embedding and retrieval.
"""

from pathlib import Path
import argparse
import statistics

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion import load_documents


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 50

# RecursiveCharacterTextSplitter works with characters,
# so we approximately convert tokens to characters.
CHARS_PER_TOKEN = 4

# Legal-friendly separators
SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "; ",
    ": ",
    ", ",
    " ",
    ""
]


# ---------------------------------------------------------
# Create splitter
# ---------------------------------------------------------

def create_splitter(
    chunk_size=CHUNK_SIZE_TOKENS,
    chunk_overlap=CHUNK_OVERLAP_TOKENS
):
    """
    Create a character-based text splitter.

    256 tokens ≈ 1024 characters
    50 tokens  ≈ 200 characters
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * CHARS_PER_TOKEN,
        chunk_overlap=chunk_overlap * CHARS_PER_TOKEN,
        separators=SEPARATORS,
        keep_separator=True
    )


# ---------------------------------------------------------
# Create chunks
# ---------------------------------------------------------

def create_chunks(
    documents,
    chunk_size=CHUNK_SIZE_TOKENS,
    chunk_overlap=CHUNK_OVERLAP_TOKENS
):
    """
    Split LangChain Documents into smaller chunks.

    Original metadata such as source file and page number
    is preserved.
    """

    if not documents:
        return []

    splitter = create_splitter(
        chunk_size,
        chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    # Remove extremely small chunks
    chunks = [
        chunk
        for chunk in chunks
        if len(chunk.page_content.strip()) >= 20
    ]

    # Add chunk numbers
    page_counters = {}

    for index, chunk in enumerate(chunks):

        source = chunk.metadata.get(
            "source_file",
            "unknown.pdf"
        )

        page = chunk.metadata.get(
            "page_number",
            0
        )

        # Unique key for each PDF + page
        key = (source, page)

        if key not in page_counters:
            page_counters[key] = 0

        # Overall chunk number
        chunk.metadata["chunk_index"] = index

        # Chunk number within that page
        chunk.metadata["chunk_index_in_page"] = (
            page_counters[key]
        )

        page_counters[key] += 1

    return chunks


# ---------------------------------------------------------
# Chunk statistics
# ---------------------------------------------------------

def calculate_chunk_statistics(chunks):

    if not chunks:
        return None

    lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    documents = {
        chunk.metadata.get(
            "source_file",
            "unknown.pdf"
        )
        for chunk in chunks
    }

    return {
        "total_chunks": len(chunks),

        "source_documents": len(documents),

        "min_characters": min(lengths),

        "max_characters": max(lengths),

        "average_characters": statistics.mean(lengths),

        "median_characters": statistics.median(lengths)
    }


# ---------------------------------------------------------
# Display statistics
# ---------------------------------------------------------

def display_statistics(stats):

    if stats is None:
        print("No chunks were created.")
        return

    print("\n" + "=" * 60)
    print("                 CHUNK STATISTICS")
    print("=" * 60)

    print(
        f"Total chunks          : "
        f"{stats['total_chunks']}"
    )

    print(
        f"Source documents      : "
        f"{stats['source_documents']}"
    )

    print()

    print(
        f"Minimum length        : "
        f"{stats['min_characters']} characters"
    )

    print(
        f"Maximum length        : "
        f"{stats['max_characters']} characters"
    )

    print(
        f"Average length        : "
        f"{stats['average_characters']:.1f} characters"
    )

    print(
        f"Median length         : "
        f"{stats['median_characters']:.1f} characters"
    )

    print("=" * 60)


# ---------------------------------------------------------
# Display sample chunks
# ---------------------------------------------------------

def display_sample_chunks(
    chunks,
    number_of_documents=3
):

    if not chunks:
        return

    # Store one sample chunk from each PDF
    samples = {}

    for chunk in chunks:

        source = chunk.metadata.get(
            "source_file",
            "unknown.pdf"
        )

        if source not in samples:
            samples[source] = chunk

        if len(samples) >= number_of_documents:
            break

    print("\n" + "=" * 60)
    print("                  SAMPLE CHUNKS")
    print("=" * 60)

    for chunk in samples.values():

        print(
            f"\nSource : "
            f"{chunk.metadata.get('source_file')}"
        )

        print(
            f"Page   : "
            f"{chunk.metadata.get('page_number')}"
        )

        print(
            f"Chunk  : "
            f"{chunk.metadata.get('chunk_index')}"
        )

        print(
            f"\nText:\n"
            f"{chunk.page_content[:500]}..."
        )

        print("-" * 60)


# ---------------------------------------------------------
# Display actual tokens
# ---------------------------------------------------------

def show_tokens(text):

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    tokens = tokenizer.tokenize(text)

    print("\n" + "=" * 60)
    print("                    TOKENS")
    print("=" * 60)

    print(
        f"Number of tokens: {len(tokens)}"
    )

    print("\nTokens:")

    for index, token in enumerate(tokens):
        print(f"{index}: {token}")

    print("=" * 60)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Chunk the legal document corpus."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE_TOKENS
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP_TOKENS
    )

    args = parser.parse_args()

    # ---------------------------------------------
    # Load documents
    # ---------------------------------------------

    documents = load_documents(
        args.data_dir
    )

    if not documents:

        print(
            "\nNo documents found."
            "\nPut PDFs inside data/raw/"
        )

        return

    print(
        f"\nLoaded {len(documents)} page documents."
    )

    # ---------------------------------------------
    # Create chunks
    # ---------------------------------------------

    print(
        f"\nCreating chunks..."
        f"\nChunk size   : {args.chunk_size} tokens"
        f"\nChunk overlap: {args.chunk_overlap} tokens"
    )

    chunks = create_chunks(
        documents,
        args.chunk_size,
        args.chunk_overlap
    )

    # ---------------------------------------------
    # Statistics
    # ---------------------------------------------

    stats = calculate_chunk_statistics(
        chunks
    )

    display_statistics(stats)

    # ---------------------------------------------
    # Sample chunks
    # ---------------------------------------------

    display_sample_chunks(chunks)

    # ---------------------------------------------
    # Show actual tokens from first chunk
    # ---------------------------------------------

    if chunks:
        show_tokens(
            chunks[0].page_content
        )


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    main()


