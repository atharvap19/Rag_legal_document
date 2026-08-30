
"""
Task 3: Embedding Generation

Uses Sentence Transformers to convert text chunks
into numerical embedding vectors.
"""

import argparse
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def load_embedding_model(model_name=MODEL_NAME):
    """
    Load the Sentence Transformer model.

    The model is cached so it is loaded only once.
    """

    print(f"Loading embedding model: {model_name}")

    return SentenceTransformer(model_name)


# ---------------------------------------------------------
# Get embedding dimension
# ---------------------------------------------------------

def get_embedding_dimension(model=None):
    """
    Return the size of each embedding vector.
    """

    if model is None:
        model = load_embedding_model()

    return model.get_sentence_embedding_dimension()


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

def generate_embeddings(
    texts,
    model=None,
    batch_size=BATCH_SIZE
):
    """
    Convert a list of texts into embedding vectors.

    Returns:
        NumPy array with shape:

        (number_of_texts, embedding_dimension)
    """

    if not texts:
        dimension = get_embedding_dimension(model)

        return np.zeros(
            (0, dimension),
            dtype="float32"
        )

    if model is None:
        model = load_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return np.asarray(
        embeddings,
        dtype="float32"
    )


# ---------------------------------------------------------
# Generate query embedding
# ---------------------------------------------------------

def embed_query(
    query,
    model=None
):
    """
    Generate an embedding for a single query.
    """

    embedding = generate_embeddings(
        [query],
        model=model,
        batch_size=1
    )

    return embedding[0]


# ---------------------------------------------------------
# Self-test
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Test the embedding model."
    )

    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Sentence Transformer model to use."
    )

    args = parser.parse_args()

    # Load model
    model = load_embedding_model(
        args.model
    )

    # Sample legal texts
    texts = [
        "Either party may terminate this Agreement upon thirty days written notice.",

        "The Receiving Party shall keep all Confidential Information in strict confidence.",

        "This Agreement shall be governed by the laws of the State of Delaware."
    ]

    # Generate embeddings
    embeddings = generate_embeddings(
        texts,
        model=model
    )

    # Display results
    print("\n" + "=" * 60)
    print("              EMBEDDING SELF-TEST")
    print("=" * 60)

    print(
        f"Model                : "
        f"{args.model}"
    )

    print(
        f"Embedding dimension  : "
        f"{get_embedding_dimension(model)}"
    )

    print(
        f"Embeddings generated : "
        f"{len(embeddings)}"
    )

    print(
        f"Array shape          : "
        f"{embeddings.shape}"
    )

    print(
        f"Array data type      : "
        f"{embeddings.dtype}"
    )

    print(
        f"Batch size           : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Max sequence length  : "
        f"{model.max_seq_length} tokens"
    )

    # Check vector normalization
    norm = np.linalg.norm(
        embeddings[0]
    )

    print(
        f"First vector norm    : "
        f"{norm:.4f}"
    )

    # Cosine similarity
    similarity = float(
        embeddings[0] @ embeddings[1]
    )

    print(
        f"Similarity (text 1 & 2): "
        f"{similarity:.4f}"
    )

    print("=" * 60)


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    main()

