
"""
Task 6 - RAG Evaluation

Evaluates the Legal RAG system using:

1. ROUGE-L
2. Faithfulness
3. Answer Relevance

Questions and reference answers are read from:

    evaluation/questions.json

Results are saved to:

    evaluation/results/

Run:

    python evaluation.py

Quick test:

    python evaluation.py --limit 3

Use fewer retrieved chunks:

    python evaluation.py --top-k 3
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from rouge_score import rouge_scorer

from src.rag import answer_question, load_api_key, get_model_name
from src.vector_store import load_vector_store


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_DIR = PROJECT_ROOT / "evaluation"

QUESTIONS_FILE = EVALUATION_DIR / "questions.json"

RESULTS_DIR = EVALUATION_DIR / "results"

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

DEFAULT_TOP_K = 5


# ---------------------------------------------------------
# Load questions
# ---------------------------------------------------------

def load_questions():

    if not QUESTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Questions file not found:\n{QUESTIONS_FILE}"
        )

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        questions = json.load(file)

    if not isinstance(questions, list):

        raise ValueError(
            "questions.json must contain a list."
        )

    return questions


# ---------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------

def calculate_rouge_l(
    reference,
    generated
):

    if not reference or not generated:

        return None

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    score = scorer.score(
        reference,
        generated
    )

    return score["rougeL"].fmeasure


# ---------------------------------------------------------
# Gemini evaluation
# ---------------------------------------------------------

def evaluate_with_gemini(
    question,
    answer,
    context
):

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=load_api_key()
    )

    prompt = f"""
You are evaluating a legal document RAG system.

Evaluate the answer using ONLY the supplied context.

Give two scores between 0 and 1.

1. Faithfulness:
Does the answer contain claims supported by the context?

2. Answer Relevance:
Does the answer directly answer the question?

Question:
{question}

Context:
{context}

Answer:
{answer}

Return ONLY valid JSON:

{{
    "faithfulness": 0.0,
    "answer_relevance": 0.0
}}
"""

    response = client.models.generate_content(

        model=get_model_name(),

        contents=prompt,

        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json"
        )
    )

    result = json.loads(
        response.text
    )

    faithfulness = float(
        result["faithfulness"]
    )

    relevance = float(
        result["answer_relevance"]
    )

    # Keep values between 0 and 1

    faithfulness = max(
        0,
        min(1, faithfulness)
    )

    relevance = max(
        0,
        min(1, relevance)
    )

    return faithfulness, relevance


# ---------------------------------------------------------
# Evaluate one question
# ---------------------------------------------------------

def evaluate_question(
    question,
    reference_answer,
    index,
    metadata,
    top_k,
    use_judge=True
):

    try:

        # Run RAG
        generated_answer, results = answer_question(
            question,
            index=index,
            metadata=metadata,
            top_k=top_k
        )

        # Get retrieved context
        contexts = []

        for result in results:

            contexts.append(
                result["text"]
            )

        context = "\n\n---\n\n".join(
            contexts
        )

        # ROUGE-L
        rouge_l = calculate_rouge_l(
            reference_answer,
            generated_answer
        )

        # Gemini judge
        faithfulness = None
        relevance = None

        if context and use_judge:

            faithfulness, relevance = (
                evaluate_with_gemini(
                    question,
                    generated_answer,
                    context
                )
            )

        # Sources
        sources = []

        for result in results:

            sources.append(
                f"{result['source_file']}, "
                f"page {result['page_number']}, "
                f"chunk {result['chunk_index']}"
            )

        return {

            "question":
                question,

            "reference_answer":
                reference_answer,

            "generated_answer":
                generated_answer,

            "rouge_l":
                rouge_l,

            "faithfulness":
                faithfulness,

            "answer_relevance":
                relevance,

            "sources":
                sources,

            "error":
                ""
        }

    except Exception as error:

        return {

            "question":
                question,

            "reference_answer":
                reference_answer,

            "generated_answer":
                "",

            "rouge_l":
                None,

            "faithfulness":
                None,

            "answer_relevance":
                None,

            "sources":
                [],

            "error":
                str(error)
        }


# ---------------------------------------------------------
# Calculate averages
# ---------------------------------------------------------

def calculate_average(
    results,
    metric
):

    values = [

        result[metric]

        for result in results

        if result[metric] is not None

    ]

    if not values:

        return None

    return sum(values) / len(values)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

def display_results(results):

    print()
    print("=" * 100)
    print("                    RAG EVALUATION RESULTS")
    print("=" * 100)

    print(
        f"{'#':<5}"
        f"{'QUESTION':<50}"
        f"{'ROUGE-L':>10}"
        f"{'FAITHFUL':>12}"
        f"{'RELEVANCE':>12}"
    )

    print("-" * 100)

    for number, result in enumerate(
        results,
        start=1
    ):

        question = result["question"]

        # Shorten long questions
        if len(question) > 47:

            question = (
                question[:44]
                + "..."
            )

        rouge = result["rouge_l"]

        faithfulness = (
            result["faithfulness"]
        )

        relevance = (
            result["answer_relevance"]
        )

        rouge_text = (
            "n/a"
            if rouge is None
            else f"{rouge:.3f}"
        )

        faith_text = (
            "n/a"
            if faithfulness is None
            else f"{faithfulness:.3f}"
        )

        relevance_text = (
            "n/a"
            if relevance is None
            else f"{relevance:.3f}"
        )

        print(
            f"{number:<5}"
            f"{question:<50}"
            f"{rouge_text:>10}"
            f"{faith_text:>12}"
            f"{relevance_text:>12}"
        )

    print("-" * 100)

    average_rouge = calculate_average(
        results,
        "rouge_l"
    )

    average_faithfulness = calculate_average(
        results,
        "faithfulness"
    )

    average_relevance = calculate_average(
        results,
        "answer_relevance"
    )

    print(
        f"{'':<5}"
        f"{'AVERAGE':<50}"
        f"{format_score(average_rouge):>10}"
        f"{format_score(average_faithfulness):>12}"
        f"{format_score(average_relevance):>12}"
    )

    print("=" * 100)


def format_score(value):

    if value is None:

        return "n/a"

    return f"{value:.3f}"


# ---------------------------------------------------------
# Detailed results
# ---------------------------------------------------------

def display_detailed_results(
    results
):

    print()
    print("=" * 100)
    print("                    DETAILED RESULTS")
    print("=" * 100)

    for number, result in enumerate(
        results,
        start=1
    ):

        print()
        print(
            f"[{number}] "
            f"{result['question']}"
        )

        print(
            "\nReference Answer:"
        )

        print(
            result["reference_answer"]
        )

        print(
            "\nGenerated Answer:"
        )

        print(
            result["generated_answer"]
        )

        print(
            "\nScores:"
        )

        print(
            f"ROUGE-L        : "
            f"{format_score(result['rouge_l'])}"
        )

        print(
            f"Faithfulness   : "
            f"{format_score(result['faithfulness'])}"
        )

        print(
            f"Answer Relevance: "
            f"{format_score(result['answer_relevance'])}"
        )

        print(
            "\nSources:"
        )

        for source in result["sources"]:

            print(
                f"  - {source}"
            )

        if result["error"]:

            print(
                "\nError:"
            )

            print(
                result["error"]
            )

        print("-" * 100)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

def save_results(
    results
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # -----------------------------------------------------
    # JSON
    # -----------------------------------------------------

    json_file = (
        RESULTS_DIR
        / f"evaluation_{timestamp}.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------

    csv_file = (
        RESULTS_DIR
        / f"evaluation_{timestamp}.csv"
    )

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            "question",

            "reference_answer",

            "generated_answer",

            "rouge_l",

            "faithfulness",

            "answer_relevance",

            "sources",

            "error"

        ])

        for result in results:

            writer.writerow([

                result["question"],

                result["reference_answer"],

                result["generated_answer"],

                result["rouge_l"],

                result["faithfulness"],

                result["answer_relevance"],

                "; ".join(
                    result["sources"]
                ),

                result["error"]

            ])

    print()
    print(
        f"JSON saved: {json_file}"
    )

    print(
        f"CSV saved : {csv_file}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate the Legal RAG system."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions."
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show reference and generated answers."
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save JSON and CSV results."
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load questions
    # -----------------------------------------------------

    try:

        questions = load_questions()

    except Exception as error:

        print(
            f"ERROR loading questions: {error}"
        )

        return

    # -----------------------------------------------------
    # Limit questions if requested
    # -----------------------------------------------------

    if args.limit:

        questions = questions[
            :args.limit
        ]

    print(
        f"\nEvaluating {len(questions)} questions..."
    )

    # -----------------------------------------------------
    # Load FAISS
    # -----------------------------------------------------

    try:

        index, metadata = load_vector_store(
            VECTORSTORE_DIR
        )

    except Exception as error:

        print(
            f"ERROR loading vector store: "
            f"{error}"
        )

        return

    # -----------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------

    results = []

    for number, item in enumerate(
        questions,
        start=1
    ):

        question = item[
            "question"
        ]

        reference_answer = item.get(
            "reference_answer",
            ""
        )

        print(
            f"\n[{number}/{len(questions)}] "
            f"{question}"
        )

        result = evaluate_question(

            question,

            reference_answer,

            index,

            metadata,

            args.top_k

        )

        results.append(
            result
        )

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    display_results(
        results
    )

    # -----------------------------------------------------
    # Detailed output
    # -----------------------------------------------------

    if args.detailed:

        display_detailed_results(
            results
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    if not args.no_save:

        save_results(
            results
        )


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":

    main()
