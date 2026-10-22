#!/usr/bin/env python3
"""Evaluate keyword, vector, and hybrid retrieval against labelled questions."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from insurance_rag.clients import create_openai_client, create_search_client
from insurance_rag.config import (
    OUTPUTS_DIR,
    PROJECT_ROOT,
    load_environment,
    load_openai_settings,
    load_search_settings,
)
from insurance_rag.logging_setup import add_logging_arguments, configure_logging
from insurance_rag.retrieval import (
    SEARCH_MODES,
    VECTOR_MODES,
    create_filter,
    embed_query,
    search,
)

logger = logging.getLogger(__name__)

# Resolved from the project root so the script works from any directory.
DEFAULT_INPUT = PROJECT_ROOT / "retrieval_questions.json"
DEFAULT_OUTPUT = OUTPUTS_DIR / "retrieved_chunks" / "retrieval_results.json"
DEFAULT_MODES = SEARCH_MODES
STANDARD_METRIC_CUTOFFS = (1, 3, 5, 10, 20)


def display_path(path: Path) -> str:
    """Return ``path`` relative to the project root when inside it, else as given.

    This keeps machine-specific absolute paths out of the saved results file.
    """
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Azure AI Search retrieval using expected chunk IDs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            f"Evaluation questions JSON file (default: {display_path(DEFAULT_INPUT)})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Detailed results JSON file (default: {display_path(DEFAULT_OUTPUT)}).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
        help="Retrieval modes to evaluate (default: keyword vector hybrid).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of chunks to retrieve for each question (default: 5).",
    )
    parser.add_argument(
        "--question-id",
        help="Evaluate one question only, for example Q001.",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    if args.top < 1 or args.top > 50:
        parser.error("--top must be between 1 and 50")
    # Avoid repeating a mode if it is supplied more than once.
    args.modes = list(dict.fromkeys(args.modes))
    return args


def metric_cutoffs(top: int) -> tuple[int, ...]:
    """Return standard cutoffs up to top, always including top itself."""
    return tuple(
        sorted({cutoff for cutoff in STANDARD_METRIC_CUTOFFS if cutoff <= top} | {top})
    )


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load and validate the evaluation questions and their expected chunk IDs."""
    questions: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    for index, item in enumerate(read_question_array(path), start=1):
        question_id = validate_question(item, index, seen_question_ids)
        seen_question_ids.add(question_id)
        questions.append(item)
    return questions


def read_question_array(path: Path) -> list[Any]:
    """Read the evaluation file, which must hold a non-empty JSON array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or not a non-empty array.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError("The evaluation file must contain a non-empty JSON array.")
    return data


def is_non_blank_string(value: Any) -> bool:
    """Return whether ``value`` is a string containing non-whitespace text."""
    return isinstance(value, str) and bool(value.strip())


def validate_question(item: Any, index: int, seen_question_ids: set[str]) -> str:
    """Validate one evaluation item and return its question ID.

    Raises:
        ValueError: If a required field is missing or invalid, or the
            question ID or an expected chunk ID is duplicated.
    """
    if not isinstance(item, dict):
        raise ValueError(f"Evaluation item {index} must be a JSON object.")
    question_id = item.get("question_id")
    if not is_non_blank_string(question_id):
        raise ValueError(f"Evaluation item {index} has no question_id.")
    if question_id in seen_question_ids:
        raise ValueError(f"Duplicate question_id: {question_id}")
    if not is_non_blank_string(item.get("question")):
        raise ValueError(f"Question {question_id} has no question text.")

    expected = item.get("expected_chunk_ids")
    if (
        not isinstance(expected, list)
        or not expected
        or not all(is_non_blank_string(value) for value in expected)
    ):
        raise ValueError(
            f"Question {question_id} must have at least one expected_chunk_id."
        )
    if len(expected) != len(set(expected)):
        raise ValueError(f"Question {question_id} contains duplicate expected IDs.")

    for optional_field in ("document_code", "document_type"):
        value = item.get(optional_field)
        if value is not None and not is_non_blank_string(value):
            raise ValueError(f"Question {question_id} has an invalid {optional_field}.")
    return question_id


def score_result(
    question: dict[str, Any],
    mode: str,
    retrieved: list[dict[str, Any]],
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
    """Score one question's retrieved chunks against its expected chunk IDs.

    Returns hit, recall and all-expected metrics at each cutoff, plus the
    reciprocal rank of the first relevant chunk.
    """
    expected_ids = list(question["expected_chunk_ids"])
    expected_set = set(expected_ids)
    retrieved_ids = [item["chunk_id"] for item in retrieved]
    relevant_ranks = [
        rank
        for rank, chunk_id in enumerate(retrieved_ids, start=1)
        if chunk_id in expected_set
    ]
    first_relevant_rank = relevant_ranks[0] if relevant_ranks else None

    metrics: dict[str, Any] = {}
    for cutoff in cutoffs:
        found_ids = expected_set.intersection(retrieved_ids[:cutoff])
        found_count = len(found_ids)
        metrics[f"at_{cutoff}"] = {
            "hit": found_count > 0,
            "expected_found": found_count,
            "expected_total": len(expected_set),
            "recall": found_count / len(expected_set),
            "all_expected": found_count == len(expected_set),
        }

    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "mode": mode,
        "document_code": question.get("document_code"),
        "document_type": question.get("document_type"),
        "expected_chunk_ids": expected_ids,
        "retrieved_chunk_ids": retrieved_ids,
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "hits": {
            f"hit_at_{cutoff}": metrics[f"at_{cutoff}"]["hit"] for cutoff in cutoffs
        },
        "metrics": metrics,
        "retrieved_results": [
            {
                "rank": item.get("rank"),
                "score": item.get("score"),
                "chunk_id": item.get("chunk_id"),
                "document_code": item.get("document_code"),
                "title": item.get("title"),
                "section_heading": item.get("section_heading"),
                "subsection_headings": item.get("subsection_headings"),
                "semantic_heading": item.get("semantic_heading"),
                "start_pdf_page": item.get("start_pdf_page"),
                "end_pdf_page": item.get("end_pdf_page"),
                "text": item.get("text"),
                "is_expected": item.get("chunk_id") in expected_set,
            }
            for item in retrieved
        ],
    }


def calculate_summary(
    results: list[dict[str, Any]],
    modes: list[str],
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
    """Aggregate per-question scores into hit rate, recall and MRR for each mode."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["mode"]].append(result)

    summary: dict[str, Any] = {}
    for mode in modes:
        mode_results = grouped[mode]
        count = len(mode_results)
        summary[mode] = {
            "questions": count,
            "mrr": sum(result["reciprocal_rank"] for result in mode_results) / count,
            "cutoffs": {
                f"at_{cutoff}": {
                    "hit_rate": sum(
                        int(result["metrics"][f"at_{cutoff}"]["hit"])
                        for result in mode_results
                    )
                    / count,
                    "mean_recall": sum(
                        result["metrics"][f"at_{cutoff}"]["recall"]
                        for result in mode_results
                    )
                    / count,
                    "all_expected_rate": sum(
                        int(result["metrics"][f"at_{cutoff}"]["all_expected"])
                        for result in mode_results
                    )
                    / count,
                    "misses": sum(
                        not result["metrics"][f"at_{cutoff}"]["hit"]
                        for result in mode_results
                    ),
                }
                for cutoff in cutoffs
            },
        }
    return summary


def print_summary(summary: dict[str, Any], cutoffs: tuple[int, ...]) -> None:
    """Print the aggregated metrics for each retrieval mode as a table."""
    print("\nRetrieval evaluation summary")
    print("=" * 78)
    print(
        f"{'Mode':<10} {'K':>4} {'Questions':>10} {'Hit@K':>10} "
        f"{'Recall@K':>10} {'All@K':>10} {'Misses':>9}"
    )
    print("-" * 78)
    for mode, metrics in summary.items():
        for cutoff in cutoffs:
            cutoff_metrics = metrics["cutoffs"][f"at_{cutoff}"]
            print(
                f"{mode:<10} {cutoff:>4} {metrics['questions']:>10} "
                f"{cutoff_metrics['hit_rate']:>9.1%} "
                f"{cutoff_metrics['mean_recall']:>9.1%} "
                f"{cutoff_metrics['all_expected_rate']:>9.1%} "
                f"{cutoff_metrics['misses']:>9}"
            )
    print("\nMean reciprocal rank")
    for mode, metrics in summary.items():
        print(f"  {mode:<10} {metrics['mrr']:.3f}")


def print_question_summary(
    question: dict[str, Any],
    results: list[dict[str, Any]],
    cutoffs: tuple[int, ...],
) -> None:
    """Print retrieval metrics and ranked chunks for one question."""
    print(f"\nQuestion summary: {question['question_id']}")
    print(f"Question: {question['question']}")
    print("Expected chunks:")
    for chunk_id in question["expected_chunk_ids"]:
        print(f"  - {chunk_id}")

    print("\n" + "=" * 86)
    print(
        f"{'Mode':<10} {'K':>4} {'Hit':>8} {'Expected':>10} "
        f"{'Recall':>10} {'All':>8} {'First hit':>12}"
    )
    print("-" * 86)
    for result in results:
        first_hit = result["first_relevant_rank"]
        for cutoff in cutoffs:
            metrics = result["metrics"][f"at_{cutoff}"]
            expected = f"{metrics['expected_found']}/{metrics['expected_total']}"
            print(
                f"{result['mode']:<10} {cutoff:>4} "
                f"{str(metrics['hit']):>8} "
                f"{expected:>10} "
                f"{metrics['recall']:>9.1%} "
                f"{str(metrics['all_expected']):>8} "
                f"{str(first_hit) if first_hit is not None else 'not found':>12}"
            )
    print("\nReciprocal rank")
    for result in results:
        print(f"  {result['mode']:<10} {result['reciprocal_rank']:.3f}")

    print("\nReturned chunks")
    print("-" * 79)
    for result in results:
        print(f"{result['mode'].upper()}:")
        for retrieved in result["retrieved_results"]:
            expected_marker = " [EXPECTED]" if retrieved["is_expected"] else ""
            score = retrieved["score"]
            score_text = (
                f"{score:.6f}" if isinstance(score, (int, float)) else "unknown"
            )
            print(
                f"  {retrieved['rank']}. {retrieved['chunk_id']} "
                f"(score: {score_text}){expected_marker}"
            )


def main() -> int:
    """Run the evaluation for each question and mode and save the results.

    Returns:
        The process exit code.
    """
    load_environment()
    args = parse_args()
    configure_logging(args.verbose, args.quiet)
    questions = load_questions(args.input)
    cutoffs = metric_cutoffs(args.top)

    if args.question_id:
        requested_id = args.question_id.strip().upper()
        questions = [
            question
            for question in questions
            if question["question_id"].upper() == requested_id
        ]
        if not questions:
            raise ValueError(f"Question ID not found: {args.question_id}")

    search_settings = load_search_settings()
    search_client = create_search_client(search_settings)

    openai_client = None
    embedding_deployment = ""
    if any(mode in VECTOR_MODES for mode in args.modes):
        openai_settings = load_openai_settings()
        openai_client = create_openai_client(openai_settings)
        embedding_deployment = openai_settings.embedding_deployment

    evaluation_results: list[dict[str, Any]] = []
    total_runs = len(questions) * len(args.modes)
    completed = 0
    for question in questions:
        question_results: list[dict[str, Any]] = []
        query_vector = None
        if openai_client is not None:
            query_vector = embed_query(
                openai_client, embedding_deployment, question["question"]
            ).vector

        filter_expression = create_filter(
            question.get("document_code"), question.get("document_type")
        )
        for mode in args.modes:
            retrieved = search(
                search_client,
                question["question"],
                mode,
                args.top,
                filter_expression,
                query_vector if mode in VECTOR_MODES else None,
            )
            scored_result = score_result(question, mode, retrieved, cutoffs)
            evaluation_results.append(scored_result)
            question_results.append(scored_result)
            completed += 1
            logger.info(
                "Evaluated %d/%d: %s (%s)",
                completed,
                total_runs,
                question["question_id"],
                mode,
            )

        print_question_summary(question, question_results, cutoffs)

    summary = calculate_summary(evaluation_results, args.modes, cutoffs)
    output = {
        "input_file": display_path(args.input),
        "index_name": search_settings.index_name,
        "top": args.top,
        "metric_cutoffs": list(cutoffs),
        "modes": args.modes,
        "summary": summary,
        "results": evaluation_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print_summary(summary, cutoffs)
    logger.info("Detailed results: %s", display_path(args.output))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
