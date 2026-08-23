#!/usr/bin/env python3
"""Evaluate keyword, vector, and hybrid retrieval against labelled questions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from search_document_chunks import (
    create_filter,
    create_query_embedding,
    require_environment,
    search,
)


DEFAULT_INPUT = Path("src/search_chunks/retrieval_questions.json")
DEFAULT_OUTPUT = Path("src/search_chunks/retrieval_results.json")
DEFAULT_MODES = ("keyword", "vector", "hybrid")
STANDARD_METRIC_CUTOFFS = (1, 3, 5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Azure AI Search retrieval using expected chunk IDs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Evaluation questions JSON file (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Detailed results JSON file (default: {DEFAULT_OUTPUT}).",
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
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError("The evaluation file must contain a non-empty JSON array.")

    questions: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation item {index} must be a JSON object.")
        question_id = item.get("question_id")
        question = item.get("question")
        expected = item.get("expected_chunk_ids")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Evaluation item {index} has no question_id.")
        if question_id in seen_question_ids:
            raise ValueError(f"Duplicate question_id: {question_id}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question {question_id} has no question text.")
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(value, str) and value.strip() for value in expected)
        ):
            raise ValueError(
                f"Question {question_id} must have at least one expected_chunk_id."
            )
        if len(expected) != len(set(expected)):
            raise ValueError(f"Question {question_id} contains duplicate expected IDs.")

        for optional_field in ("document_code", "document_type"):
            value = item.get(optional_field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"Question {question_id} has an invalid {optional_field}."
                )

        seen_question_ids.add(question_id)
        questions.append(item)
    return questions


def score_result(
    question: dict[str, Any],
    mode: str,
    retrieved: list[dict[str, Any]],
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
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
            f"hit_at_{cutoff}": metrics[f"at_{cutoff}"]["hit"]
            for cutoff in cutoffs
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
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["mode"]].append(result)

    summary: dict[str, Any] = {}
    for mode in modes:
        mode_results = grouped[mode]
        count = len(mode_results)
        summary[mode] = {
            "questions": count,
            "mrr": sum(result["reciprocal_rank"] for result in mode_results)
            / count,
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


def print_summary(
    summary: dict[str, Any], cutoffs: tuple[int, ...]
) -> None:
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
    load_dotenv()
    args = parse_args()
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

    search_config = require_environment(
        ("AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_INDEX_NAME")
    )
    search_key = (
        os.getenv("AZURE_SEARCH_QUERY_KEY", "").strip()
        or os.getenv("AZURE_SEARCH_ADMIN_KEY", "").strip()
    )
    if not search_key:
        raise ValueError(
            "Missing AZURE_SEARCH_QUERY_KEY or AZURE_SEARCH_ADMIN_KEY environment variable."
        )

    openai_config: dict[str, str] | None = None
    if any(mode in {"vector", "hybrid"} for mode in args.modes):
        openai_config = require_environment(
            (
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            )
        )

    search_client = SearchClient(
        endpoint=search_config["AZURE_SEARCH_ENDPOINT"],
        index_name=search_config["AZURE_SEARCH_INDEX_NAME"],
        credential=AzureKeyCredential(search_key),
    )

    evaluation_results: list[dict[str, Any]] = []
    total_runs = len(questions) * len(args.modes)
    completed = 0
    for question in questions:
        question_results: list[dict[str, Any]] = []
        query_vector = None
        if openai_config is not None:
            query_vector = create_query_embedding(
                question["question"],
                openai_config["AZURE_OPENAI_ENDPOINT"],
                openai_config["AZURE_OPENAI_API_KEY"],
                openai_config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            )

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
                query_vector if mode in {"vector", "hybrid"} else None,
            )
            scored_result = score_result(question, mode, retrieved, cutoffs)
            evaluation_results.append(scored_result)
            question_results.append(scored_result)
            completed += 1
            print(
                f"Evaluated {completed}/{total_runs}: "
                f"{question['question_id']} ({mode})"
            )

        print_question_summary(question, question_results, cutoffs)

    summary = calculate_summary(evaluation_results, args.modes, cutoffs)
    output = {
        "input_file": str(args.input),
        "index_name": search_config["AZURE_SEARCH_INDEX_NAME"],
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
    print(f"\nDetailed results: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
