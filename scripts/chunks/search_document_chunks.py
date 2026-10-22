#!/usr/bin/env python3
"""Search indexed insurance-policy chunks using Azure AI Search."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from insurance_rag.clients import create_openai_client, create_search_client
from insurance_rag.config import (
    load_environment,
    load_openai_settings,
    load_search_settings,
)
from insurance_rag.logging_setup import add_logging_arguments, configure_logging
from insurance_rag.retrieval import (
    DEFAULT_SEARCH_MODE,
    SEARCH_MODES,
    VECTOR_MODES,
    create_filter,
    embed_query,
    search,
)

logger = logging.getLogger(__name__)

DEFAULT_TOP = 5


def parse_args() -> argparse.Namespace:
    """Parse command-line options for searching the chunk index."""
    parser = argparse.ArgumentParser(
        description="Search policy chunks in an Azure AI Search index."
    )
    parser.add_argument("question", help="Question or search text.")
    parser.add_argument(
        "--mode",
        choices=SEARCH_MODES,
        default=DEFAULT_SEARCH_MODE,
        help=f"Retrieval method (default: {DEFAULT_SEARCH_MODE}).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=f"Number of results to return (default: {DEFAULT_TOP}).",
    )
    parser.add_argument(
        "--document-code",
        help="Restrict results to a document code, for example IP-HO-2-012.",
    )
    parser.add_argument(
        "--document-type",
        help=(
            "Restrict results to a document type, for example "
            "insurance_product_information."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of formatted text.",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("question must not be empty")
    if args.top < 1 or args.top > 50:
        parser.error("--top must be between 1 and 50")
    return args


def page_range(result: dict[str, Any], kind: str) -> str:
    """Format a result's start and end pages of the given kind as a range."""
    start = result.get(f"start_{kind}_page")
    end = result.get(f"end_{kind}_page")
    if start is None and end is None:
        return "unknown"
    if start == end or end is None:
        return str(start)
    return f"{start}-{end}"


def print_results(
    question: str,
    mode: str,
    filter_expression: str | None,
    results: list[dict[str, Any]],
) -> None:
    """Print the search results with their scores, metadata and text."""
    print(f"Question: {question}")
    print(f"Mode: {mode}")
    if filter_expression:
        print(f"Filter: {filter_expression}")
    print(f"Results returned: {len(results)}")

    if not results:
        print("\nNo matching chunks were found.")
        return

    for result in results:
        print("\n" + "=" * 80)
        score = result.get("score")
        score_text = f"{score:.6f}" if isinstance(score, (int, float)) else "unknown"
        print(f"Rank: {result['rank']} | Search score: {score_text}")
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Title: {result.get('title')}")
        print(f"Document code: {result.get('document_code')}")
        print(f"Document type: {result.get('document_type')}")
        print(f"Section: {result.get('section_heading')}")

        subsections = result.get("subsection_headings") or []
        if subsections:
            print(f"Subsection: {' > '.join(subsections)}")
        if result.get("semantic_heading"):
            print(f"Semantic heading: {result['semantic_heading']}")

        print(f"PDF page: {page_range(result, 'pdf')}")
        print(f"Printed page: {page_range(result, 'printed')}")
        print("\nText:")
        print(result.get("text", ""))


def main() -> int:
    """Search the index for the question and print the results.

    Returns:
        The process exit code.
    """
    load_environment()
    args = parse_args()
    configure_logging(args.verbose, args.quiet)
    search_settings = load_search_settings()

    query_vector = None
    if args.mode in VECTOR_MODES:
        openai_settings = load_openai_settings()
        query_vector = embed_query(
            create_openai_client(openai_settings),
            openai_settings.embedding_deployment,
            args.question,
        ).vector

    search_client = create_search_client(search_settings)
    filter_expression = create_filter(args.document_code, args.document_type)
    results = search(
        search_client,
        args.question,
        args.mode,
        args.top,
        filter_expression,
        query_vector,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "question": args.question,
                    "mode": args.mode,
                    "filter": filter_expression,
                    "result_count": len(results),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print_results(args.question, args.mode, filter_expression, results)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
