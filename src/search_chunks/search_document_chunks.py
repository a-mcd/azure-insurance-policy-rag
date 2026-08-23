#!/usr/bin/env python3
"""Search indexed insurance-policy chunks using Azure AI Search."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery


VECTOR_DIMENSIONS = 1536
VECTOR_FIELD = "content_vector"
DEFAULT_TOP = 5

RESULT_FIELDS = [
    "chunk_id",
    "chunk_sequence",
    "chunk_type",
    "document_code",
    "document_name",
    "document_type",
    "title",
    "section_heading",
    "subsection_headings",
    "semantic_heading",
    "semantic_heading_level",
    "text",
    "token_count",
    "start_pdf_page",
    "end_pdf_page",
    "start_printed_page",
    "end_printed_page",
    "source_pdf_pages",
    "source_printed_pages",
    "table_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search policy chunks in an Azure AI Search index."
    )
    parser.add_argument("question", help="Question or search text.")
    parser.add_argument(
        "--mode",
        choices=("hybrid", "vector", "keyword"),
        default="hybrid",
        help="Retrieval method (default: hybrid).",
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
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("question must not be empty")
    if args.top < 1 or args.top > 50:
        parser.error("--top must be between 1 and 50")
    return args


def require_environment(names: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            values[name] = value
        else:
            missing.append(name)
    if missing:
        raise ValueError("Missing environment variables: " + ", ".join(missing))
    return values


def create_query_embedding(
    question: str,
    endpoint: str,
    api_key: str,
    deployment: str,
) -> list[float]:
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        max_retries=5,
    )
    response = client.embeddings.create(model=deployment, input=question)
    vector = response.data[0].embedding
    if len(vector) != VECTOR_DIMENSIONS:
        raise RuntimeError(
            f"Query embedding has {len(vector)} dimensions; "
            f"the index expects {VECTOR_DIMENSIONS}."
        )
    if not all(math.isfinite(value) for value in vector):
        raise RuntimeError("The query embedding contains a non-finite value.")
    return vector


def escape_odata_string(value: str) -> str:
    return value.replace("'", "''")


def create_filter(document_code: str | None, document_type: str | None) -> str | None:
    clauses: list[str] = []
    if document_code:
        clauses.append(f"document_code eq '{escape_odata_string(document_code)}'")
    if document_type:
        clauses.append(f"document_type eq '{escape_odata_string(document_type)}'")
    return " and ".join(clauses) if clauses else None


def search(
    client: SearchClient,
    question: str,
    mode: str,
    top: int,
    filter_expression: str | None,
    query_vector: list[float] | None,
) -> list[dict[str, Any]]:
    vector_queries = None
    if mode in {"hybrid", "vector"}:
        if query_vector is None:
            raise RuntimeError("A query vector is required for vector search.")
        vector_queries = [
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=max(200, top),
                fields=VECTOR_FIELD,
            )
        ]

    # Supplying both search_text and vector_queries performs hybrid retrieval.
    # None is used for pure vector search so no keyword score is added.
    search_text = question if mode in {"hybrid", "keyword"} else None
    search_options: dict[str, Any] = {
        "search_text": search_text,
        "filter": filter_expression,
        "select": RESULT_FIELDS,
        "top": top,
        "include_total_count": True,
    }
    if vector_queries is not None:
        search_options["vector_queries"] = vector_queries
        search_options["vector_filter_mode"] = "preFilter"

    response = client.search(
        **search_options,
    )

    results: list[dict[str, Any]] = []
    for rank, item in enumerate(response, start=1):
        result = dict(item)
        result["rank"] = rank
        result["score"] = result.pop("@search.score", None)
        results.append(result)
    return results


def page_range(result: dict[str, Any], kind: str) -> str:
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
    load_dotenv()
    args = parse_args()

    search_config = require_environment(
        (
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_INDEX_NAME",
        )
    )
    search_key = (
        os.getenv("AZURE_SEARCH_QUERY_KEY", "").strip()
        or os.getenv("AZURE_SEARCH_ADMIN_KEY", "").strip()
    )
    if not search_key:
        raise ValueError(
            "Missing AZURE_SEARCH_QUERY_KEY or AZURE_SEARCH_ADMIN_KEY environment variable."
        )

    query_vector = None
    if args.mode in {"hybrid", "vector"}:
        openai_config = require_environment(
            (
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            )
        )
        query_vector = create_query_embedding(
            args.question,
            openai_config["AZURE_OPENAI_ENDPOINT"],
            openai_config["AZURE_OPENAI_API_KEY"],
            openai_config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        )

    search_client = SearchClient(
        endpoint=search_config["AZURE_SEARCH_ENDPOINT"],
        index_name=search_config["AZURE_SEARCH_INDEX_NAME"],
        credential=AzureKeyCredential(search_key),
    )
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
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
