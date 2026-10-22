"""Query embedding and keyword, vector or hybrid search over the chunk index."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import OpenAI

from insurance_rag.config import VECTOR_DIMENSIONS, VECTOR_FIELD

SearchMode = Literal["keyword", "vector", "hybrid"]
SEARCH_MODES: tuple[SearchMode, ...] = ("keyword", "vector", "hybrid")
VECTOR_MODES = frozenset({"vector", "hybrid"})

# Vector search scored best on the labelled questions in retrieval_questions.json
# (see outputs/retrieved_chunks/retrieval_results.json), so it is the default.
DEFAULT_SEARCH_MODE: SearchMode = "vector"

# Approximate nearest-neighbour search considers this many candidates before
# the top results are returned, which keeps recall high for small ``top`` values.
MIN_NEAREST_NEIGHBOURS = 200

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


@dataclass(frozen=True)
class EmbeddingResult:
    """A query embedding and the number of input tokens it used."""

    vector: list[float]
    input_tokens: int


def embed_query(client: OpenAI, deployment: str, text: str) -> EmbeddingResult:
    """Embed the query text and validate the resulting vector.

    Raises:
        RuntimeError: If the vector has the wrong size or a non-finite value.
    """
    response = client.embeddings.create(model=deployment, input=text)
    vector = response.data[0].embedding
    if len(vector) != VECTOR_DIMENSIONS:
        raise RuntimeError(
            f"Query embedding has {len(vector)} dimensions; "
            f"the index expects {VECTOR_DIMENSIONS}."
        )
    if not all(math.isfinite(value) for value in vector):
        raise RuntimeError("The query embedding contains a non-finite value.")

    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "input_tokens", 0)
    return EmbeddingResult(vector=vector, input_tokens=input_tokens or 0)


def escape_odata_string(value: str) -> str:
    """Escape a value for use inside a single-quoted OData string literal."""
    return value.replace("'", "''")


def create_filter(
    document_code: str | None = None, document_type: str | None = None
) -> str | None:
    """Build an OData filter for the optional document code and type, if given."""
    clauses: list[str] = []
    if document_code:
        clauses.append(f"document_code eq '{escape_odata_string(document_code)}'")
    if document_type:
        clauses.append(f"document_type eq '{escape_odata_string(document_type)}'")
    return " and ".join(clauses) if clauses else None


def search(
    client: SearchClient,
    query: str,
    mode: SearchMode,
    top: int,
    filter_expression: str | None = None,
    query_vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Run a keyword, vector or hybrid search and return ranked results.

    Each result is a dict of the ``RESULT_FIELDS`` plus ``rank`` (starting at 1)
    and ``score`` (the Azure AI Search relevance score).

    Raises:
        ValueError: If ``mode`` is not a supported search mode.
        RuntimeError: If a vector mode is requested without a query vector.
    """
    if mode not in SEARCH_MODES:
        raise ValueError(f"Unsupported search mode: {mode}")

    search_options: dict[str, Any] = {
        # Supplying both search_text and vector_queries performs hybrid
        # retrieval. None is used for pure vector search so no keyword score
        # is added.
        "search_text": query if mode in {"hybrid", "keyword"} else None,
        "filter": filter_expression,
        "select": RESULT_FIELDS,
        "top": top,
        "include_total_count": True,
    }
    if mode in VECTOR_MODES:
        if query_vector is None:
            raise RuntimeError("A query vector is required for vector search.")
        search_options["vector_queries"] = [
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=max(MIN_NEAREST_NEIGHBOURS, top),
                fields=VECTOR_FIELD,
            )
        ]
        search_options["vector_filter_mode"] = "preFilter"

    results: list[dict[str, Any]] = []
    for rank, item in enumerate(client.search(**search_options), start=1):
        result = dict(item)
        result["rank"] = rank
        result["score"] = result.pop("@search.score", None)
        results.append(result)
    return results
