#!/usr/bin/env python3
"""Generate a grounded answer from chunks stored in Azure AI Search."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_TOP = 10
DEFAULT_MAX_COMPLETION_TOKENS = 4000
VECTOR_FIELD_CANDIDATES = (
    "embedding",
    "content_vector",
    "text_vector",
    "contentVector",
    "vector",
)
SOURCE_FIELD_CANDIDATES = (
    "chunk_id",
    "title",
    "document_code",
    "document_type",
    "section",
    "section_heading",
    "semantic_heading",
    "source_pdf_pages",
    "source_printed_pages",
    "pdf_page",
    "printed_page",
    "text",
)

SYSTEM_PROMPT = """You answer questions about the supplied home-insurance documents.

Rules:
- Use only the supplied policy excerpts. Do not use outside knowledge.
- Cite every factual statement using the source number in square brackets, for example [1].
- A citation must support the statement immediately before it.
- Clearly distinguish Admiral, Gold and Platinum policy levels when the excerpts do.
- Clearly distinguish standard, optional and excluded cover.
- Do not make a definitive claim decision or promise that a claim will be accepted.
- If the excerpts do not contain enough evidence, say that the available documents do not provide enough information.
- Do not invent policy terms, limits, exclusions, page numbers or citations.
- Give a direct, concise answer in British English.
"""


@dataclass(frozen=True)
class Settings:
    openai_endpoint: str
    openai_key: str
    embedding_deployment: str
    chat_deployment: str
    search_endpoint: str
    search_key: str
    search_index: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve policy chunks with vector search and generate a cited answer."
    )
    parser.add_argument("question", help="The user's insurance question")
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP})",
    )
    parser.add_argument(
        "--document-code",
        help="Optional exact document_code filter, for example IP-HO-2-012",
    )
    parser.add_argument(
        "--vector-field",
        help="Azure Search vector field. By default it is detected from the index.",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Include Azure Search scores in the source list",
    )
    args = parser.parse_args()

    if not args.question.strip():
        parser.error("question cannot be empty")
    if not 1 <= args.top <= 50:
        parser.error("--top must be between 1 and 50")
    return args


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    required = {
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        ),
        "AZURE_OPENAI_CHAT_DEPLOYMENT": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        "AZURE_SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "AZURE_SEARCH_ADMIN_KEY": os.getenv("AZURE_SEARCH_ADMIN_KEY"),
        "AZURE_SEARCH_INDEX_NAME": os.getenv("AZURE_SEARCH_INDEX_NAME"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(sorted(missing))
        )

    return Settings(
        openai_endpoint=required["AZURE_OPENAI_ENDPOINT"],  # type: ignore[arg-type]
        openai_key=required["AZURE_OPENAI_API_KEY"],  # type: ignore[arg-type]
        embedding_deployment=required[
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        ],  # type: ignore[arg-type]
        chat_deployment=required["AZURE_OPENAI_CHAT_DEPLOYMENT"],  # type: ignore[arg-type]
        search_endpoint=required["AZURE_SEARCH_ENDPOINT"],  # type: ignore[arg-type]
        search_key=required["AZURE_SEARCH_ADMIN_KEY"],  # type: ignore[arg-type]
        search_index=required["AZURE_SEARCH_INDEX_NAME"],  # type: ignore[arg-type]
    )


def make_openai_client(settings: Settings) -> OpenAI:
    base_url = settings.openai_endpoint.rstrip("/") + "/openai/v1/"
    return OpenAI(
        base_url=base_url,
        api_key=settings.openai_key,
    )


def get_index_fields(settings: Settings) -> dict[str, Any]:
    index_client = SearchIndexClient(
        endpoint=settings.search_endpoint,
        credential=AzureKeyCredential(settings.search_key),
    )
    index = index_client.get_index(settings.search_index)
    return {field.name: field for field in index.fields}


def find_vector_field(fields: dict[str, Any], requested: str | None) -> str:
    if requested:
        if requested not in fields:
            raise ValueError(
                f"Vector field '{requested}' is not in the index. "
                f"Available fields: {', '.join(sorted(fields))}"
            )
        field = fields[requested]
        if not getattr(field, "vector_search_dimensions", None):
            raise ValueError(f"Index field '{requested}' is not configured as a vector field.")
        return requested

    vector_fields = [
        name
        for name, field in fields.items()
        if getattr(field, "vector_search_dimensions", None)
    ]
    if len(vector_fields) == 1:
        return vector_fields[0]

    for candidate in VECTOR_FIELD_CANDIDATES:
        if candidate in vector_fields:
            return candidate

    if not vector_fields:
        raise ValueError("The Azure Search index does not contain a vector field.")
    raise ValueError(
        "The index contains multiple vector fields. Supply --vector-field with one of: "
        + ", ".join(vector_fields)
    )


def create_question_embedding(
    client: OpenAI, deployment: str, question: str
) -> list[float]:
    response = client.embeddings.create(model=deployment, input=[question])
    return response.data[0].embedding


def escape_odata_string(value: str) -> str:
    return value.replace("'", "''")


def retrieve_chunks(
    settings: Settings,
    question_vector: list[float],
    vector_field: str,
    available_fields: Iterable[str],
    top: int,
    document_code: str | None,
) -> list[dict[str, Any]]:
    search_client = SearchClient(
        endpoint=settings.search_endpoint,
        index_name=settings.search_index,
        credential=AzureKeyCredential(settings.search_key),
    )
    selected_fields = [
        field for field in SOURCE_FIELD_CANDIDATES if field in available_fields
    ]
    if "text" not in selected_fields:
        raise ValueError("The Azure Search index must contain a retrievable 'text' field.")

    filter_expression = None
    if document_code:
        if "document_code" not in available_fields:
            raise ValueError("The index has no document_code field to filter on.")
        filter_expression = f"document_code eq '{escape_odata_string(document_code)}'"

    vector_query = VectorizedQuery(
        vector=question_vector,
        k_nearest_neighbors=top,
        fields=vector_field,
    )
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        vector_filter_mode="preFilter",
        filter=filter_expression,
        select=selected_fields,
        top=top,
    )
    return [dict(result) for result in results]


def display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def source_label(chunk: dict[str, Any]) -> str:
    parts = []
    for field in ("title", "document_code", "section", "section_heading", "semantic_heading"):
        value = display_value(chunk.get(field)).strip()
        if value and value not in parts:
            parts.append(value)
    return " | ".join(parts) or display_value(chunk.get("chunk_id")) or "Policy excerpt"


def build_context(chunks: list[dict[str, Any]]) -> str:
    sources = []
    for number, chunk in enumerate(chunks, start=1):
        metadata = [f"Source: {source_label(chunk)}"]
        chunk_id = display_value(chunk.get("chunk_id")).strip()
        if chunk_id:
            metadata.append(f"Chunk ID: {chunk_id}")
        pdf_pages = display_value(
            chunk.get("source_pdf_pages") or chunk.get("pdf_page")
        ).strip()
        printed_pages = display_value(
            chunk.get("source_printed_pages") or chunk.get("printed_page")
        ).strip()
        if pdf_pages:
            metadata.append(f"PDF page: {pdf_pages}")
        if printed_pages:
            metadata.append(f"Printed page: {printed_pages}")
        sources.append(
            f"[{number}]\n" + "\n".join(metadata) + f"\nText:\n{chunk['text'].strip()}"
        )
    return "\n\n".join(sources)


def generate_answer(
    client: OpenAI,
    deployment: str,
    question: str,
    context: str,
) -> str:
    user_prompt = f"""Question:
{question}

Policy excerpts:
{context}

Answer the question using only these excerpts and include numbered citations."""

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # GPT-5 counts hidden reasoning and visible answer tokens against this
        # shared limit. Minimal reasoning is sufficient for grounded RAG and
        # leaves more of the allowance available for the cited answer.
        reasoning_effort="minimal",
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    )
    answer = response.choices[0].message.content
    if not answer:
        choice = response.choices[0]
        finish_reason = choice.finish_reason or "unknown"
        completion_details = (
            getattr(response.usage, "completion_tokens_details", None)
            if response.usage
            else None
        )
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
        reasoning_detail = (
            f", reasoning tokens: {reasoning_tokens}"
            if reasoning_tokens is not None
            else ""
        )
        raise RuntimeError(
            "The chat model returned an empty answer "
            f"(finish reason: {finish_reason}{reasoning_detail})."
        )
    return answer.strip()


def print_sources(chunks: list[dict[str, Any]], show_scores: bool) -> None:
    print("\nSources")
    print("-------")
    for number, chunk in enumerate(chunks, start=1):
        details = [f"[{number}] {source_label(chunk)}"]
        chunk_id = display_value(chunk.get("chunk_id")).strip()
        if chunk_id:
            details.append(f"chunk: {chunk_id}")
        if show_scores and chunk.get("@search.score") is not None:
            details.append(f"score: {float(chunk['@search.score']):.6f}")
        print(" | ".join(details))


def main() -> int:
    args = parse_args()
    try:
        settings = load_settings()
        openai_client = make_openai_client(settings)
        index_fields = get_index_fields(settings)
        vector_field = find_vector_field(index_fields, args.vector_field)
        question_vector = create_question_embedding(
            openai_client, settings.embedding_deployment, args.question.strip()
        )
        chunks = retrieve_chunks(
            settings=settings,
            question_vector=question_vector,
            vector_field=vector_field,
            available_fields=index_fields,
            top=args.top,
            document_code=args.document_code,
        )
        if not chunks:
            raise RuntimeError("Vector search returned no policy chunks.")

        answer = generate_answer(
            client=openai_client,
            deployment=settings.chat_deployment,
            question=args.question.strip(),
            context=build_context(chunks),
        )
        print("Answer")
        print("------")
        print(answer)
        print_sources(chunks, args.show_scores)
        return 0
    except Exception as exc:  # Keep CLI errors concise and actionable.
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
