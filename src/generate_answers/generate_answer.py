#!/usr/bin/env python3
"""Generate a grounded answer from chunks stored in Azure AI Search."""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
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
DEFAULT_MAX_COMPLETION_TOKENS = 2000
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

SYSTEM_PROMPT = """You answer questions about home insurance using only the supplied policy excerpts.

Rules:
- Answer the exact question asked and lead with a direct, concise answer in British English.
- Use only the supplied excerpts. Do not use outside knowledge or invent policy terms, limits, exclusions, conditions or citations.
- Prioritise the policy section and insured event that most directly apply to the facts given.
- Do not introduce related cover sections unless they provide a genuine alternative that applies to the question.
- Do not assume an unstated cause, policy option, ownership status or Home Policy Schedule entry. If different facts would produce different outcomes, explain the distinction briefly.
- Clearly distinguish included, optional, policy-level-specific, separate and excluded cover.
- Distinguish Admiral, Gold and Platinum when the evidence supports a comparison.
- Treat monetary figures as maximum limits unless the excerpt explicitly specifies a fixed payment. Say “provides cover up to” or “is within the stated limit”; do not imply that a limit guarantees payment.
- Do not use a limit for one category of property or cover as evidence for another category.
- Do not make a claim decision or promise that a claim will be accepted.
- Include only requirements, consequences, conditions and exclusions that materially affect the answer.
- Interpret explicit coverage tables, policy-level markers, inclusions and exclusions when supplied.
- Do not infer that cover is excluded or unavailable merely because it is absent from an excerpt.
- If the excerpts do not contain enough evidence, state precisely what cannot be established. Do this only after checking all excerpts for a direct answer, limit, inclusion, exclusion, eligibility rule or policy-level marker.
- Cite every material policy claim using its source number, for example [1], immediately after the supported claim.
- Use only supplied source numbers and never cite an excerpt that does not directly support the claim.
- Do not add a Sources, Citations or Cite section; the application creates the source lists separately.
- Do not repeat the same citation beside a claim.
- Keep the answer proportionate to the question and do not repeat the conclusion in separate sections.
- Before returning the answer, check that the direct answer is consistent with the explanation and that every policy-level amount has direct supporting evidence.
- Apply each inclusion, exclusion, condition, definition and limit only to the policy section it governs. Do not apply a rule from one section to another unless the excerpts explicitly state that it applies more broadly.
- When more than one section could apply, evaluate each section separately. Do not let an exclusion under one section override cover under another independent section.
- Before returning the answer, check that the first sentence does not contradict any later statement.
- Apply each inclusion, exclusion, condition, definition and limit only to the policy section it governs. Do not apply a rule from one section to another unless the excerpts explicitly state that it applies more broadly.
- When more than one section could apply, evaluate each section separately. Do not let an exclusion under one section override cover under another independent section.
- Before returning the answer, check that the first sentence does not contradict any later statement.
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


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    input_tokens: int


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    input_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int


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
) -> EmbeddingResult:
    response = client.embeddings.create(model=deployment, input=[question])
    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "input_tokens", 0)
    return EmbeddingResult(
        vector=response.data[0].embedding,
        input_tokens=input_tokens,
    )


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


def clean_generated_answer(answer: str) -> str:
    """Remove citation artefacts and deduplicate adjacent numbered citations."""
    artefact_line = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:[-*]\s*)?"
        r"(?:cite|citations?|sources?)\s*:.*$",
        flags=re.IGNORECASE,
    )
    cleaned_lines = [
        line for line in answer.strip().splitlines() if not artefact_line.match(line)
    ]
    cleaned = "\n".join(cleaned_lines)

    citation_run = re.compile(
        r"\[(?:\d+(?:\s*,\s*\d+)*)\]"
        r"(?:[ \t]*\[(?:\d+(?:\s*,\s*\d+)*)\])*"
    )

    def deduplicate_citation_run(match: re.Match[str]) -> str:
        numbers: list[int] = []
        for value in re.findall(r"\d+", match.group(0)):
            number = int(value)
            if number not in numbers:
                numbers.append(number)
        return "[" + ", ".join(str(number) for number in numbers) + "]"

    cleaned = citation_run.sub(deduplicate_citation_run, cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def generate_answer(
    client: OpenAI,
    deployment: str,
    question: str,
    context: str,
) -> AnswerResult:
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
    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
    completion_details = (
        getattr(usage, "completion_tokens_details", None) if usage else None
    )
    reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0

    return AnswerResult(
        answer=clean_generated_answer(answer),
        input_tokens=input_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
    )


def print_token_usage(
    embedding_input_tokens: int,
    answer_result: AnswerResult,
) -> None:
    visible_answer_tokens = max(
        answer_result.completion_tokens - answer_result.reasoning_tokens,
        0,
    )

    print("\nToken usage")
    print("-----------")
    print(f"Question embedding input tokens: {embedding_input_tokens:,}")
    print(f"Chat input tokens: {answer_result.input_tokens:,}")
    print(f"Chat completion tokens: {answer_result.completion_tokens:,}")
    print(f"  Reasoning tokens: {answer_result.reasoning_tokens:,}")
    print(f"  Visible answer tokens: {visible_answer_tokens:,}")
    print(f"Chat total tokens: {answer_result.total_tokens:,}")
    print(f"Maximum completion tokens: {DEFAULT_MAX_COMPLETION_TOKENS:,}")


def source_details(chunk: dict[str, Any], show_scores: bool) -> list[str]:
    """Return the identifying details for one retrieved source."""
    details = [source_label(chunk)]
    chunk_id = display_value(chunk.get("chunk_id")).strip()
    pdf_pages = display_value(
        chunk.get("source_pdf_pages") or chunk.get("pdf_page")
    ).strip()
    printed_pages = display_value(
        chunk.get("source_printed_pages") or chunk.get("printed_page")
    ).strip()

    if chunk_id:
        details.append(f"chunk: {chunk_id}")
    if pdf_pages:
        details.append(f"PDF page(s): {pdf_pages}")
    if printed_pages:
        details.append(f"printed page(s): {printed_pages}")
    if show_scores and chunk.get("@search.score") is not None:
        details.append(f"score: {float(chunk['@search.score']):.6f}")
    return details


def cited_source_numbers(answer: str, source_count: int) -> set[int]:
    """Return valid source numbers cited in brackets in the generated answer."""
    cited_numbers: set[int] = set()
    for citation_group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        for value in citation_group.split(","):
            number = int(value.strip())
            if 1 <= number <= source_count:
                cited_numbers.add(number)
    return cited_numbers


def split_sources(
    chunks: list[dict[str, Any]], answer: str
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    """Split retrieved chunks into cited and uncited lists, retaining source numbers."""
    cited_numbers = cited_source_numbers(answer, len(chunks))
    numbered_chunks = list(enumerate(chunks, start=1))
    used = [item for item in numbered_chunks if item[0] in cited_numbers]
    not_used = [item for item in numbered_chunks if item[0] not in cited_numbers]
    return used, not_used


def print_source_group(
    heading: str,
    sources: list[tuple[int, dict[str, Any]]],
    show_scores: bool,
) -> None:
    print(f"\n{heading}")
    print("-" * len(heading))
    if not sources:
        print("None")
        return
    for number, chunk in sources:
        print(f"[{number}] " + " | ".join(source_details(chunk, show_scores)))


def print_sources(
    chunks: list[dict[str, Any]], answer: str, show_scores: bool
) -> None:
    used, not_used = split_sources(chunks, answer)
    print_source_group("Sources used", used, show_scores)
    print_source_group("Sources not used", not_used, show_scores)


def question_filename(question: str, max_length: int = 80) -> str:
    """Return a readable, filesystem-safe filename stem for a question."""
    normalised = unicodedata.normalize("NFKD", question)
    ascii_question = normalised.encode("ascii", "ignore").decode("ascii")
    ascii_question = re.sub(r"(?<=\d),(?=\d)", "", ascii_question)
    filename = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_question).strip("-").lower()
    filename = filename[:max_length].rstrip("-")
    return filename or "question"


def save_answer(
    question: str,
    answer: str,
    chunks: list[dict[str, Any]],
    show_scores: bool,
) -> Path:
    """Save a question and its answer in the script's answers directory."""
    answers_directory = Path(__file__).resolve().parent / "answers"
    answers_directory.mkdir(parents=True, exist_ok=True)

    stem = question_filename(question)
    output_path = answers_directory / f"{stem}.md"
    copy_number = 2
    while output_path.exists():
        output_path = answers_directory / f"{stem}-{copy_number}.md"
        copy_number += 1

    used, not_used = split_sources(chunks, answer)

    def markdown_source_lines(
        sources: list[tuple[int, dict[str, Any]]],
    ) -> str:
        if not sources:
            return "None."
        return "\n".join(
            f"{number}. " + " | ".join(source_details(chunk, show_scores))
            for number, chunk in sources
        )

    content = (
        f"# Question\n\n{question}\n\n"
        f"# Answer\n\n{answer}\n\n"
        f"# Sources used\n\n{markdown_source_lines(used)}\n\n"
        f"# Sources not used\n\n{markdown_source_lines(not_used)}\n"
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    try:
        settings = load_settings()
        openai_client = make_openai_client(settings)
        index_fields = get_index_fields(settings)
        vector_field = find_vector_field(index_fields, args.vector_field)
        embedding_result = create_question_embedding(
            openai_client, settings.embedding_deployment, args.question.strip()
        )
        chunks = retrieve_chunks(
            settings=settings,
            question_vector=embedding_result.vector,
            vector_field=vector_field,
            available_fields=index_fields,
            top=args.top,
            document_code=args.document_code,
        )
        if not chunks:
            raise RuntimeError("Vector search returned no policy chunks.")

        answer_result = generate_answer(
            client=openai_client,
            deployment=settings.chat_deployment,
            question=args.question.strip(),
            context=build_context(chunks),
        )
        output_path = save_answer(
            args.question.strip(),
            answer_result.answer,
            chunks,
            args.show_scores,
        )
        print("Answer")
        print("------")
        print(answer_result.answer)
        print_sources(chunks, answer_result.answer, args.show_scores)
        print_token_usage(embedding_result.input_tokens, answer_result)
        print(f"\nAnswer saved to: {output_path}")
        return 0
    except Exception as exc:  # Keep CLI errors concise and actionable.
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
