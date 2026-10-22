"""Grounded answer generation, source reporting and saving answers."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from azure.search.documents import SearchClient
from openai import OpenAI

from insurance_rag.clients import create_openai_client, create_search_client
from insurance_rag.config import (
    OUTPUTS_DIR,
    ChatSettings,
    load_chat_settings,
    load_environment,
    load_openai_settings,
    load_search_settings,
)
from insurance_rag.retrieval import (
    DEFAULT_SEARCH_MODE,
    create_filter,
    embed_query,
    search,
)

logger = logging.getLogger(__name__)

DEFAULT_TOP = 10


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
"""  # noqa: E501 (prompt text is kept unwrapped)


@dataclass(frozen=True)
class AnswerResult:
    """A generated answer and its token usage."""

    answer: str
    input_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    max_completion_tokens: int


def display_value(value: Any) -> str:
    """Convert a field value to display text, joining lists with commas."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def source_label(chunk: dict[str, Any]) -> str:
    """Build a readable label for a chunk from its title and heading metadata.

    The label lists the document title and code, then the section, subsection
    and semantic headings, skipping empty values and headings already shown
    (ignoring case and a trailing colon). Several subsection headings that
    share a page are separated by commas.
    """
    parts = []
    seen = set()
    for field in (
        "title",
        "document_code",
        "section_heading",
        "subsection_headings",
        "semantic_heading",
    ):
        value = display_value(chunk.get(field)).strip()
        key = value.rstrip(":").strip().casefold()
        if key and key not in seen:
            seen.add(key)
            parts.append(value)
    return " | ".join(parts) or display_value(chunk.get("chunk_id")) or "Policy excerpt"


def page_numbers(chunk: dict[str, Any]) -> tuple[str, str]:
    """Return a chunk's PDF and printed page numbers as comma-separated text."""
    return (
        display_value(chunk.get("source_pdf_pages")).strip(),
        display_value(chunk.get("source_printed_pages")).strip(),
    )


def build_context(chunks: list[dict[str, Any]]) -> str:
    """Format the retrieved chunks as numbered sources for the prompt."""
    sources = []
    for number, chunk in enumerate(chunks, start=1):
        metadata = [f"Source: {source_label(chunk)}"]
        chunk_id = display_value(chunk.get("chunk_id")).strip()
        if chunk_id:
            metadata.append(f"Chunk ID: {chunk_id}")
        pdf_pages, printed_pages = page_numbers(chunk)
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
    chat: ChatSettings,
    question: str,
    context: str,
) -> AnswerResult:
    """Ask the chat model for a cited answer grounded in the supplied context."""
    user_prompt = f"""Question:
{question}

Policy excerpts:
{context}

Answer the question using only these excerpts and include numbered citations."""

    options: dict[str, Any] = {"max_completion_tokens": chat.max_completion_tokens}
    # Non-reasoning models reject reasoning_effort, so it is only sent when set.
    if chat.reasoning_effort is not None:
        options["reasoning_effort"] = chat.reasoning_effort

    response = client.chat.completions.create(
        model=chat.deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        **options,
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
        max_completion_tokens=chat.max_completion_tokens,
    )


def format_token_usage(
    embedding_input_tokens: int,
    answer_result: AnswerResult,
) -> str:
    """Return a report of the token usage for the embedding and chat requests."""
    visible_answer_tokens = max(
        answer_result.completion_tokens - answer_result.reasoning_tokens,
        0,
    )

    return "\n".join(
        [
            "Token usage",
            "-----------",
            f"Question embedding input tokens: {embedding_input_tokens:,}",
            f"Chat input tokens: {answer_result.input_tokens:,}",
            f"Chat completion tokens: {answer_result.completion_tokens:,}",
            f"  Reasoning tokens: {answer_result.reasoning_tokens:,}",
            f"  Visible answer tokens: {visible_answer_tokens:,}",
            f"Chat total tokens: {answer_result.total_tokens:,}",
            f"Maximum completion tokens: {answer_result.max_completion_tokens:,}",
        ]
    )


def source_details(chunk: dict[str, Any], show_scores: bool) -> list[str]:
    """Return the identifying details for one retrieved source."""
    details = [source_label(chunk)]
    chunk_id = display_value(chunk.get("chunk_id")).strip()
    pdf_pages, printed_pages = page_numbers(chunk)

    if chunk_id:
        details.append(f"chunk: {chunk_id}")
    if pdf_pages:
        details.append(f"PDF page(s): {pdf_pages}")
    if printed_pages:
        details.append(f"printed page(s): {printed_pages}")
    if show_scores and chunk.get("score") is not None:
        details.append(f"score: {float(chunk['score']):.6f}")
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


def format_source_group(
    heading: str,
    sources: list[tuple[int, dict[str, Any]]],
    show_scores: bool,
) -> str:
    """Return a heading followed by the numbered sources in the group."""
    lines = [heading, "-" * len(heading)]
    if not sources:
        lines.append("None")
    for number, chunk in sources:
        lines.append(f"[{number}] " + " | ".join(source_details(chunk, show_scores)))
    return "\n".join(lines)


def format_sources(chunks: list[dict[str, Any]], answer: str, show_scores: bool) -> str:
    """Return the sources the answer cited, then those it did not."""
    used, not_used = split_sources(chunks, answer)
    return "\n\n".join(
        [
            format_source_group("Sources used", used, show_scores),
            format_source_group("Sources not used", not_used, show_scores),
        ]
    )


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
    """Save a question and its answer as Markdown in ``outputs/answers``."""
    answers_directory = OUTPUTS_DIR / "answers"
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


@dataclass(frozen=True)
class AnswerPipeline:
    """The clients and deployments needed to answer questions."""

    openai_client: OpenAI
    search_client: SearchClient
    embedding_deployment: str
    chat: ChatSettings


@dataclass(frozen=True)
class QuestionAnswer:
    """A question, its generated answer, the retrieved chunks and token usage."""

    question: str
    result: AnswerResult
    chunks: list[dict[str, Any]]
    embedding_input_tokens: int


def create_answer_pipeline() -> AnswerPipeline:
    """Load settings from ``.env`` and create the clients for answering questions.

    Raises:
        ValueError: If a required environment variable is missing.
    """
    load_environment()
    openai_settings = load_openai_settings()
    chat_settings = load_chat_settings()
    return AnswerPipeline(
        openai_client=create_openai_client(openai_settings),
        search_client=create_search_client(load_search_settings()),
        embedding_deployment=openai_settings.embedding_deployment,
        chat=chat_settings,
    )


def answer_question(
    pipeline: AnswerPipeline,
    question: str,
    top: int = DEFAULT_TOP,
    document_code: str | None = None,
) -> QuestionAnswer:
    """Retrieve chunks for the question with vector search and generate an answer.

    Raises:
        RuntimeError: If no chunks are retrieved or the model returns no answer.
    """
    question = question.strip()
    embedding = embed_query(
        pipeline.openai_client, pipeline.embedding_deployment, question
    )
    chunks = search(
        pipeline.search_client,
        question,
        mode=DEFAULT_SEARCH_MODE,
        top=top,
        filter_expression=create_filter(document_code=document_code),
        query_vector=embedding.vector,
    )
    if not chunks:
        raise RuntimeError("Vector search returned no policy chunks.")
    logger.debug(
        "Retrieved %d chunks: %s",
        len(chunks),
        ", ".join(str(chunk.get("chunk_id")) for chunk in chunks),
    )

    result = generate_answer(
        client=pipeline.openai_client,
        chat=pipeline.chat,
        question=question,
        context=build_context(chunks),
    )
    return QuestionAnswer(
        question=question,
        result=result,
        chunks=chunks,
        embedding_input_tokens=embedding.input_tokens,
    )


def format_answer(answer: QuestionAnswer, show_scores: bool) -> str:
    """Return a report of the answer, its cited and uncited sources, and token usage."""
    return "\n\n".join(
        [
            f"Answer\n------\n{answer.result.answer}",
            format_sources(answer.chunks, answer.result.answer, show_scores),
            format_token_usage(answer.embedding_input_tokens, answer.result),
        ]
    )
