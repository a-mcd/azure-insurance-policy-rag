#!/usr/bin/env python3
"""Convert extracted IP-document JSON into retrieval-ready JSONL chunks.

The input must contain a top-level ``documents`` array. Chunks never cross a
subsection boundary, retain source-page metadata, and include document context
in the text whose tokens are counted and later embedded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import tiktoken
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("Missing dependency. Install it with: pip install tiktoken") from exc


DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_TARGET_TOKENS = 500
DEFAULT_MAX_TOKENS = 700
DEFAULT_OVERLAP_TOKENS = 75


@dataclass(frozen=True)
class Block:
    """One indivisible or pre-split content block."""

    text: str
    content_type: str
    pdf_pages: tuple[int, ...]
    printed_pages: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk extracted insurance product-information JSON for RAG."
    )
    parser.add_argument("input", type=Path, help="Input ip_documentation.json file")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("ip_document_chunks.jsonl")
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    return parser.parse_args()


def get_encoding(model: str) -> Any:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # text-embedding-3 models use cl100k_base. This also makes the script
        # usable with Azure deployment names unknown to tiktoken.
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoding: Any) -> int:
    return len(encoding.encode(text))


def normalise_text(value: str) -> str:
    return re.sub(r"[ \t]+", " ", value.replace("\r\n", "\n")).strip()


def render_table(item: dict[str, Any]) -> str:
    """Render either list-based or dictionary-based table rows as searchable text."""
    columns = [str(column) for column in item.get("columns", [])]
    rendered_rows: list[str] = []

    for row in item.get("rows", []):
        if isinstance(row, dict):
            rendered_rows.append(
                "; ".join(f"{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}" for key, value in row.items())
            )
        elif isinstance(row, list):
            rendered_rows.append(
                "; ".join(
                    f"{columns[index] if index < len(columns) else f'Column {index + 1}'}: {value}"
                    for index, value in enumerate(row)
                )
            )

    return "\n".join(rendered_rows)


def render_item(item: dict[str, Any]) -> str:
    content_type = item.get("content_type", "unknown")
    text = normalise_text(str(item.get("text", "")))

    if content_type == "table":
        text = render_table(item)
    elif content_type == "important_notice":
        label = item.get("label") or "Important"
        title = item.get("title")
        text = " — ".join(str(part) for part in (label, title, text) if part)

    return normalise_text(text)


def split_long_text(text: str, token_limit: int, encoding: Any) -> list[str]:
    """Split an oversized block, preferring paragraph/sentence boundaries."""
    if count_tokens(text, encoding) <= token_limit:
        return [text]

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    pieces: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        candidate = " ".join([*current, sentence])
        if current and count_tokens(candidate, encoding) > token_limit:
            pieces.append(" ".join(current))
            current = []

        if count_tokens(sentence, encoding) <= token_limit:
            current.append(sentence)
            continue

        # Last resort: split one exceptionally long sentence by tokens.
        token_ids = encoding.encode(sentence)
        pieces.extend(
            encoding.decode(token_ids[start : start + token_limit]).strip()
            for start in range(0, len(token_ids), token_limit)
        )

    if current:
        pieces.append(" ".join(current))
    return [piece for piece in pieces if piece]


def context_lines(document: dict[str, Any], section: dict[str, Any], subsection: dict[str, Any]) -> list[str]:
    lines = [
        f"Document: {document.get('document_name', 'Unknown')}",
        f"Document type: {document.get('document_type', 'Unknown')}",
    ]
    if document.get("product"):
        lines.append(f"Product: {document['product']}")
    if document.get("fca_reference_number"):
        lines.append(f"FCA reference number: {document['fca_reference_number']}")
    lines.append(f"Section: {section.get('section_heading', 'Unknown')}")

    headings = subsection.get("subsection_headings") or []
    if headings:
        lines.append(f"Subsection: {' > '.join(map(str, headings))}")
    return lines


def make_blocks(
    subsection: dict[str, Any], body_token_limit: int, encoding: Any
) -> list[Block]:
    blocks: list[Block] = []
    for item in subsection.get("content", []):
        text = render_item(item)
        if not text:
            continue
        for piece in split_long_text(text, body_token_limit, encoding):
            blocks.append(
                Block(
                    text=piece,
                    content_type=str(item.get("content_type", "unknown")),
                    pdf_pages=tuple(item.get("source_pdf_pages") or []),
                    printed_pages=tuple(item.get("source_printed_pages") or []),
                )
            )
    return blocks


def trailing_overlap(blocks: list[Block], limit: int, encoding: Any) -> list[Block]:
    selected: list[Block] = []
    used = 0
    for block in reversed(blocks):
        size = count_tokens(block.text, encoding)
        if selected and used + size > limit:
            break
        if not selected and size > limit:
            continue
        selected.append(block)
        used += size
    return list(reversed(selected))


def unique_sorted(values: Iterable[int]) -> list[int]:
    return sorted(set(values))


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result[:48] or "section"


def build_chunk(
    document: dict[str, Any],
    section: dict[str, Any],
    subsection: dict[str, Any],
    blocks: list[Block],
    sequence: int,
    encoding: Any,
) -> dict[str, Any]:
    body = "\n\n".join(block.text for block in blocks)
    embedding_text = "\n".join(context_lines(document, section, subsection)) + "\n\n" + body
    identity = f"{document.get('document_name')}|{section.get('section_heading')}|{subsection.get('subsection_headings')}|{sequence}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    pdf_pages = unique_sorted(page for block in blocks for page in block.pdf_pages)
    printed_pages = unique_sorted(page for block in blocks for page in block.printed_pages)

    return {
        "chunk_id": f"{slug(str(document.get('document_name', 'document')))}-{digest}",
        "document_name": document.get("document_name"),
        "document_type": document.get("document_type"),
        "product": document.get("product"),
        "fca_reference_number": document.get("fca_reference_number"),
        "section_heading": section.get("section_heading"),
        "subsection_headings": subsection.get("subsection_headings") or [],
        "content_types": sorted(set(block.content_type for block in blocks)),
        "text": body,
        "embedding_text": embedding_text,
        "token_count": count_tokens(embedding_text, encoding),
        "start_pdf_page": min(pdf_pages) if pdf_pages else subsection.get("start_pdf_page"),
        "end_pdf_page": max(pdf_pages) if pdf_pages else subsection.get("end_pdf_page"),
        "start_printed_page": min(printed_pages) if printed_pages else subsection.get("start_printed_page"),
        "end_printed_page": max(printed_pages) if printed_pages else subsection.get("end_printed_page"),
        "source_pdf_pages": pdf_pages,
        "source_printed_pages": printed_pages,
    }


def chunk_subsection(
    document: dict[str, Any],
    section: dict[str, Any],
    subsection: dict[str, Any],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    encoding: Any,
) -> list[dict[str, Any]]:
    context = "\n".join(context_lines(document, section, subsection)) + "\n\n"
    context_tokens = count_tokens(context, encoding)
    body_limit = max_tokens - context_tokens
    if body_limit < 50:
        raise ValueError("Maximum token size is too small for the chunk metadata")

    blocks = make_blocks(subsection, body_limit, encoding)
    chunks: list[dict[str, Any]] = []
    current: list[Block] = []

    for block in blocks:
        candidate = [*current, block]
        candidate_text = context + "\n\n".join(item.text for item in candidate)
        candidate_tokens = count_tokens(candidate_text, encoding)

        should_flush = current and (
            candidate_tokens > max_tokens
            or count_tokens(context + "\n\n".join(item.text for item in current), encoding)
            >= target_tokens
        )
        if should_flush:
            chunks.append(
                build_chunk(document, section, subsection, current, len(chunks) + 1, encoding)
            )
            current = trailing_overlap(current, overlap_tokens, encoding)

        current.append(block)

    if current:
        chunks.append(build_chunk(document, section, subsection, current, len(chunks) + 1, encoding))

    return chunks


def chunk_documents(data: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    documents = data.get("documents")
    if not isinstance(documents, list):
        raise ValueError("Input JSON must contain a top-level 'documents' array")
    if not 0 <= args.overlap_tokens < args.target_tokens <= args.max_tokens:
        raise ValueError("Require: 0 <= overlap < target <= max token counts")

    encoding = get_encoding(args.model)
    chunks: list[dict[str, Any]] = []
    for document in documents:
        for section in document.get("sections", []):
            for subsection in section.get("subsections", []):
                chunks.extend(
                    chunk_subsection(
                        document,
                        section,
                        subsection,
                        args.target_tokens,
                        args.max_tokens,
                        args.overlap_tokens,
                        encoding,
                    )
                )
    return chunks


def main() -> int:
    args = parse_args()
    try:
        with args.input.open(encoding="utf-8") as source:
            data = json.load(source)
        chunks = chunk_documents(data, args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as destination:
            for chunk in chunks:
                destination.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    sizes = [chunk["token_count"] for chunk in chunks]
    print(f"Documents: {len(data['documents'])}")
    print(f"Chunks: {len(chunks)}")
    if sizes:
        print(f"Token range: {min(sizes)}-{max(sizes)}")
        print(f"Average tokens: {sum(sizes) / len(sizes):.1f}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
