#!/usr/bin/env python3
"""Create retrieval-ready JSONL chunks from extracted HH policy booklets.

The chunker understands the ``hh_documents.json`` hierarchy and treats
headings, prose groups, important notices, and individual table rows as
semantic units. It retains source pages and emits the exact context-rich text
to embed in the ``embedding_text`` field.
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
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency. Install it with: pip install tiktoken") from exc


DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_TARGET_TOKENS = 500
DEFAULT_MAX_TOKENS = 700
DEFAULT_OVERLAP_TOKENS = 75


@dataclass(frozen=True)
class Unit:
    """A semantic unit used to construct chunks."""

    text: str
    content_types: tuple[str, ...]
    headings: tuple[str, ...]
    pdf_pages: tuple[int, ...]
    printed_pages: tuple[int, ...]
    table_types: tuple[str, ...] = ()
    boundary_before: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk extracted home-insurance policy JSON for RAG."
    )
    parser.add_argument("input", type=Path, help="Input hh_documents.json file")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("hh_document_chunks.jsonl")
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
        # Also supports Azure deployment names that tiktoken does not recognise.
        return tiktoken.get_encoding("cl100k_base")


def token_count(text: str, encoding: Any) -> int:
    return len(encoding.encode(text))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def humanise_key(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


def render_value(value: Any, indent: int = 0) -> list[str]:
    """Render nested table values without discarding their field names."""
    prefix = "  " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, nested in value.items():
            if key in {"source_pdf_pages", "source_printed_pages"}:
                continue
            if isinstance(nested, (dict, list)):
                lines.append(f"{prefix}{humanise_key(str(key))}:")
                lines.extend(render_value(nested, indent + 1))
            else:
                lines.append(f"{prefix}{humanise_key(str(key))}: {clean_text(nested)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                nested_lines = render_value(item, indent + 1)
                if nested_lines:
                    lines.append(f"{prefix}• {nested_lines[0].lstrip()}")
                    lines.extend(nested_lines[1:])
            else:
                lines.append(f"{prefix}• {clean_text(item)}")
        return lines
    return [f"{prefix}{clean_text(value)}"]


def render_table_row(row: Any, columns: list[str], table_type: str) -> str:
    lines = [f"Table type: {humanise_key(table_type)}"]
    if isinstance(row, dict):
        lines.extend(render_value(row))
    elif isinstance(row, list):
        for index, value in enumerate(row):
            column = columns[index] if index < len(columns) else f"column_{index + 1}"
            lines.append(f"{humanise_key(column)}: {clean_text(value)}")
    else:
        lines.append(clean_text(row))
    return "\n".join(line for line in lines if line.strip())


def pages_from(item: dict[str, Any], row: Any = None) -> tuple[tuple[int, ...], tuple[int, ...]]:
    row_dict = row if isinstance(row, dict) else {}
    pdf = tuple(row_dict.get("source_pdf_pages") or item.get("source_pdf_pages") or [])
    printed = tuple(
        row_dict.get("source_printed_pages") or item.get("source_printed_pages") or []
    )
    return pdf, printed


def split_text(text: str, limit: int, encoding: Any) -> list[str]:
    if token_count(text, encoding) <= limit:
        return [text]

    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n(?=•)", text) if part.strip()]
    output: list[str] = []
    current: list[str] = []
    for part in parts:
        candidate = "\n".join([*current, part])
        if current and token_count(candidate, encoding) > limit:
            output.append("\n".join(current))
            current = []
        if token_count(part, encoding) <= limit:
            current.append(part)
        else:
            ids = encoding.encode(part)
            output.extend(
                encoding.decode(ids[start : start + limit]).strip()
                for start in range(0, len(ids), limit)
            )
    if current:
        output.append("\n".join(current))
    return [part for part in output if part]


def unit_parts(unit: Unit, limit: int, encoding: Any) -> list[Unit]:
    parts = split_text(unit.text, limit, encoding)
    return [
        Unit(
            text=part,
            content_types=unit.content_types,
            headings=unit.headings,
            pdf_pages=unit.pdf_pages,
            printed_pages=unit.printed_pages,
            table_types=unit.table_types,
            boundary_before=unit.boundary_before and index == 0,
        )
        for index, part in enumerate(parts)
    ]


def make_units(subsection: dict[str, Any], body_limit: int, encoding: Any) -> list[Unit]:
    """Group headings with following prose and make each table row retrievable."""
    units: list[Unit] = []
    heading_path: dict[int, str] = {}
    prose: list[str] = []
    prose_types: set[str] = set()
    prose_pdf: set[int] = set()
    prose_printed: set[int] = set()
    prose_boundary_before = False

    def active_headings() -> tuple[str, ...]:
        return tuple(heading_path[level] for level in sorted(heading_path))

    def flush_prose() -> None:
        nonlocal prose_boundary_before
        if not prose:
            return
        text = "\n\n".join(prose)
        unit = Unit(
            text=text,
            content_types=tuple(sorted(prose_types)),
            headings=active_headings(),
            pdf_pages=tuple(sorted(prose_pdf)),
            printed_pages=tuple(sorted(prose_printed)),
            boundary_before=prose_boundary_before,
        )
        units.extend(unit_parts(unit, body_limit, encoding))
        prose.clear()
        prose_types.clear()
        prose_pdf.clear()
        prose_printed.clear()
        prose_boundary_before = False

    for item in subsection.get("content", []):
        kind = str(item.get("content_type", "unknown"))
        pdf, printed = pages_from(item)

        if kind == "heading":
            flush_prose()
            level = int(item.get("heading_level", 3))
            for old_level in [value for value in heading_path if value >= level]:
                del heading_path[old_level]
            heading = clean_text(item.get("text"))
            if heading:
                prose_boundary_before = level <= 3
                heading_path[level] = heading
                prose.append(heading)
                prose_types.add("heading")
                prose_pdf.update(pdf)
                prose_printed.update(printed)
            continue

        if kind == "table":
            flush_prose()
            table_type = str(item.get("table_type", "generic"))
            columns = [str(value) for value in item.get("columns", [])]
            row_units: list[Unit] = []
            for row in item.get("rows", []):
                row_pdf, row_printed = pages_from(item, row)
                row_units.append(Unit(
                    text=render_table_row(row, columns, table_type),
                    content_types=("table",),
                    headings=active_headings(),
                    pdf_pages=row_pdf,
                    printed_pages=row_printed,
                    table_types=(table_type,),
                ))

            whole_table_text = "\n\n".join(row.text for row in row_units)
            if row_units and token_count(whole_table_text, encoding) <= body_limit:
                units.append(Unit(
                    text=whole_table_text,
                    content_types=("table",),
                    headings=active_headings(),
                    pdf_pages=tuple(sorted({page for row in row_units for page in row.pdf_pages})),
                    printed_pages=tuple(sorted({page for row in row_units for page in row.printed_pages})),
                    table_types=(table_type,),
                ))
            else:
                for row_unit in row_units:
                    units.extend(unit_parts(row_unit, body_limit, encoding))
            continue

        if kind == "important_notice":
            flush_prose()
            notice = " — ".join(
                clean_text(value)
                for value in (item.get("label") or "Important", item.get("title"), item.get("text"))
                if value
            )
            unit = Unit(
                text=notice,
                content_types=("important_notice",),
                headings=active_headings(),
                pdf_pages=pdf,
                printed_pages=printed,
            )
            units.extend(unit_parts(unit, body_limit, encoding))
            continue

        text = clean_text(item.get("text"))
        if text:
            prose.append(text)
            prose_types.add(kind)
            prose_pdf.update(pdf)
            prose_printed.update(printed)

    flush_prose()
    return units


def document_context(
    document: dict[str, Any], section: dict[str, Any], subsection: dict[str, Any]
) -> str:
    lines = [
        f"Document: {document.get('document_name', 'Unknown')}",
        f"Title: {document.get('title', 'Unknown')}",
        f"Document type: {document.get('document_type', 'Unknown')}",
    ]
    if document.get("document_code"):
        lines.append(f"Document code: {document['document_code']}")
    lines.append(f"Section: {section.get('section_heading', 'Unknown')}")
    subsection_headings = subsection.get("subsection_headings") or []
    if subsection_headings:
        lines.append(f"Subsection: {' > '.join(map(str, subsection_headings))}")
    return "\n".join(lines)


def unique_sorted(values: Iterable[int]) -> list[int]:
    return sorted(set(values))


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:48] or "document"


def build_chunk(
    document: dict[str, Any],
    section: dict[str, Any],
    subsection: dict[str, Any],
    units: list[Unit],
    sequence: int,
    encoding: Any,
) -> dict[str, Any]:
    body = "\n\n".join(unit.text for unit in units)
    embedding_text = f"{document_context(document, section, subsection)}\n\n{body}"
    pdf_pages = unique_sorted(page for unit in units for page in unit.pdf_pages)
    printed_pages = unique_sorted(page for unit in units for page in unit.printed_pages)
    identity = "|".join(
        [
            str(document.get("document_name")),
            str(section.get("section_heading")),
            str(subsection.get("subsection_headings")),
            str(sequence),
        ]
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return {
        "chunk_id": f"{slug(str(document.get('document_name', 'document')))}-{digest}",
        "document_name": document.get("document_name"),
        "document_type": document.get("document_type"),
        "document_code": document.get("document_code"),
        "title": document.get("title"),
        "sha256": document.get("sha256"),
        "section_heading": section.get("section_heading"),
        "subsection_headings": subsection.get("subsection_headings") or [],
        "content_headings": list(dict.fromkeys(h for unit in units for h in unit.headings)),
        "content_types": sorted(set(value for unit in units for value in unit.content_types)),
        "table_types": sorted(set(value for unit in units for value in unit.table_types)),
        "text": body,
        "embedding_text": embedding_text,
        "token_count": token_count(embedding_text, encoding),
        "start_pdf_page": min(pdf_pages) if pdf_pages else subsection.get("start_pdf_page"),
        "end_pdf_page": max(pdf_pages) if pdf_pages else subsection.get("end_pdf_page"),
        "start_printed_page": min(printed_pages) if printed_pages else subsection.get("start_printed_page"),
        "end_printed_page": max(printed_pages) if printed_pages else subsection.get("end_printed_page"),
        "source_pdf_pages": pdf_pages,
        "source_printed_pages": printed_pages,
    }


def overlap_units(units: list[Unit], limit: int, encoding: Any) -> list[Unit]:
    selected: list[Unit] = []
    used = 0
    for unit in reversed(units):
        size = token_count(unit.text, encoding)
        if used + size > limit:
            break
        selected.append(unit)
        used += size
    return list(reversed(selected))


def chunk_subsection(
    document: dict[str, Any],
    section: dict[str, Any],
    subsection: dict[str, Any],
    target: int,
    maximum: int,
    overlap: int,
    encoding: Any,
) -> list[dict[str, Any]]:
    context = document_context(document, section, subsection) + "\n\n"
    body_limit = maximum - token_count(context, encoding)
    if body_limit < 50:
        raise ValueError("Maximum token size is too small for the metadata context")

    units = make_units(subsection, body_limit, encoding)
    chunks: list[dict[str, Any]] = []
    current: list[Unit] = []

    for unit in units:
        if unit.boundary_before and current:
            chunks.append(
                build_chunk(document, section, subsection, current, len(chunks) + 1, encoding)
            )
            current = []

        candidate = [*current, unit]
        candidate_size = token_count(
            context + "\n\n".join(value.text for value in candidate), encoding
        )
        current_size = token_count(
            context + "\n\n".join(value.text for value in current), encoding
        ) if current else 0

        if current and (candidate_size > maximum or current_size >= target):
            chunks.append(
                build_chunk(document, section, subsection, current, len(chunks) + 1, encoding)
            )
            current = overlap_units(current, overlap, encoding)
            while current and token_count(
                context + "\n\n".join(value.text for value in [*current, unit]), encoding
            ) > maximum:
                current.pop(0)
        current.append(unit)

    if current:
        chunks.append(build_chunk(document, section, subsection, current, len(chunks) + 1, encoding))
    return chunks


def create_chunks(data: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
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
        chunks = create_chunks(data, args)
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
        print(f"Table chunks: {sum(bool(chunk['table_types']) for chunk in chunks)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
