#!/usr/bin/env python3
"""Create semantic JSONL chunks from the extracted insurance-document JSON.

The input must use the shared ``documents -> sections -> subsections -> content``
schema produced by the IP and HH PDF extractors. Chunk boundaries occur at a
section, subsection, or level-three heading. Tables are isolated and normally
emitted as one complete table row per chunk. Small policy-comparison tables
that need their full context are retained as complete tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import tiktoken
except ImportError:  # pragma: no cover - fallback is exercised only without dependency
    tiktoken = None


DEFAULT_MAX_TOKENS = 500

COVER_STATUS_LABELS = {
    "hec": "Home Emergency cover (HEC)",
    "hex": "Home Emergency Extra cover (HEX)",
}

HOME_EMERGENCY_DEFINITION_TERMS = {
    "associated home cover",
    "emergency",
    "heating system",
    "home",
    "our contractor",
    "period of insurance",
    "reimbursement basis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantically chunk one extracted IP or HH JSON file."
    )
    parser.add_argument("input", type=Path, help="Extracted JSON input file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="JSONL output file")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--encoding", default="cl100k_base", help="Tiktoken encoding name")
    return parser.parse_args()


def normalise_soft_line_wraps(value: Any) -> str:
    """Join PDF line wraps while preserving paragraphs and list boundaries."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    output: list[str] = []
    current = ""

    def flush_line() -> None:
        nonlocal current
        if current:
            output.append(current)
            current = ""

    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            flush_line()
            if output and output[-1] != "":
                output.append("")
            continue

        starts_list_item = bool(
            re.match(r"^(?:[•*-]|\d+[.)]|[A-Za-z][.)])\s+", line)
        )
        if starts_list_item:
            flush_line()
            current = line
        elif current:
            # PDF extraction can leave a hyphen at the end of a visual line,
            # for example ``hot-\nwater``. Preserve the hyphen but do not add
            # an artificial space after it.
            separator = "" if current.endswith("-") else " "
            current = f"{current}{separator}{line}"
        else:
            current = line

    flush_line()
    while output and output[-1] == "":
        output.pop()
    return "\n".join(output).strip()


def clean_text(value: Any) -> str:
    return normalise_soft_line_wraps(value)


def unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def humanise(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def display_label(value: Any) -> str:
    key = str(value).strip().casefold()
    return COVER_STATUS_LABELS.get(key, humanise(str(value)))


def render_item(item: dict[str, Any]) -> str:
    content_type = item.get("content_type")
    text = clean_text(item.get("text"))

    if content_type == "heading":
        return text
    if content_type == "list_item":
        marker = clean_text(item.get("list_marker")) or "•"
        return f"{marker} {text}".strip()
    if content_type == "important_notice":
        heading = clean_text(item.get("title") or item.get("label"))
        return "\n".join(part for part in (heading, text) if part)
    return text


def table_header_lines(table: dict[str, Any]) -> list[str]:
    lines = [f"Table: {humanise(str(table.get('table_type', 'table')))}"]
    columns = table.get("columns")
    if columns:
        lines.append("Columns: " + " | ".join(display_label(c) for c in columns))
    for group in table.get("permission_groups", []):
        heading = clean_text(group.get("heading"))
        permissions = ", ".join(display_label(p) for p in group.get("permissions", []))
        lines.append(f"{heading}: {permissions}")
    return lines


def render_table_row(row: Any, columns: list[Any] | None = None) -> str:
    if isinstance(row, list):
        labels = columns or list(range(1, len(row) + 1))
        return " | ".join(
            f"{display_label(label)}: {clean_text(value)}"
            for label, value in zip(labels, row)
        )

    if not isinstance(row, dict):
        return clean_text(row)

    parts: list[str] = []
    for key, value in row.items():
        if key in {"source_pdf_pages", "source_printed_pages"}:
            continue
        label = humanise(str(key))
        if key == "cover_status" and isinstance(value, dict):
            status = ", ".join(
                f"{display_label(k)}: {humanise(str(v))}"
                for k, v in value.items()
            )
            parts.append(f"Cover status: {status}")
        elif key == "permissions" and isinstance(value, dict):
            permissions = ", ".join(
                f"{humanise(str(k))}: {'Allowed' if v else 'Not allowed'}"
                for k, v in value.items()
            )
            parts.append(f"Permissions: {permissions}")
        elif isinstance(value, (dict, list)):
            parts.append(f"{label}: {json.dumps(value, ensure_ascii=False)}")
        else:
            parts.append(f"{label}: {clean_text(value)}")
    return " | ".join(parts)


def metadata_prefix(meta: dict[str, Any]) -> str:
    labels = (
        ("Title", meta.get("title")),
        ("Document type", meta.get("document_type")),
        ("Document code", meta.get("document_code")),
        ("Section", meta.get("section_heading")),
        ("Subsection", "; ".join(meta.get("subsection_headings") or [])),
        ("Semantic heading", meta.get("semantic_heading")),
    )
    return "\n".join(f"{label}: {value}" for label, value in labels if value)


class Chunker:
    def __init__(self, encoding_name: str, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("--max-tokens must be greater than zero")
        if tiktoken is None:
            print(
                "Warning: tiktoken is not installed; using a conservative UTF-8 "
                "estimate. Install tiktoken for model-aligned token counts.",
                file=sys.stderr,
            )
            self.encoding = ApproximateEncoding()
        else:
            self.encoding = tiktoken.get_encoding(encoding_name)
        self.max_tokens = max_tokens

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def embedding_text(self, meta: dict[str, Any], text: str) -> str:
        prefix = metadata_prefix(meta)
        return f"{prefix}\n\n{text}" if prefix else text

    def fits(self, meta: dict[str, Any], text: str) -> bool:
        return self.count(self.embedding_text(meta, text)) <= self.max_tokens

    def split_long_text(self, meta: dict[str, Any], text: str) -> list[str]:
        """Split an indivisible long item using sentence boundaries, then tokens."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", text) if s.strip()]
        pieces: list[str] = []
        current = ""
        for sentence in sentences or [text]:
            candidate = f"{current}\n\n{sentence}".strip()
            if current and not self.fits(meta, candidate):
                pieces.append(current)
                current = sentence
            else:
                current = candidate
            if current and not self.fits(meta, current):
                prefix_tokens = self.count(self.embedding_text(meta, ""))
                allowance = max(1, self.max_tokens - prefix_tokens)
                tokens = self.encoding.encode(current)
                pieces.extend(self.encoding.decode(tokens[i : i + allowance]).strip() for i in range(0, len(tokens), allowance))
                current = ""
        if current:
            pieces.append(current)
        return [piece for piece in pieces if piece]


class ApproximateEncoding:
    """Reversible, conservative fallback of one token per three UTF-8 bytes."""

    def encode(self, text: str) -> list[bytes]:
        raw = text.encode("utf-8")
        return [raw[index : index + 3] for index in range(0, len(raw), 3)]

    def decode(self, tokens: list[bytes]) -> str:
        return b"".join(tokens).decode("utf-8", errors="ignore")


def item_pages(item: dict[str, Any], key: str) -> list[int]:
    return [int(page) for page in item.get(key, [])]


def base_metadata(document: dict[str, Any], section: str, subsection: list[str]) -> dict[str, Any]:
    return {
        "document_name": document.get("document_name"),
        "document_type": document.get("document_type"),
        "document_code": document.get("document_code"),
        "title": document.get("title"),
        "document_sha256": document.get("sha256"),
        "fca_reference_number": document.get("fca_reference_number"),
        "section_heading": section,
        "subsection_headings": subsection,
        "semantic_heading": None,
        "semantic_heading_level": None,
    }


def split_ip_section_paragraphs(
    document: dict[str, Any], section_heading: str
) -> bool:
    """Return whether paragraphs in this IP section are standalone chunks."""
    if clean_text(document.get("document_type")).casefold() != (
        "insurance_product_information"
    ):
        return False

    normalised_heading = re.sub(
        r"[^a-z]+", " ", clean_text(section_heading).casefold()
    ).strip()
    return normalised_heading in {
        "what is insured",
        "what is not insured",
        "are there any restrictions on cover",
        "where am i covered",
    }


def is_home_emergency_definitions_heading(value: Any) -> bool:
    """Return whether a heading introduces the Home Emergency glossary."""
    normalised = re.sub(
        r"[^a-z]+", " ", clean_text(value).casefold()
    ).strip()
    return normalised == (
        "home emergency cover and home emergency extra cover definitions"
    )


def is_home_emergency_cover_heading(value: Any) -> bool:
    """Return whether a heading introduces the HEC/HEX overview and table."""
    normalised = re.sub(
        r"[^a-z]+", " ", clean_text(value).casefold()
    ).strip()
    return normalised == "home emergency cover and home emergency extra cover"


def is_home_emergency_definition_term(value: Any) -> bool:
    """Recognise definition labels even when extraction marks them as text."""
    normalised = re.sub(
        r"[^a-z]+", " ", clean_text(value).casefold()
    ).strip()
    return normalised in HOME_EMERGENCY_DEFINITION_TERMS


def make_draft(meta: dict[str, Any], text: str, items: list[dict[str, Any]], chunk_type: str = "content") -> dict[str, Any]:
    pdf_pages = unique(page for item in items for page in item_pages(item, "source_pdf_pages"))
    printed_pages = unique(page for item in items for page in item_pages(item, "source_printed_pages"))
    orders = [int(item["order"]) for item in items if item.get("order") is not None]
    return {
        **meta,
        "chunk_type": chunk_type,
        "text": text.strip(),
        "source_pdf_pages": sorted(pdf_pages),
        "source_printed_pages": sorted(printed_pages),
        "start_pdf_page": min(pdf_pages) if pdf_pages else None,
        "end_pdf_page": max(pdf_pages) if pdf_pages else None,
        "start_printed_page": min(printed_pages) if printed_pages else None,
        "end_printed_page": max(printed_pages) if printed_pages else None,
        "source_start_order": min(orders) if orders else None,
        "source_end_order": max(orders) if orders else None,
    }


def split_table(chunker: Chunker, meta: dict[str, Any], table: dict[str, Any]) -> list[dict[str, Any]]:
    """Emit one self-contained chunk for every complete table row.

    The table heading and column labels are repeated in every chunk. A single
    long row remains intact even when the chunk exceeds ``max_tokens``.
    """
    headers = table_header_lines(table)
    rows = table.get("rows", [])

    # This is a small comparison table whose rows only make full sense when
    # Admiral, Gold and Platinum can be considered together. Keeping it whole
    # also allows one retrieved chunk to answer cross-policy questions.
    if table.get("table_type") == "emergency_cover_by_level" and rows:
        row_texts = [
            render_table_row(row, table.get("columns")) for row in rows
        ]
        text = "\n\n".join([*headers, *row_texts])
        provenance_items = [table, *(row for row in rows if isinstance(row, dict))]
        draft = make_draft(meta, text, provenance_items, "table")
        draft.update(
            {
                "table_type": table.get("table_type"),
                "table_is_split": False,
                "table_chunk_index": 1,
                "table_chunk_count": 1,
                "table_row_start": 1,
                "table_row_end": len(rows),
                "contains_complete_rows": True,
                "contains_complete_table": True,
            }
        )
        return [draft]

    groups: list[tuple[int | None, Any | None, str | None]] = [
        (index, row, render_table_row(row, table.get("columns")))
        for index, row in enumerate(rows)
    ]
    if not groups:
        # Retain an explicitly extracted empty table as a header-only chunk.
        groups.append((None, None, None))

    drafts: list[dict[str, Any]] = []
    for group_index, (row_index, row, row_text) in enumerate(groups, start=1):
        row_pdf = item_pages(row, "source_pdf_pages") if isinstance(row, dict) else []
        row_printed = item_pages(row, "source_printed_pages") if isinstance(row, dict) else []
        row_heading = (
            clean_text(row.get("row_heading"))
            if isinstance(row, dict)
            else ""
        )
        row_meta = dict(meta)
        if row_heading:
            # Preserve meaningful parent context and append the row-specific
            # subject. Generic introductory headings describe how to read the
            # table rather than what the row covers, so use the human-readable
            # table type for those instead.
            parent_heading = clean_text(meta.get("semantic_heading"))
            if (
                not parent_heading
                or parent_heading.casefold()
                in {"how to read this document", "what we’ll not pay", "what we'll not pay"}
            ):
                parent_heading = humanise(
                    str(table.get("table_type", "table"))
                )
            combined_heading = f"{parent_heading} - {row_heading}"
            row_meta.update(
                {
                    "semantic_heading": combined_heading,
                    "semantic_heading_level": 4,
                }
            )
        item = dict(table)
        if row_pdf:
            item["source_pdf_pages"] = unique(row_pdf)
        if row_printed:
            item["source_printed_pages"] = unique(row_printed)
        text = "\n\n".join(headers + ([row_text] if row_text else []))
        draft = make_draft(row_meta, text, [item], "table")
        draft.update(
            {
                "table_type": table.get("table_type"),
                "table_is_split": len(rows) > 1,
                "table_chunk_index": group_index,
                "table_chunk_count": len(groups),
                "table_row_start": row_index + 1 if row_index is not None else None,
                "table_row_end": row_index + 1 if row_index is not None else None,
                "contains_complete_rows": True,
                "contains_complete_table": len(rows) <= 1,
            }
        )
        drafts.append(draft)
    return drafts


def chunk_subsection(chunker: Chunker, document: dict[str, Any], section: str, subsection: dict[str, Any]) -> list[dict[str, Any]]:
    subsection_headings = list(subsection.get("subsection_headings") or [])
    is_definitions_subsection = any(
        clean_text(heading).casefold() == "definitions"
        for heading in subsection_headings
    )
    base = base_metadata(document, section, subsection_headings)
    paragraphs_are_atomic = split_ip_section_paragraphs(document, section)
    drafts: list[dict[str, Any]] = []
    current_items: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_meta = dict(base)
    in_home_emergency_definitions = False

    def flush() -> None:
        nonlocal current_items, current_parts
        if current_parts:
            drafts.append(make_draft(current_meta, "\n\n".join(current_parts), current_items))
        current_items = []
        current_parts = []

    def add_oversized_level_four_block(block_items: list[dict[str, Any]]) -> None:
        """Split an oversized level-four block only at level-five boundaries."""
        parent_item = block_items[0]
        parent_heading = render_item(parent_item)
        body_items = block_items[1:]
        level_five_segments: list[list[dict[str, Any]]] = []
        segment: list[dict[str, Any]] = []

        for block_item in body_items:
            if (
                block_item.get("content_type") == "heading"
                and block_item.get("heading_level") == 5
            ):
                if segment:
                    level_five_segments.append(segment)
                segment = [block_item]
            else:
                segment.append(block_item)
        if segment:
            level_five_segments.append(segment)

        # Without a level-five boundary there is no safe structural split.
        if not any(
            item.get("content_type") == "heading" and item.get("heading_level") == 5
            for item in body_items
        ):
            text = "\n\n".join(
                part for part in (render_item(item) for item in block_items) if part
            )
            drafts.append(make_draft(current_meta, text, block_items))
            return

        def group_text(items: list[dict[str, Any]]) -> str:
            body = "\n\n".join(
                part for part in (render_item(item) for item in items) if part
            )
            return f"{parent_heading}\n\n{body}" if body else parent_heading

        grouped_segments: list[list[dict[str, Any]]] = []
        group: list[dict[str, Any]] = []
        for level_five_segment in level_five_segments:
            candidate = [*group, *level_five_segment]
            if group and not chunker.fits(current_meta, group_text(candidate)):
                grouped_segments.append(group)
                group = list(level_five_segment)
            else:
                group = candidate
        if group:
            grouped_segments.append(group)

        for group_index, grouped_items in enumerate(grouped_segments):
            provenance_items = (
                [parent_item, *grouped_items]
                if group_index == 0
                else grouped_items
            )
            drafts.append(
                make_draft(current_meta, group_text(grouped_items), provenance_items)
            )

    content = subsection.get("content", [])
    item_index = 0
    while item_index < len(content):
        item = content[item_index]
        content_type = item.get("content_type")

        # Keep a complete level-four block atomic. This includes level-five
        # headings and all following text up to the next level-three or
        # level-four heading, or the end of the subsection. If the complete
        # block is larger than max_tokens, semantic integrity takes priority
        # and the resulting content chunk is intentionally oversized.
        if content_type == "heading" and item.get("heading_level") == 4:
            if is_home_emergency_cover_heading(item.get("text")):
                # This overview follows "Important phone numbers" in the same
                # subsection. Start fresh so its explanatory text and coverage
                # table do not inherit that unrelated level-three heading.
                flush()
                current_meta = {
                    **base,
                    "semantic_heading": clean_text(item.get("text")),
                    "semantic_heading_level": 4,
                }

            block_end = item_index + 1
            while block_end < len(content):
                candidate_item = content[block_end]
                if candidate_item.get("content_type") == "table":
                    break
                if (
                    candidate_item.get("content_type") == "heading"
                    and candidate_item.get("heading_level") in {3, 4}
                ):
                    break
                block_end += 1

            block_items = content[item_index:block_end]
            block_parts = [render_item(block_item) for block_item in block_items]
            block_parts = [part for part in block_parts if part]
            block_text = "\n\n".join(block_parts)

            if block_text:
                # Glossary terms are independently useful retrieval units.
                # Keep each definition in its own chunk even when it is well
                # below the normal target size, and expose the term as the
                # semantic heading used by the embedding metadata.
                if (
                    is_definitions_subsection
                    or in_home_emergency_definitions
                ):
                    flush()
                    definition_meta = {
                        **base,
                        "semantic_heading": clean_text(item.get("text")),
                        "semantic_heading_level": 4,
                    }
                    if chunker.fits(definition_meta, block_text):
                        drafts.append(
                            make_draft(definition_meta, block_text, block_items)
                        )
                    else:
                        for piece in chunker.split_long_text(
                            definition_meta, block_text
                        ):
                            drafts.append(
                                make_draft(
                                    definition_meta,
                                    piece,
                                    block_items,
                                )
                            )
                    next_item = (
                        content[block_end] if block_end < len(content) else None
                    )
                    if (
                        next_item
                        and next_item.get("content_type") == "table"
                        and not next_item.get("semantic_heading_level")
                    ):
                        # A table immediately following a definition paragraph
                        # belongs to that definition unless it supplies its own
                        # semantic context. For example, the Contents table
                        # inherits the level-four "Contents" heading.
                        current_meta = dict(definition_meta)
                    else:
                        current_meta = dict(base)
                    item_index = block_end
                    continue

                combined_text = "\n\n".join(current_parts + [block_text])
                if current_parts and not chunker.fits(current_meta, combined_text):
                    flush()

                if not chunker.fits(current_meta, block_text):
                    flush()
                    add_oversized_level_four_block(block_items)
                else:
                    current_items.extend(block_items)
                    current_parts.append(block_text)

            item_index = block_end
            continue

        # In this glossary the visually styled definition labels may be
        # extracted as ordinary paragraphs rather than level-four headings.
        # Treat those known labels as structural boundaries so each term and
        # its following definition are emitted as an independent chunk.
        rendered = render_item(item)
        if (
            in_home_emergency_definitions
            and is_home_emergency_definition_term(rendered)
        ):
            flush()
            current_meta = {
                **base,
                "semantic_heading": clean_text(rendered),
                "semantic_heading_level": 4,
            }
            current_items.append(item)
            current_parts.append(rendered)
            item_index += 1
            continue

        if content_type == "table":
            flush()
            table_meta = dict(current_meta)
            if item.get("semantic_heading_level") is not None:
                table_meta.update(
                    {
                        "semantic_heading": clean_text(
                            item.get("semantic_heading")
                        ),
                        "semantic_heading_level": item.get(
                            "semantic_heading_level"
                        ),
                    }
                )
            drafts.extend(split_table(chunker, table_meta, item))
            current_meta = dict(base)
            item_index += 1
            continue

        # Each benefit or exclusion paragraph in these IP sections is an
        # independent retrieval unit. Do not merge it with neighbouring
        # paragraphs, even when the resulting chunk is below max_tokens.
        if paragraphs_are_atomic and content_type == "paragraph":
            flush()
            rendered = render_item(item)
            if rendered:
                if chunker.fits(current_meta, rendered):
                    drafts.append(make_draft(current_meta, rendered, [item]))
                else:
                    for piece in chunker.split_long_text(current_meta, rendered):
                        drafts.append(make_draft(current_meta, piece, [item]))
            current_meta = dict(base)
            item_index += 1
            continue

        if content_type == "heading" and item.get("heading_level") == 3:
            flush()
            in_home_emergency_definitions = (
                is_home_emergency_definitions_heading(item.get("text"))
            )
            current_meta = {
                **base,
                "semantic_heading": clean_text(item.get("text")),
                "semantic_heading_level": 3,
            }

            # When a level-three heading directly introduces a table, keep it
            # as semantic context on every table part instead of emitting a
            # separate heading-only content chunk. The table renderer supplies
            # the actual chunk text and repeats its columns for split tables.
            next_item = content[item_index + 1] if item_index + 1 < len(content) else None
            if next_item and (
                next_item.get("content_type") == "table"
                or (
                    next_item.get("content_type") == "heading"
                    and next_item.get("heading_level") == 4
                )
            ):
                item_index += 1
                continue

        rendered = render_item(item)
        if not rendered:
            item_index += 1
            continue
        candidate = "\n\n".join(current_parts + [rendered])
        if current_parts and not chunker.fits(current_meta, candidate):
            flush()

        if not chunker.fits(current_meta, rendered):
            for piece in chunker.split_long_text(current_meta, rendered):
                drafts.append(make_draft(current_meta, piece, [item]))
        else:
            current_items.append(item)
            current_parts.append(rendered)

        item_index += 1

    flush()
    return drafts


def finalise_document_chunks(chunker: Chunker, document: dict[str, Any], drafts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(drafts)
    code = clean_text(document.get("document_code")) or Path(str(document.get("document_name", "document"))).stem
    slug = re.sub(r"[^a-z0-9]+", "-", code.lower()).strip("-")
    chunks: list[dict[str, Any]] = []
    for sequence, draft in enumerate(drafts, start=1):
        embedding_text = chunker.embedding_text(draft, draft["text"])
        digest = hashlib.sha256(
            f"{code}|{sequence}|{embedding_text}".encode("utf-8")
        ).hexdigest()[:12]
        chunks.append(
            {
                "chunk_id": f"{slug}-{sequence:04d}-{digest}",
                "chunk_sequence": sequence,
                **draft,
                "embedding_text": embedding_text,
                "token_count": chunker.count(embedding_text),
            }
        )
    return chunks


def load_documents(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, list):
        raise ValueError("Input must contain a top-level 'documents' list")
    return documents


def main() -> int:
    args = parse_args()
    chunker = Chunker(args.encoding, args.max_tokens)
    documents = load_documents(args.input)
    all_chunks: list[dict[str, Any]] = []

    for document in documents:
        drafts: list[dict[str, Any]] = []
        for section in document.get("sections", []):
            section_heading = clean_text(section.get("section_heading"))
            for subsection in section.get("subsections", []):
                drafts.extend(chunk_subsection(chunker, document, section_heading, subsection))
        all_chunks.extend(finalise_document_chunks(chunker, document, drafts))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for chunk in all_chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    counts = [chunk["token_count"] for chunk in all_chunks]
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(all_chunks)}")
    if counts:
        print(f"Token range: {min(counts)}-{max(counts)}")
        print(f"Average tokens: {sum(counts) / len(counts):.1f}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
