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
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# tiktoken is required: chunk boundaries, token counts and therefore chunk IDs
# depend on the exact tokenizer, so there is deliberately no fallback.
import tiktoken

from insurance_rag.logging_setup import add_logging_arguments, configure_logging

logger = logging.getLogger(__name__)

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
    """Parse command-line options for chunking an extracted document file."""
    parser = argparse.ArgumentParser(
        description="Semantically chunk one extracted IP or HH JSON file."
    )
    parser.add_argument("input", type=Path, help="Extracted JSON input file")
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="JSONL output file"
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--encoding", default="cl100k_base", help="Tiktoken encoding name"
    )
    add_logging_arguments(parser)
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

        starts_list_item = bool(re.match(r"^(?:[•*-]|\d+[.)]|[A-Za-z][.)])\s+", line))
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
    """Return the value as text with soft line wraps joined."""
    return normalise_soft_line_wraps(value)


def unique(values: Iterable[Any]) -> list[Any]:
    """Return the values with duplicates removed, keeping first-seen order."""
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def humanise(value: str) -> str:
    """Convert a snake_case identifier to sentence-case text."""
    return value.replace("_", " ").strip().capitalize()


def display_label(value: Any) -> str:
    """Return the display label for a cover status or column key."""
    key = str(value).strip().casefold()
    return COVER_STATUS_LABELS.get(key, humanise(str(value)))


def render_item(item: dict[str, Any]) -> str:
    """Render one content item (heading, list item, notice or text) as text."""
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
    """Return the lines describing a table's type, columns and permission groups."""
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
    """Render a table row as labelled ``label: value`` pairs."""
    if isinstance(row, list):
        labels = columns or list(range(1, len(row) + 1))
        return " | ".join(
            f"{display_label(label)}: {clean_text(value)}"
            for label, value in zip(labels, row, strict=True)
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
                f"{display_label(k)}: {humanise(str(v))}" for k, v in value.items()
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
    """Build the metadata header that is prepended to a chunk's embedding text."""
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
    """Measures and splits text so chunks fit within a token limit."""

    def __init__(self, encoding_name: str, max_tokens: int) -> None:
        """Set up the tiktoken encoding used to count and split tokens.

        Raises:
            ValueError: If ``max_tokens`` is not positive or the encoding name
                is unknown.
        """
        if max_tokens <= 0:
            raise ValueError("--max-tokens must be greater than zero")
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.max_tokens = max_tokens

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        return len(self.encoding.encode(text))

    def embedding_text(self, meta: dict[str, Any], text: str) -> str:
        """Return the text to embed: the metadata prefix followed by ``text``."""
        prefix = metadata_prefix(meta)
        return f"{prefix}\n\n{text}" if prefix else text

    def fits(self, meta: dict[str, Any], text: str) -> bool:
        """Return whether ``text`` with its metadata prefix fits the token limit."""
        return self.count(self.embedding_text(meta, text)) <= self.max_tokens

    def split_long_text(self, meta: dict[str, Any], text: str) -> list[str]:
        """Split an indivisible long item using sentence boundaries, then tokens."""
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", text) if s.strip()
        ]
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
                pieces.extend(
                    self.encoding.decode(tokens[i : i + allowance]).strip()
                    for i in range(0, len(tokens), allowance)
                )
                current = ""
        if current:
            pieces.append(current)
        return [piece for piece in pieces if piece]


def item_pages(item: dict[str, Any], key: str) -> list[int]:
    """Return the page numbers stored under ``key`` as integers."""
    return [int(page) for page in item.get(key, [])]


def base_metadata(
    document: dict[str, Any], section: str, subsection: list[str]
) -> dict[str, Any]:
    """Return the document and section metadata shared by a subsection's chunks."""
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


def split_ip_section_paragraphs(document: dict[str, Any], section_heading: str) -> bool:
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
    normalised = re.sub(r"[^a-z]+", " ", clean_text(value).casefold()).strip()
    return normalised == (
        "home emergency cover and home emergency extra cover definitions"
    )


def is_home_emergency_cover_heading(value: Any) -> bool:
    """Return whether a heading introduces the HEC/HEX overview and table."""
    normalised = re.sub(r"[^a-z]+", " ", clean_text(value).casefold()).strip()
    return normalised == "home emergency cover and home emergency extra cover"


def is_home_emergency_definition_term(value: Any) -> bool:
    """Recognise definition labels even when extraction marks them as text."""
    normalised = re.sub(r"[^a-z]+", " ", clean_text(value).casefold()).strip()
    return normalised in HOME_EMERGENCY_DEFINITION_TERMS


def make_draft(
    meta: dict[str, Any],
    text: str,
    items: list[dict[str, Any]],
    chunk_type: str = "content",
) -> dict[str, Any]:
    """Create a chunk draft with its text and page and order ranges."""
    pdf_pages = unique(
        page for item in items for page in item_pages(item, "source_pdf_pages")
    )
    printed_pages = unique(
        page for item in items for page in item_pages(item, "source_printed_pages")
    )
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


def split_table(
    chunker: Chunker, meta: dict[str, Any], table: dict[str, Any]
) -> list[dict[str, Any]]:
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
        row_texts = [render_table_row(row, table.get("columns")) for row in rows]
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
        row_printed = (
            item_pages(row, "source_printed_pages") if isinstance(row, dict) else []
        )
        row_heading = (
            clean_text(row.get("row_heading")) if isinstance(row, dict) else ""
        )
        row_meta = dict(meta)
        if row_heading:
            # Preserve meaningful parent context and append the row-specific
            # subject. Generic introductory headings describe how to read the
            # table rather than what the row covers, so use the human-readable
            # table type for those instead.
            parent_heading = clean_text(meta.get("semantic_heading"))
            if not parent_heading or parent_heading.casefold() in {
                "how to read this document",
                "what we’ll not pay",
                "what we'll not pay",
            }:
                parent_heading = humanise(str(table.get("table_type", "table")))
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


def chunk_subsection(
    chunker: Chunker, document: dict[str, Any], section: str, subsection: dict[str, Any]
) -> list[dict[str, Any]]:
    """Split a subsection into chunk drafts that fit the token limit.

    Items are grouped along heading boundaries where possible, and long
    items are split by sentence and then by token.
    """
    return SubsectionChunker(chunker, document, section, subsection).run()


def is_heading(item: dict[str, Any] | None, *levels: int) -> bool:
    """Return whether ``item`` is a heading at one of the given levels."""
    return (
        item is not None
        and item.get("content_type") == "heading"
        and item.get("heading_level") in levels
    )


def render_items(items: list[dict[str, Any]]) -> str:
    """Render items and join the non-empty results with blank lines."""
    return "\n\n".join(part for part in (render_item(item) for item in items) if part)


class SubsectionChunker:
    """Builds the chunk drafts for one subsection.

    Content items are read in order. Consecutive items are accumulated into
    the current chunk until the next one would exceed the token limit or a
    structural boundary (heading, table, glossary term) starts a new chunk.
    """

    def __init__(
        self,
        chunker: Chunker,
        document: dict[str, Any],
        section: str,
        subsection: dict[str, Any],
    ) -> None:
        """Prepare the shared metadata and an empty current chunk."""
        self.chunker = chunker
        self.content: list[dict[str, Any]] = subsection.get("content", [])
        subsection_headings = list(subsection.get("subsection_headings") or [])
        self.is_definitions_subsection = any(
            clean_text(heading).casefold() == "definitions"
            for heading in subsection_headings
        )
        self.base = base_metadata(document, section, subsection_headings)
        self.paragraphs_are_atomic = split_ip_section_paragraphs(document, section)
        self.drafts: list[dict[str, Any]] = []
        self.current_items: list[dict[str, Any]] = []
        self.current_parts: list[str] = []
        self.current_meta = dict(self.base)
        self.in_home_emergency_definitions = False

    def run(self) -> list[dict[str, Any]]:
        """Process every content item and return the chunk drafts."""
        index = 0
        while index < len(self.content):
            index = self.process_item(index)
        self.flush()
        return self.drafts

    def process_item(self, index: int) -> int:
        """Handle the item at ``index`` and return the index of the next item."""
        item = self.content[index]
        content_type = item.get("content_type")

        if is_heading(item, 4):
            return self.add_level_four_block(index)

        # In this glossary the visually styled definition labels may be
        # extracted as ordinary paragraphs rather than level-four headings.
        # Treat those known labels as structural boundaries so each term and
        # its following definition are emitted as an independent chunk.
        rendered = render_item(item)
        if self.in_home_emergency_definitions and is_home_emergency_definition_term(
            rendered
        ):
            self.start_definition_term(item, rendered)
            return index + 1

        if content_type == "table":
            self.add_table(item)
            return index + 1

        # Each benefit or exclusion paragraph in these IP sections is an
        # independent retrieval unit. Do not merge it with neighbouring
        # paragraphs, even when the resulting chunk is below max_tokens.
        if self.paragraphs_are_atomic and content_type == "paragraph":
            self.add_atomic_paragraph(item)
            return index + 1

        if is_heading(item, 3) and self.start_level_three_section(index):
            return index + 1

        self.add_flowing_item(item)
        return index + 1

    # Building chunks -----------------------------------------------------

    def flush(self) -> None:
        """Emit the accumulated items as one chunk and start a new one."""
        if self.current_parts:
            self.drafts.append(
                make_draft(
                    self.current_meta,
                    "\n\n".join(self.current_parts),
                    self.current_items,
                )
            )
        self.current_items = []
        self.current_parts = []

    def heading_meta(self, heading_text: Any, level: int) -> dict[str, Any]:
        """Return the base metadata with a semantic heading at ``level``."""
        return {
            **self.base,
            "semantic_heading": clean_text(heading_text),
            "semantic_heading_level": level,
        }

    def add_standalone(
        self, meta: dict[str, Any], text: str, items: list[dict[str, Any]]
    ) -> None:
        """Emit ``text`` as its own chunk, splitting it if it is too long."""
        if self.chunker.fits(meta, text):
            self.drafts.append(make_draft(meta, text, items))
        else:
            for piece in self.chunker.split_long_text(meta, text):
                self.drafts.append(make_draft(meta, piece, items))

    def add_flowing_item(self, item: dict[str, Any]) -> None:
        """Add an item to the current chunk, starting a new chunk if it is full."""
        rendered = render_item(item)
        if not rendered:
            return
        candidate = "\n\n".join([*self.current_parts, rendered])
        if self.current_parts and not self.chunker.fits(self.current_meta, candidate):
            self.flush()

        if not self.chunker.fits(self.current_meta, rendered):
            for piece in self.chunker.split_long_text(self.current_meta, rendered):
                self.drafts.append(make_draft(self.current_meta, piece, [item]))
        else:
            self.current_items.append(item)
            self.current_parts.append(rendered)

    # Item handlers -------------------------------------------------------

    def start_definition_term(self, item: dict[str, Any], rendered: str) -> None:
        """Start a new chunk for a Home Emergency glossary term."""
        self.flush()
        self.current_meta = self.heading_meta(rendered, 4)
        self.current_items.append(item)
        self.current_parts.append(rendered)

    def add_table(self, item: dict[str, Any]) -> None:
        """Emit a table as its own chunks, using its semantic heading if it has one."""
        self.flush()
        table_meta = dict(self.current_meta)
        if item.get("semantic_heading_level") is not None:
            table_meta.update(
                {
                    "semantic_heading": clean_text(item.get("semantic_heading")),
                    "semantic_heading_level": item.get("semantic_heading_level"),
                }
            )
        self.drafts.extend(split_table(self.chunker, table_meta, item))
        self.current_meta = dict(self.base)

    def add_atomic_paragraph(self, item: dict[str, Any]) -> None:
        """Emit an IP benefit or exclusion paragraph as its own chunk."""
        self.flush()
        rendered = render_item(item)
        if rendered:
            self.add_standalone(self.current_meta, rendered, [item])
        self.current_meta = dict(self.base)

    def start_level_three_section(self, index: int) -> bool:
        """Start a new chunk under a level-three heading.

        Returns:
            True if the heading only provides context for a following table or
            level-four heading and should not be added as chunk text.
        """
        item = self.content[index]
        self.flush()
        self.in_home_emergency_definitions = is_home_emergency_definitions_heading(
            item.get("text")
        )
        self.current_meta = self.heading_meta(item.get("text"), 3)

        # When a level-three heading directly introduces a table, keep it
        # as semantic context on every table part instead of emitting a
        # separate heading-only content chunk. The table renderer supplies
        # the actual chunk text and repeats its columns for split tables.
        next_item = self.item_at(index + 1)
        return next_item is not None and (
            next_item.get("content_type") == "table" or is_heading(next_item, 4)
        )

    def item_at(self, index: int) -> dict[str, Any] | None:
        """Return the content item at ``index``, or None past the end."""
        return self.content[index] if index < len(self.content) else None

    # Level-four blocks ---------------------------------------------------

    def level_four_block_end(self, index: int) -> int:
        """Return the index just after the level-four block starting at ``index``.

        The block runs up to the next table, level-three or level-four heading,
        or the end of the subsection.
        """
        block_end = index + 1
        while block_end < len(self.content):
            candidate_item = self.content[block_end]
            if candidate_item.get("content_type") == "table":
                break
            if is_heading(candidate_item, 3, 4):
                break
            block_end += 1
        return block_end

    def add_level_four_block(self, index: int) -> int:
        """Add a complete level-four block and return the index after it.

        A level-four block is kept atomic. This includes level-five headings
        and all following text up to the next level-three or level-four
        heading, or the end of the subsection. If the complete block is larger
        than max_tokens, semantic integrity takes priority and the resulting
        content chunk is intentionally oversized.
        """
        item = self.content[index]
        if is_home_emergency_cover_heading(item.get("text")):
            # This overview follows "Important phone numbers" in the same
            # subsection. Start fresh so its explanatory text and coverage
            # table do not inherit that unrelated level-three heading.
            self.flush()
            self.current_meta = self.heading_meta(item.get("text"), 4)

        block_end = self.level_four_block_end(index)
        block_items = self.content[index:block_end]
        block_text = render_items(block_items)
        if not block_text:
            return block_end

        if self.is_definitions_subsection or self.in_home_emergency_definitions:
            self.add_definition_block(item, block_items, block_text, block_end)
            return block_end

        combined_text = "\n\n".join([*self.current_parts, block_text])
        if self.current_parts and not self.chunker.fits(
            self.current_meta, combined_text
        ):
            self.flush()

        if not self.chunker.fits(self.current_meta, block_text):
            self.flush()
            self.add_oversized_level_four_block(block_items)
        else:
            self.current_items.extend(block_items)
            self.current_parts.append(block_text)
        return block_end

    def add_definition_block(
        self,
        item: dict[str, Any],
        block_items: list[dict[str, Any]],
        block_text: str,
        block_end: int,
    ) -> None:
        """Emit one glossary definition as its own chunk.

        Glossary terms are independently useful retrieval units. Keep each
        definition in its own chunk even when it is well below the normal
        target size, and expose the term as the semantic heading used by the
        embedding metadata.
        """
        self.flush()
        definition_meta = self.heading_meta(item.get("text"), 4)
        self.add_standalone(definition_meta, block_text, block_items)

        next_item = self.item_at(block_end)
        if (
            next_item
            and next_item.get("content_type") == "table"
            and not next_item.get("semantic_heading_level")
        ):
            # A table immediately following a definition paragraph
            # belongs to that definition unless it supplies its own
            # semantic context. For example, the Contents table
            # inherits the level-four "Contents" heading.
            self.current_meta = dict(definition_meta)
        else:
            self.current_meta = dict(self.base)

    def add_oversized_level_four_block(self, block_items: list[dict[str, Any]]) -> None:
        """Split an oversized level-four block only at level-five boundaries."""
        parent_item = block_items[0]
        body_items = block_items[1:]

        # Without a level-five boundary there is no safe structural split.
        if not any(is_heading(item, 5) for item in body_items):
            self.drafts.append(
                make_draft(self.current_meta, render_items(block_items), block_items)
            )
            return

        parent_heading = render_item(parent_item)
        grouped_segments = self.group_level_five_segments(
            parent_heading, split_at_level_five_headings(body_items)
        )
        for group_index, grouped_items in enumerate(grouped_segments):
            provenance_items = (
                [parent_item, *grouped_items] if group_index == 0 else grouped_items
            )
            self.drafts.append(
                make_draft(
                    self.current_meta,
                    level_four_group_text(parent_heading, grouped_items),
                    provenance_items,
                )
            )

    def group_level_five_segments(
        self, parent_heading: str, segments: list[list[dict[str, Any]]]
    ) -> list[list[dict[str, Any]]]:
        """Combine consecutive level-five segments while they fit the token limit."""
        grouped_segments: list[list[dict[str, Any]]] = []
        group: list[dict[str, Any]] = []
        for segment in segments:
            candidate = [*group, *segment]
            if group and not self.chunker.fits(
                self.current_meta, level_four_group_text(parent_heading, candidate)
            ):
                grouped_segments.append(group)
                group = list(segment)
            else:
                group = candidate
        if group:
            grouped_segments.append(group)
        return grouped_segments


def split_at_level_five_headings(
    items: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split items into segments that each start at a level-five heading."""
    segments: list[list[dict[str, Any]]] = []
    segment: list[dict[str, Any]] = []
    for item in items:
        if is_heading(item, 5):
            if segment:
                segments.append(segment)
            segment = [item]
        else:
            segment.append(item)
    if segment:
        segments.append(segment)
    return segments


def level_four_group_text(parent_heading: str, items: list[dict[str, Any]]) -> str:
    """Return a group of level-five segments prefixed by their level-four heading."""
    body = render_items(items)
    return f"{parent_heading}\n\n{body}" if body else parent_heading


def finalise_document_chunks(
    chunker: Chunker, document: dict[str, Any], drafts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Assign sequence numbers and stable IDs to a document's chunk drafts."""
    code = (
        clean_text(document.get("document_code"))
        or Path(str(document.get("document_name", "document"))).stem
    )
    slug = re.sub(r"[^a-z0-9]+", "-", code.lower()).strip("-")
    chunks: list[dict[str, Any]] = []
    for sequence, draft in enumerate(drafts, start=1):
        embedding_text = chunker.embedding_text(draft, draft["text"])
        digest = hashlib.sha256(
            f"{code}|{sequence}|{embedding_text}".encode()
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
    """Load the ``documents`` list from an extracted JSON file."""
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, list):
        raise ValueError("Input must contain a top-level 'documents' list")
    return documents


def main() -> int:
    """Chunk every document in the input file and write the chunks as JSONL.

    Returns:
        The process exit code.
    """
    args = parse_args()
    configure_logging(args.verbose, args.quiet)
    chunker = Chunker(args.encoding, args.max_tokens)
    documents = load_documents(args.input)
    all_chunks: list[dict[str, Any]] = []

    for document in documents:
        drafts: list[dict[str, Any]] = []
        for section in document.get("sections", []):
            section_heading = clean_text(section.get("section_heading"))
            for subsection in section.get("subsections", []):
                drafts.extend(
                    chunk_subsection(chunker, document, section_heading, subsection)
                )
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
        logger.error("%s", exc)
        raise SystemExit(1) from exc
