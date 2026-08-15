#!/usr/bin/env python3
"""Create semantic JSONL chunks from the extracted insurance-document JSON.

The input must use the shared ``documents -> sections -> subsections -> content``
schema produced by the IP and HH PDF extractors. Chunk boundaries occur at a
section, subsection, or level-three heading. Tables are isolated and, when
necessary, split only between complete rows.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantically chunk one extracted IP or HH JSON file."
    )
    parser.add_argument("input", type=Path, help="Extracted JSON input file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="JSONL output file")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--encoding", default="cl100k_base", help="Tiktoken encoding name")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return re.sub(r"[ \t]+\n", "\n", str(value or "")).strip()


def unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def humanise(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


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
        lines.append("Columns: " + " | ".join(humanise(str(c)) for c in columns))
    for group in table.get("permission_groups", []):
        heading = clean_text(group.get("heading"))
        permissions = ", ".join(humanise(str(p)) for p in group.get("permissions", []))
        lines.append(f"{heading}: {permissions}")
    return lines


def render_table_row(row: Any, columns: list[Any] | None = None) -> str:
    if isinstance(row, list):
        labels = columns or list(range(1, len(row) + 1))
        return " | ".join(
            f"{humanise(str(label))}: {clean_text(value)}"
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
            status = ", ".join(f"{humanise(str(k))}: {humanise(str(v))}" for k, v in value.items())
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
    headers = table_header_lines(table)
    rows = table.get("rows", [])
    rendered = [render_table_row(row, table.get("columns")) for row in rows]
    groups: list[tuple[int, int, list[str]]] = []
    start = 0
    current: list[str] = []

    for index, row_text in enumerate(rendered):
        candidate_rows = current + [row_text]
        candidate = "\n\n".join(headers + candidate_rows)
        if current and not chunker.fits(meta, candidate):
            groups.append((start, index - 1, current))
            start = index
            current = [row_text]
        else:
            current = candidate_rows
    if current or not rows:
        groups.append((start, max(start, len(rows) - 1), current))

    drafts: list[dict[str, Any]] = []
    for group_index, (row_start, row_end, row_texts) in enumerate(groups, start=1):
        selected_rows = rows[row_start : row_end + 1]
        row_pdf = unique(page for row in selected_rows if isinstance(row, dict) for page in item_pages(row, "source_pdf_pages"))
        row_printed = unique(page for row in selected_rows if isinstance(row, dict) for page in item_pages(row, "source_printed_pages"))
        item = dict(table)
        if row_pdf:
            item["source_pdf_pages"] = row_pdf
        if row_printed:
            item["source_printed_pages"] = row_printed
        text = "\n\n".join(headers + row_texts)
        draft = make_draft(meta, text, [item], "table")
        draft.update(
            {
                "table_type": table.get("table_type"),
                "table_is_split": len(groups) > 1,
                "table_chunk_index": group_index,
                "table_chunk_count": len(groups),
                "table_row_start": row_start + 1 if rows else None,
                "table_row_end": row_end + 1 if rows else None,
                "contains_complete_rows": True,
                "contains_complete_table": len(groups) == 1,
            }
        )
        drafts.append(draft)
    return drafts


def chunk_subsection(chunker: Chunker, document: dict[str, Any], section: str, subsection: dict[str, Any]) -> list[dict[str, Any]]:
    subsection_headings = list(subsection.get("subsection_headings") or [])
    base = base_metadata(document, section, subsection_headings)
    drafts: list[dict[str, Any]] = []
    current_items: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_meta = dict(base)

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

        if content_type == "table":
            flush()
            drafts.extend(split_table(chunker, current_meta, item))
            item_index += 1
            continue

        if content_type == "heading" and item.get("heading_level") == 3:
            flush()
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
