#!/usr/bin/env python3
"""Extract structured sections from similarly formatted insurance IP PDFs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pymupdf


SECTION_HEADINGS = (
    "What is this type of insurance?",
    "What is insured?",
    "What is not insured?",
    "Are there any restrictions on cover?",
    "Where am I covered?",
    "What are my obligations?",
    "When and how do I pay?",
    "When does the cover start and end?",
    "How do I cancel the contract?",
)

INSURED_HEADING = "What is insured?"
NOT_INSURED_HEADING = "What is not insured?"

MINIMUM_SECTION_CHARACTERS = 20

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract structured sections from -IP- PDF documents."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the PDF files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="JSON file to create.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    """Normalise whitespace while retaining meaningful line boundaries."""
    cleaned_lines = []

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def printed_page_number(page: pymupdf.Page) -> int:
    """Return the numeric printed-page label, falling back to the PDF page."""
    match = re.search(r"\d+", page.get_label().strip())
    return int(match.group()) if match else page.number + 1


def span_is_bold(span: dict[str, Any]) -> bool:
    """Return whether a PDF text span uses bold formatting."""
    return bool(span.get("flags", 0) & pymupdf.TEXT_FONT_BOLD) or (
        "bold" in span.get("font", "").casefold()
    )


def join_pdf_lines(lines: list[str]) -> str:
    """Join PDF lines, removing whitespace after line-ending hyphens."""
    joined_text = "\n".join(lines)
    return re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "-", joined_text)


def extract_formatted_blocks(
    page: pymupdf.Page,
    clip: pymupdf.Rect,
    excluded_headings: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Extract text blocks while preserving boldness and indentation."""
    formatted_blocks = []

    for block in page.get_text("dict", clip=clip, sort=True)["blocks"]:
        lines = block.get("lines", [])
        spans = [
            span
            for line in lines
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]

        if not spans:
            continue

        text = clean_text(
            join_pdf_lines(
                [
                    "".join(
                        span.get("text", "")
                        for span in line.get("spans", [])
                    )
                    for line in lines
                ]
            )
        )

        for heading in excluded_headings:
            text = remove_heading(text, heading)

        if not text:
            continue

        formatted_blocks.append(
            {
                "text": text,
                "x0": min(span["bbox"][0] for span in spans),
                "starts_bold": span_is_bold(spans[0]),
                "all_bold": all(span_is_bold(span) for span in spans),
            }
        )

    return formatted_blocks


def is_bullet_point(text: str) -> bool:
    """Return whether a text block begins with a bullet marker."""
    return bool(re.match(r"^\s*[-•✓✗]\s*", text))


def group_parent_with_bullets(
    content_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group a colon-ended parent with its immediately following bullets."""
    grouped_blocks: list[dict[str, Any]] = []
    block_index = 0

    while block_index < len(content_blocks):
        parent = content_blocks[block_index]
        parent_text = clean_text(parent["text"])
        next_index = block_index + 1
        child_blocks = []

        if parent_text.endswith(":"):
            while (
                next_index < len(content_blocks)
                and is_bullet_point(content_blocks[next_index]["text"])
            ):
                child_blocks.append(content_blocks[next_index])
                next_index += 1

        if child_blocks:
            grouped_blocks.append(
                {
                    "pdf_page_numbers": list(
                        dict.fromkeys(
                            block["pdf_page_number"]
                            for block in [parent, *child_blocks]
                        )
                    ),
                    "printed_page_numbers": list(
                        dict.fromkeys(
                            block["printed_page_number"]
                            for block in [parent, *child_blocks]
                        )
                    ),
                    "text": "\n".join(
                        [
                            parent_text.replace("\n", " "),
                            *(
                                clean_text(block["text"]).replace("\n", " ")
                                for block in child_blocks
                            ),
                        ]
                    ),
                }
            )
            block_index = next_index
            continue

        grouped_blocks.append(
            {
                "pdf_page_numbers": [parent["pdf_page_number"]],
                "printed_page_numbers": [parent["printed_page_number"]],
                "text": parent_text.replace("\n", " "),
            }
        )
        block_index += 1

    return grouped_blocks


def split_panel_subsections(
    page_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split a panel using bold formatting and indentation."""
    subsections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    all_blocks = [
        block
        for page_record in page_records
        for block in page_record["blocks"]
    ]

    if not all_blocks:
        return []

    outer_x = min(block["x0"] for block in all_blocks)
    alignment_tolerance = 4.0
    subsection_x_positions = [
        block["x0"]
        for block in all_blocks
        if (
            block["all_bold"]
            and block["x0"] <= outer_x + alignment_tolerance
        )
    ]

    subsection_x = (
        min(subsection_x_positions)
        if subsection_x_positions
        else None
    )

    for page_record in page_records:
        pdf_page = page_record["pdf_page_number"]
        printed_page = page_record["printed_page_number"]

        page_blocks = page_record["blocks"]

        for block_index, block in enumerate(page_blocks):
            next_block = (
                page_blocks[block_index + 1]
                if block_index + 1 < len(page_blocks)
                else None
            )
            is_bullet_parent = (
                block["text"].strip().endswith(":")
                and next_block is not None
                and is_bullet_point(next_block["text"])
            )
            is_subsection_heading = (
                subsection_x is not None
                and block["all_bold"]
                and block["x0"] <= subsection_x + alignment_tolerance
                and not is_bullet_parent
            )

            if is_subsection_heading:
                if current is not None and current["content_blocks"]:
                    subsections.append(current)

                current = {
                    "subsection_headings": [block["text"]],
                    "heading_pdf_page_number": pdf_page,
                    "heading_printed_page_number": printed_page,
                    "content_blocks": [],
                }
            else:
                if current is None:
                    current = {
                        "subsection_headings": [],
                        "content_blocks": [],
                    }

                current["content_blocks"].append(
                    {
                        "pdf_page_number": pdf_page,
                        "printed_page_number": printed_page,
                        "text": block["text"],
                    }
                )

    if current is not None and current["content_blocks"]:
        subsections.append(current)

    output = []

    for subsection in subsections:
        source_blocks = subsection["content_blocks"]
        subsection_headings = subsection["subsection_headings"]

        if (
            subsection_headings
            and subsection_headings[-1].strip().endswith(":")
            and source_blocks
            and is_bullet_point(source_blocks[0]["text"])
        ):
            source_blocks = [
                {
                    "pdf_page_number": subsection[
                        "heading_pdf_page_number"
                    ],
                    "printed_page_number": subsection[
                        "heading_printed_page_number"
                    ],
                    "text": subsection_headings[-1],
                },
                *source_blocks,
            ]

        content_blocks = group_parent_with_bullets(source_blocks)
        pdf_pages = list(
            dict.fromkeys(
                page
                for item in content_blocks
                for page in item["pdf_page_numbers"]
            )
        )
        printed_pages = list(
            dict.fromkeys(
                page
                for item in content_blocks
                for page in item["printed_page_numbers"]
            )
        )
        content = [
            {
                "content_type": "paragraph",
                "text": block["text"],
                "source_pdf_pages": block["pdf_page_numbers"],
                "source_printed_pages": block["printed_page_numbers"],
                "order": order,
            }
            for order, block in enumerate(content_blocks, start=1)
        ]
        output.append(
            {
                "subsection_headings": subsection["subsection_headings"],
                "start_printed_page": printed_pages[0],
                "end_printed_page": printed_pages[-1],
                "start_pdf_page": pdf_pages[0],
                "end_pdf_page": pdf_pages[-1],
                "content": content,
            }
        )

    return output


def normalise_heading(text: str) -> str:
    """Normalise a heading for case-insensitive comparison."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def find_heading(
    page: pymupdf.Page,
    heading: str,
) -> pymupdf.Rect | None:
    """Find a heading using case-insensitive word-block matching."""
    matches = page.search_for(heading)

    if matches:
        return matches[0]

    # Fall back to matching text blocks because capitalisation can vary.
    expected = normalise_heading(heading)

    for block in page.get_text("blocks", sort=True):
        block_text = clean_text(block[4])

        if expected in normalise_heading(block_text):
            return pymupdf.Rect(
                block[0],
                block[1],
                block[2],
                block[3],
            )

    return None


def remove_heading(text: str, heading: str) -> str:
    """Remove a section heading from extracted section text."""
    pattern = re.compile(
        re.escape(heading),
        flags=re.IGNORECASE,
    )

    return clean_text(pattern.sub("", text, count=1))


def extract_product(page: pymupdf.Page) -> str | None:
    """Extract the product name from the first page."""
    text = clean_text(page.get_text("text", sort=True))

    match = re.search(
        r"\bProduct\s*:\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(1).strip()


def extract_fca_reference_number(
    page: pymupdf.Page,
) -> str | None:
    """Extract the insurer's FCA reference number."""
    text = clean_text(page.get_text("text", sort=True))

    patterns = (
        # For example: Financial Services Register reference number: 309378
        r"(?:Financial Services Register\s+)?"
        r"reference\s+number\s*:\s*(\d+)",

        # For example: (FRN202106), FRN 202106 or FRN: 202106
        r"\bFRN\s*:?\s*(\d+)\b",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1)

    return None


def extract_first_page_introduction(
    page: pymupdf.Page,
) -> dict[str, Any] | None:
    """Extract the insurance-type description from page one."""
    heading_rect = find_heading(
        page,
        "What is this type of insurance?",
    )
    insured_rect = find_heading(page, INSURED_HEADING)
    not_insured_rect = find_heading(page, NOT_INSURED_HEADING)

    if heading_rect is None:
        return None

    following_positions = [
        rect.y0
        for rect in (insured_rect, not_insured_rect)
        if rect is not None and rect.y0 > heading_rect.y0
    ]

    section_bottom = (
        min(following_positions)
        if following_positions
        else page.rect.height
    )

    section_area = pymupdf.Rect(
        0,
        heading_rect.y0,
        page.rect.width,
        section_bottom,
    )

    formatted_blocks = extract_formatted_blocks(
        page,
        section_area,
        ("What is this type of insurance?",),
    )

    return {
        "section_heading": "What is this type of insurance?",
        "subsections": split_panel_subsections(
            [
                {
                    "pdf_page_number": page.number + 1,
                    "printed_page_number": printed_page_number(page),
                    "blocks": formatted_blocks,
                }
            ]
        ),
    }


def find_next_full_width_heading(
    page: pymupdf.Page,
    minimum_y: float = 0,
) -> tuple[str, pymupdf.Rect] | None:
    """Find the first full-width section heading below a position."""
    excluded_headings = {
        "What is this type of insurance?",
        INSURED_HEADING,
        NOT_INSURED_HEADING,
    }

    matches = []

    for heading in SECTION_HEADINGS:
        if heading in excluded_headings:
            continue

        heading_rect = find_heading(page, heading)

        if heading_rect is not None and heading_rect.y0 >= minimum_y:
            matches.append((heading, heading_rect))

    if not matches:
        return None

    return min(
        matches,
        key=lambda item: item[1].y0,
    )


def extract_insured_panels(
    document: pymupdf.Document,
) -> list[dict[str, Any]]:
    """Extract insured and not-insured panels across multiple pages."""
    start_page_index = None
    initial_insured_rect = None
    initial_not_insured_rect = None

    # Locate the page on which the panels start.
    for page_index, page in enumerate(document):
        insured_rect = find_heading(page, INSURED_HEADING)
        not_insured_rect = find_heading(page, NOT_INSURED_HEADING)

        if insured_rect is not None and not_insured_rect is not None:
            start_page_index = page_index
            initial_insured_rect = insured_rect
            initial_not_insured_rect = not_insured_rect
            break

    if (
        start_page_index is None
        or initial_insured_rect is None
        or initial_not_insured_rect is None
    ):
        return []

    start_page = document[start_page_index]
    # The IP documents use two equally sized panels.
    column_boundary = start_page.rect.width / 2

    insured_pages = []
    not_insured_pages = []

    for page_index in range(start_page_index, document.page_count):
        page = document[page_index]
        page_number = page_index + 1

        insured_rect = find_heading(page, INSURED_HEADING)
        not_insured_rect = find_heading(page, NOT_INSURED_HEADING)

        if page_index == start_page_index:
            panel_top = min(
                initial_insured_rect.y0,
                initial_not_insured_rect.y0,
            )
        elif insured_rect is not None and not_insured_rect is not None:
            # Some continuation pages repeat the panel headings.
            panel_top = min(
                insured_rect.y0,
                not_insured_rect.y0,
            )
        else:
            # The page continues the columns without repeating headings.
            panel_top = 0

        next_heading = find_next_full_width_heading(
            page=page,
            minimum_y=panel_top,
        )

        # If a continuation page begins with a full-width section heading,
        # the panels ended on the preceding page.
        if (
            page_index > start_page_index
            and next_heading is not None
            and next_heading[1].y0 < page.rect.height * 0.15
        ):
            break

        if next_heading is not None:
            _, next_heading_rect = next_heading
            panel_bottom = next_heading_rect.y0
            panels_end_on_this_page = True
        else:
            panel_bottom = page.rect.height
            panels_end_on_this_page = False

        # A full-width section may begin at the top of the page, leaving no
        # panel content to extract.
        if panel_bottom > panel_top:
            insured_area = pymupdf.Rect(
                0,
                panel_top,
                column_boundary,
                panel_bottom,
            )

            not_insured_area = pymupdf.Rect(
                column_boundary,
                panel_top,
                page.rect.width,
                panel_bottom,
            )

            insured_blocks = extract_formatted_blocks(
                page,
                insured_area,
                (INSURED_HEADING,),
            )
            not_insured_blocks = extract_formatted_blocks(
                page,
                not_insured_area,
                (NOT_INSURED_HEADING,),
            )

            if insured_blocks:
                insured_pages.append(
                    {
                        "pdf_page_number": page_number,
                        "printed_page_number": printed_page_number(page),
                        "blocks": insured_blocks,
                    }
                )

            if not_insured_blocks:
                not_insured_pages.append(
                    {
                        "pdf_page_number": page_number,
                        "printed_page_number": printed_page_number(page),
                        "blocks": not_insured_blocks,
                    }
                )

        if panels_end_on_this_page:
            break

    return [
        {
            "section_heading": "What is insured?",
            "subsections": split_panel_subsections(insured_pages),
        },
        {
            "section_heading": "What is not insured?",
            "subsections": split_panel_subsections(not_insured_pages),
        },
    ]


def find_page_sections(
    page: pymupdf.Page,
) -> list[tuple[str, pymupdf.Rect]]:
    """Find and order recognised section headings on a page."""
    matches = []

    for heading in SECTION_HEADINGS:
        # These sections are handled separately on the first page.
        if heading in {
            "What is this type of insurance?",
            INSURED_HEADING,
            NOT_INSURED_HEADING,
        }:
            continue

        heading_rect = find_heading(page, heading)

        if heading_rect is not None:
            matches.append((heading, heading_rect))

    return sorted(
        matches,
        key=lambda item: (item[1].y0, item[1].x0),
    )


def extract_full_width_sections(
    page: pymupdf.Page,
) -> list[dict[str, Any]]:
    """Extract full-width sections using heading boundaries."""
    heading_matches = find_page_sections(page)
    sections = []

    for index, (heading, heading_rect) in enumerate(heading_matches):
        if index + 1 < len(heading_matches):
            section_bottom = heading_matches[index + 1][1].y0
        else:
            section_bottom = page.rect.height

        section_area = pymupdf.Rect(
            0,
            heading_rect.y0,
            page.rect.width,
            section_bottom,
        )

        formatted_blocks = extract_formatted_blocks(
            page,
            section_area,
            (heading,),
        )
        sections.append(
            {
                "section_heading": heading,
                "subsections": split_panel_subsections(
                    [
                        {
                            "pdf_page_number": page.number + 1,
                            "printed_page_number": printed_page_number(page),
                            "blocks": formatted_blocks,
                        }
                    ]
                ),
            }
        )

    return sections


def validate_sections(
    document_name: str,
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Report missing or unusually short sections."""
    warnings = []

    extracted_headings = {
        normalise_heading(section["section_heading"])
        for section in sections
    }

    for expected_heading in SECTION_HEADINGS:
        if normalise_heading(expected_heading) not in extracted_headings:
            warnings.append(
                {
                    "document_name": document_name,
                    "heading": expected_heading,
                    "reason": "Expected section was not found",
                }
            )

    for section in sections:
        content_items = [
            item
            for subsection in section["subsections"]
            for item in subsection["content"]
        ]

        section_text = " ".join(item["text"] for item in content_items)
        character_count = len(re.sub(r"\s+", "", section_text))

        if character_count < MINIMUM_SECTION_CHARACTERS:
            warnings.append(
                {
                    "document_name": document_name,
                    "heading": section["section_heading"],
                    "character_count": character_count,
                    "reason": "Little or no section text was extracted",
                }
            )

    return warnings


def extract_ip_pdf(pdf_path: Path) -> dict[str, Any]:
    """Extract metadata and grouped sections from one IP PDF."""
    sections = []

    with pymupdf.open(pdf_path) as document:
        if document.page_count == 0:
            raise ValueError("PDF contains no pages")

        first_page = document[0]

        product = extract_product(first_page)
        fca_reference_number = (
            extract_fca_reference_number(first_page)
        )

        insurance_type = extract_first_page_introduction(first_page)

        if insurance_type:
            sections.append(insurance_type)

        sections.extend(extract_insured_panels(document))

        # Search every page because some documents place additional
        # sections below the panels on page 1.
        for page in document:
            sections.extend(
                extract_full_width_sections(page)
            )

        page_count = document.page_count

    warnings = validate_sections(
        document_name=pdf_path.name,
        sections=sections,
    )

    return {
        "document_name": pdf_path.name,
        "document_type": "insurance_product_information",
        "product": product,
        "fca_reference_number": fca_reference_number,
        "page_count": page_count,
        "sections": sections,
        "warnings": warnings,
    }

def find_ip_pdfs(input_directory: Path) -> list[Path]:
    """Find PDF files containing -IP- in their names."""
    return sorted(
        path
        for path in input_directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.casefold() == ".pdf"
            and "-ip-" in path.name.casefold()
        )
    )


def main() -> int:
    """Extract every IP document in the input directory."""
    arguments = parse_arguments()

    input_directory = arguments.input_dir.resolve()
    output_file = arguments.output_file.resolve()

    if not input_directory.is_dir():
        print(
            f"ERROR: Input directory not found: {input_directory}",
            file=sys.stderr,
        )
        return 1

    pdf_paths = find_ip_pdfs(input_directory)

    if not pdf_paths:
        print(
            f"ERROR: No -IP- PDF files found in {input_directory}",
            file=sys.stderr,
        )
        return 1

    documents = []
    failures = []

    for pdf_path in pdf_paths:
        print(f"Extracting: {pdf_path.name}")

        try:
            document = extract_ip_pdf(pdf_path)
            documents.append(document)

            for section in document["sections"]:
                heading = section["section_heading"]

                start_pages = [
                    item["start_pdf_page"]
                    for item in section["subsections"]
                ]
                end_pages = [
                    item["end_pdf_page"]
                    for item in section["subsections"]
                ]
                start_page = min(start_pages) if start_pages else None
                end_page = max(end_pages) if end_pages else None

                if start_page is None:
                    page_label = "Unknown page"
                elif end_page is not None and end_page != start_page:
                    page_label = f"Pages {start_page}-{end_page}"
                else:
                    page_label = f"Page {start_page}"

                print(f"  {page_label}: {heading}")
        except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
            print(
                f"ERROR: {pdf_path.name}: {error}",
                file=sys.stderr,
            )
            failures.append(
                {
                    "document_name": pdf_path.name,
                    "error": str(error),
                }
            )

    warnings = [
        warning
        for document in documents
        for warning in document["warnings"]
    ]

    output = {
        "summary": {
            "documents_found": len(pdf_paths),
            "documents_extracted": len(documents),
            "documents_failed": len(failures),
            "sections_extracted": sum(
                len(document["sections"])
                for document in documents
            ),
            "warnings": len(warnings),
        },
        "warnings": warnings,
        "failures": failures,
        "documents": documents,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nExtraction summary")
    print(f"Documents found:     {len(pdf_paths)}")
    print(f"Documents extracted: {len(documents)}")
    print(f"Documents failed:    {len(failures)}")
    print(f"Sections extracted:  {output['summary']['sections_extracted']}")
    print(f"Warnings:            {len(warnings)}")
    print(f"Output:              {output_file}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
