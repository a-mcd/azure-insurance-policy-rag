#!/usr/bin/env python3
"""Extract HH- home-insurance booklets into contents-grouped JSON.

Usage:
    python3 src/extract_hh_pdfs.py \
      --input-dir data/raw \
      --output-file data/processed/hh_documents.json

Dependency:
    python3 -m pip install pymupdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pymupdf


MINIMUM_PAGE_CHARACTERS = 50
UNNUMBERED_FRONT_PAGES = 2
FINAL_PRINTED_PAGE = 67
COVER_LEVELS = ("admiral", "gold", "platinum")
EXCLUDED_PDF_PAGES = {
    1: "Front cover",
    2: "Contents page represented by the section hierarchy",
    70: "Final page containing only the document code",
}
COVERAGE_STATUS_LEGEND = {
    "included": "Included",
    "optional": "Optional",
    "not_included": "Not Included",
}
HEADING_COLOURS = {22180, 23195}

# Transcribed from the booklet's contents page. Multiple headings with the
# same start page share one subgroup so their page content is not duplicated.
CONTENTS_GROUPS = (
    {
        "group_heading": "Guide to your Home Insurance cover",
        "subgroups": (
            (1, ("Welcome to your Home Insurance cover",)),
            (4, ("Definitions",)),
            (8, ("How to make a claim",)),
            (10, ("Keeping your policy up to date",)),
        ),
    },
    {
        "group_heading": "What this policy covers:",
        "subgroups": (
            (12, ("Your buildings cover",)),
            (19, ("Your contents cover",)),
            (29, ("Your liability cover",)),
            (31, ("Cover away from your home",)),
            (33, ("Your bicycle cover",)),
        ),
    },
    {
        "group_heading": (
            "What this policy doesn’t cover and general conditions:"
        ),
        "subgroups": (
            (34, ("What this policy doesn’t cover",)),
            (37, ("General conditions that apply to your home policy",)),
        ),
    },
    {
        "group_heading": "Optional cover:",
        "subgroups": (
            (
                42,
                (
                    "Home Emergency cover (as standard on Gold)",
                    (
                        "Home Emergency Extra cover "
                        "(as standard on Platinum)"
                    ),
                ),
            ),
            (54, ("Home Legal Expenses (as standard on Platinum)",)),
        ),
    },
    {
        "group_heading": "How to make a complaint:",
        "subgroups": (
            (
                66,
                (
                    "Complaint about your home cover",
                    "Complaint about a claim under your home cover",
                    (
                        "Complaint about your Home Emergency or "
                        "Home Emergency Extra cover"
                    ),
                ),
            ),
        ),
    },
    {
        "group_heading": "How we use your personal information:",
        "subgroups": (
            (67, ("How we use your personal information",)),
        ),
    },
)


def normalise_heading(text: str) -> str:
    """Normalise a block for comparison with a defined contents heading."""
    without_page_numbers = "\n".join(
        line
        for line in text.splitlines()
        if not re.fullmatch(r"\d+", line.strip())
    )
    without_continuation = re.sub(
        r"\s*\(cont\.\)\s*$",
        "",
        without_page_numbers,
        flags=re.IGNORECASE,
    )
    return (
        re.sub(r"\s+", " ", without_continuation)
        .strip()
        .rstrip(":")
        .replace("’", "'")
        .casefold()
    )


DEFINED_HEADINGS = frozenset(
    normalise_heading(heading)
    for group in CONTENTS_GROUPS
    for heading in (
        group["group_heading"],
        *(
            subgroup_heading
            for _, subgroup_headings in group["subgroups"]
            for subgroup_heading in subgroup_headings
        ),
    )
)

# These running page titles are navigation labels used by the PDF. Their
# meaning is already represented by section_heading and subsection_headings,
# so retaining them in page content would duplicate structural context.
RUNNING_PAGE_TITLES = frozenset(
    normalise_heading(title)
    for title in (
        "Guide to Home Emergency cover and Home Emergency Extra cover",
        "Guide to your Home Emergency cover and Home Emergency Extra cover",
        "Guide to your Home Legal Expenses",
    )
)

STRUCTURAL_HEADINGS = DEFINED_HEADINGS | RUNNING_PAGE_TITLES

# These visual legends are represented semantically by each table row's
# cover_status values, so retaining their display text would be redundant.
REDUNDANT_PAGE_TEXT = frozenset(
    {
        "included optional not included",
        "included not included",
        "included optional",
        "guide to home emergency cover and",
        "home emergency extra cover",
    }
)


def is_structural_heading(text: str) -> bool:
    """Return whether text is already represented by the JSON hierarchy.

    Some PDF headings occur twice in the same extracted block because the PDF
    contains overlapping text objects. Checking the individual lines removes
    those duplicated heading blocks without discarding a block that also
    contains genuine paragraph content.
    """
    if normalise_heading(text) in STRUCTURAL_HEADINGS:
        return True

    normalised_lines = [
        normalise_heading(line)
        for line in text.splitlines()
        if line.strip() and not re.fullmatch(r"\d+", line.strip())
    ]
    return bool(normalised_lines) and all(
        line in STRUCTURAL_HEADINGS
        for line in normalised_lines
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract PDFs beginning with HH- into contents-grouped JSON."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the source PDFs.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="JSON file to create.",
    )
    return parser.parse_args()


def find_hh_pdfs(input_directory: Path) -> list[Path]:
    """Find PDFs whose filenames begin with HH-, including subdirectories."""
    return sorted(
        path
        for path in input_directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.casefold() == ".pdf"
            and path.name.casefold().startswith("hh-")
        )
    )


def calculate_sha256(pdf_path: Path) -> str:
    """Calculate the SHA-256 checksum of a source PDF."""
    digest = hashlib.sha256()

    with pdf_path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def extract_document_code(pdf_path: Path) -> str | None:
    """Parse an HH document code from the beginning of the filename."""
    match = re.match(
        r"(HH-\d+-\d+)",
        pdf_path.name,
        flags=re.IGNORECASE,
    )

    return match.group(1).upper() if match else None


def extract_document_title(document: pymupdf.Document) -> str | None:
    """Extract the policy booklet title from the first PDF page."""
    first_page_text = clean_text(document[0].get_text("text", sort=True))
    match = re.search(
        r"Guide to your\s+Home Insurance cover",
        first_page_text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return "Guide to your Home Insurance cover"


def clean_text(text: str) -> str:
    """Normalise whitespace while retaining meaningful line boundaries."""
    cleaned_lines = []

    for line in text.splitlines():
        line = line.replace("\x07", "")
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        # This PDF's embedded font maps its bullet glyph to a lowercase "y".
        # Replace only a standalone line-start marker, not words such as
        # "you", "your" or "year".
        cleaned_line = re.sub(r"^y\s+", "• ", cleaned_line)

        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)


def printed_page_number(pdf_page_number: int) -> int | None:
    """Convert a physical PDF page number to the booklet page number."""
    if pdf_page_number <= UNNUMBERED_FRONT_PAGES:
        return None

    return pdf_page_number - UNNUMBERED_FRONT_PAGES


def classify_status_colour(
    fill: tuple[float, float, float] | None,
) -> str | None:
    """Classify a coverage icon from its filled background colour."""
    if fill is None:
        return None

    red, green, blue = fill

    if green > 0.40 and red < 0.20 and blue < 0.30:
        return "included"

    if red > 0.75 and 0.20 < green < 0.55 and blue < 0.30:
        return "optional"

    if red > 0.70 and green < 0.20 and blue > 0.25:
        return "not_included"

    return None


def extract_status_icons(page: pymupdf.Page) -> list[dict[str, Any]]:
    """Extract coloured coverage icons for internal table matching."""
    icons = []

    for drawing in page.get_drawings():
        status = classify_status_colour(drawing.get("fill"))
        rectangle = drawing.get("rect")

        if status is None or rectangle is None:
            continue

        if not (
            10 <= rectangle.width <= 25
            and 10 <= rectangle.height <= 25
        ):
            continue

        icons.append(
            {
                "status": status,
                "centre_x": (rectangle.x0 + rectangle.x1) / 2,
                "centre_y": (rectangle.y0 + rectangle.y1) / 2,
            }
        )

    return icons


def status_in_cell(
    cell: tuple[float, float, float, float] | None,
    icons: list[dict[str, Any]],
) -> str | None:
    """Return the status icon whose centre lies inside a table cell."""
    if cell is None:
        return None

    x0, y0, x1, y1 = cell
    matches = [
        icon["status"]
        for icon in icons
        if (
            x0 <= icon["centre_x"] <= x1
            and y0 <= icon["centre_y"] <= y1
        )
    ]

    return matches[0] if matches else None


def clean_cell(value: str | None) -> str | None:
    """Clean extracted table-cell text while preserving an empty cell."""
    if value is None:
        return None

    cleaned = clean_text(value)
    return cleaned or None


def split_bullet_items(value: str | None) -> list[str]:
    """Split a table cell's bullet list into complete individual items."""
    if not value:
        return []

    return [
        re.sub(r"\s+", " ", item).strip()
        for item in re.split(r"(?m)^•\s*", value)
        if item.strip()
    ]


def parse_change_notification_row(
    label_value: str | None,
    details_value: str | None,
) -> dict[str, Any]:
    """Parse one row of the change-notification requirements table."""
    label = re.sub(r"\s+", " ", label_value or "").strip()
    normalised_label = normalise_heading(label)
    timing_by_label = {
        "please tell us beforehand if": "beforehand",
        "please tell us immediately if": "immediately",
        "please tell us when you renew if": "when_you_renew",
    }
    details = details_value or ""
    conditions_text, separator, notes_text = details.partition(
        "\nGood to know\n"
    )

    requirements = {
        "conditions": split_bullet_items(conditions_text),
    }
    if separator:
        requirements["good_to_know"] = split_bullet_items(notes_text)

    return {
        "notification_timing": timing_by_label.get(normalised_label),
        "label": label,
        "requirements": requirements,
    }


def is_important_border_colour(
    colour: tuple[float, float, float] | None,
) -> bool:
    """Return whether a drawing uses the booklet's pink notice colour."""
    if colour is None:
        return False

    red, green, blue = colour
    return red > 0.70 and green < 0.20 and blue > 0.25


def extract_standalone_important_notices(
    page: pymupdf.Page,
    table_rectangles: list[pymupdf.Rect],
) -> tuple[list[dict[str, Any]], list[pymupdf.Rect]]:
    """Extract pink IMPORTANT boxes that are not rows inside a table."""
    notices = []
    notice_rectangles = []
    text_blocks = page.get_text("blocks", sort=True)

    for drawing in page.get_drawings():
        rectangle = drawing.get("rect")

        if (
            rectangle is None
            or not is_important_border_colour(drawing.get("color"))
            or rectangle.width < 400
            or rectangle.height < 40
        ):
            continue

        if any(
            rectangle.intersects(table_rectangle)
            for table_rectangle in table_rectangles
        ):
            continue

        contained_text = []

        for block in text_blocks:
            x0, y0, x1, y1, text, _, block_type = block

            if block_type != 0:
                continue

            block_rectangle = pymupdf.Rect(x0, y0, x1, y1)
            block_centre = pymupdf.Point(
                (block_rectangle.x0 + block_rectangle.x1) / 2,
                (block_rectangle.y0 + block_rectangle.y1) / 2,
            )

            if rectangle.contains(block_centre):
                cleaned = clean_text(text)

                if cleaned:
                    contained_text.append(cleaned)

        notice_text = "\n\n".join(contained_text)

        if not notice_text:
            continue

        notice_lines = [
            line.strip()
            for line in notice_text.splitlines()
            if line.strip()
        ]

        if notice_lines[0].strip().casefold() != "important":
            continue

        remaining_lines = notice_lines[1:]
        title = None

        if (
            remaining_lines
            and len(remaining_lines[0]) <= 80
            and not remaining_lines[0].endswith((".", ":"))
        ):
            title = remaining_lines.pop(0)

        notices.append(
            {
                "content_type": "important_notice",
                "_x0": rectangle.x0,
                "_y0": rectangle.y0,
                "label": "IMPORTANT",
                "title": title,
                "text": "\n".join(remaining_lines) or None,
            }
        )
        notice_rectangles.append(rectangle)

    return notices, notice_rectangles


def split_table_at_important_notices(
    table_definition: dict[str, Any],
    extracted_rows: list[list[str | None]],
    table: Any,
    table_rectangle: pymupdf.Rect,
) -> list[dict[str, Any]]:
    """Split a table when a full-width IMPORTANT notice occurs mid-table."""
    data_rows = table_definition["rows"]
    notice_indexes = [
        index
        for index, row in enumerate(extracted_rows[1:])
        if (
            clean_cell(row[0]) is not None
            and clean_cell(row[0]).casefold().startswith("important\n")
            and all(clean_cell(value) is None for value in row[1:])
        )
    ]

    if not notice_indexes:
        return [
            {
                "content_type": "table",
                "_x0": table_rectangle.x0,
                "_y0": table_rectangle.y0,
                **table_definition,
            }
        ]

    content_items = []
    segment_start = 0

    for notice_index in notice_indexes:
        if segment_start < notice_index:
            first_row = table.rows[segment_start + 1]
            first_cell = next(
                cell for cell in first_row.cells if cell is not None
            )
            content_items.append(
                {
                    "content_type": "table",
                    "_x0": table_rectangle.x0,
                    "_y0": first_cell[1],
                    **table_definition,
                    "rows": data_rows[segment_start:notice_index],
                }
            )

        notice_text = clean_cell(extracted_rows[notice_index + 1][0])
        notice_lines = notice_text.splitlines() if notice_text else []
        notice_row = table.rows[notice_index + 1]
        notice_cell = next(
            cell for cell in notice_row.cells if cell is not None
        )
        content_items.append(
            {
                "content_type": "important_notice",
                "_x0": notice_cell[0],
                "_y0": notice_cell[1],
                "label": notice_lines[0] if notice_lines else "IMPORTANT",
                "title": notice_lines[1] if len(notice_lines) > 1 else None,
                "text": (
                    "\n".join(notice_lines[2:])
                    if len(notice_lines) > 2
                    else None
                ),
            }
        )
        segment_start = notice_index + 1

    if segment_start < len(data_rows):
        first_row = table.rows[segment_start + 1]
        first_cell = next(
            cell for cell in first_row.cells if cell is not None
        )
        content_items.append(
            {
                "content_type": "table",
                "_x0": table_rectangle.x0,
                "_y0": first_cell[1],
                **table_definition,
                "rows": data_rows[segment_start:],
            }
        )

    return content_items


def vertically_merged_columns(
    table: Any,
    row_index: int,
    column_names: dict[int, str],
) -> list[str]:
    """Return columns whose cells started in an earlier visual row."""
    cells = table.rows[row_index].cells
    row_top = min(
        cell[1]
        for cell in cells
        if cell is not None
    )
    inherited = []

    for column_index, column_name in column_names.items():
        if cells[column_index] is not None:
            continue

        for previous_index in range(row_index - 1, 0, -1):
            previous_cell = table.rows[previous_index].cells[column_index]
            if previous_cell is None:
                continue
            if previous_cell[1] <= row_top < previous_cell[3] - 0.5:
                inherited.append(column_name)
            break

    return inherited


def extract_tables(
    page: pymupdf.Page,
) -> tuple[list[dict[str, Any]], list[pymupdf.Rect]]:
    """Extract table text and associate vector icons with their cells."""
    icons = extract_status_icons(page)
    structured_tables = []
    table_rectangles = []

    for table in page.find_tables().tables:
        extracted_rows = table.extract()

        if not extracted_rows:
            continue

        table_rectangle = pymupdf.Rect(table.bbox)
        table_rectangles.append(table_rectangle)
        header = [clean_cell(value) for value in extracted_rows[0]]
        normalised_header = [
            normalise_heading(value or "")
            for value in header
        ]
        column_count = len(header)

        has_coverage_text_columns = (
            column_count >= 2
            and normalised_header[0].startswith("what is covered")
            and normalised_header[1].startswith("what isn't covered")
        )

        is_change_notification_table = (
            column_count == 2
            and normalised_header[0] == "please tell us beforehand if"
            and len(extracted_rows) == 3
        )

        if is_change_notification_table:
            structured_tables.append(
                {
                    "content_type": "table",
                    "_x0": table_rectangle.x0,
                    "_y0": table_rectangle.y0,
                    "table_type": "change_notification_requirements",
                    "columns": [
                        "notification_timing",
                        "requirements",
                    ],
                    "rows": [
                        parse_change_notification_row(
                            clean_cell(extracted_row[0]),
                            clean_cell(extracted_row[1]),
                        )
                        for extracted_row in extracted_rows
                    ],
                }
            )
            continue

        is_policy_change_permissions_table = (
            column_count == 8
            and len(extracted_rows) >= 4
            and normalise_heading(extracted_rows[0][1] or "")
            == "type of change"
            and normalise_heading(extracted_rows[2][0] or "")
            == "who can make a change"
        )

        if is_policy_change_permissions_table:
            permission_names = [
                re.sub(
                    r"\s+",
                    "_",
                    normalise_heading(value or ""),
                )
                for value in extracted_rows[2][1:]
            ]
            rows = []

            for row_index, extracted_row in enumerate(
                extracted_rows[3:],
                start=3,
            ):
                cells = table.rows[row_index].cells
                rows.append(
                    {
                        "role": re.sub(
                            r"\s+",
                            " ",
                            clean_cell(extracted_row[0]) or "",
                        ).strip(),
                        "permissions": {
                            permission_name: (
                                status_in_cell(cells[column_index], icons)
                                == "included"
                            )
                            for column_index, permission_name in enumerate(
                                permission_names,
                                start=1,
                            )
                        },
                    }
                )

            structured_tables.append(
                {
                    "content_type": "table",
                    "_x0": table_rectangle.x0,
                    "_y0": table_rectangle.y0,
                    "table_type": "policy_change_permissions",
                    "permission_groups": [
                        {
                            "heading": (
                                "Single policies or MultiCover policies"
                            ),
                            "permissions": permission_names[:4],
                        },
                        {
                            "heading": "MultiCover policies only",
                            "permissions": permission_names[4:],
                        },
                    ],
                    "rows": rows,
                }
            )
            continue

        if column_count == 5 and has_coverage_text_columns:
            rows = []

            for row_index, extracted_row in enumerate(
                extracted_rows[1:],
                start=1,
            ):
                cells = table.rows[row_index].cells
                row = {
                        "what_is_covered": clean_cell(extracted_row[0]),
                        "what_is_not_covered": clean_cell(
                            extracted_row[1]
                        ),
                        "cover_status": {
                            cover_level: status_in_cell(
                                cells[column_index],
                                icons,
                            )
                            for column_index, cover_level in enumerate(
                                COVER_LEVELS,
                                start=2,
                            )
                        },
                    }
                inherited = vertically_merged_columns(
                    table,
                    row_index,
                    {0: "what_is_covered", 1: "what_is_not_covered"},
                )
                if inherited:
                    row["_inherit_columns"] = inherited
                rows.append(row)

            table_definition = {
                    "table_type": "coverage_by_level",
                    "columns": [
                        "what_is_covered",
                        "what_is_not_covered",
                        *COVER_LEVELS,
                    ],
                    "rows": rows,
                }
            structured_tables.extend(
                split_table_at_important_notices(
                    table_definition=table_definition,
                    extracted_rows=extracted_rows,
                    table=table,
                    table_rectangle=table_rectangle,
                )
            )
            continue

        if column_count == 2 and has_coverage_text_columns:
            rows = []
            for row_index, extracted_row in enumerate(
                extracted_rows[1:],
                start=1,
            ):
                row = {
                    "what_is_covered": clean_cell(extracted_row[0]),
                    "what_is_not_covered": clean_cell(extracted_row[1]),
                }
                inherited = vertically_merged_columns(
                    table,
                    row_index,
                    {0: "what_is_covered", 1: "what_is_not_covered"},
                )
                if inherited:
                    row["_inherit_columns"] = inherited
                rows.append(row)

            structured_tables.append(
                {
                    "content_type": "table",
                    "_x0": table_rectangle.x0,
                    "_y0": table_rectangle.y0,
                    "table_type": "coverage_details",
                    "columns": [
                        "what_is_covered",
                        "what_is_not_covered",
                    ],
                    "rows": rows,
                }
            )
            continue

        if (
            column_count == 4
            and has_coverage_text_columns
            and normalised_header[2:] == ["hec", "hex"]
        ):
            rows = []

            for row_index, extracted_row in enumerate(
                extracted_rows[1:],
                start=1,
            ):
                cells = table.rows[row_index].cells
                row = {
                        "what_is_covered": clean_cell(extracted_row[0]),
                        "what_is_not_covered": clean_cell(
                            extracted_row[1]
                        ),
                        "cover_status": {
                            "hec": status_in_cell(cells[2], icons),
                            "hex": status_in_cell(cells[3], icons),
                        },
                    }
                inherited = vertically_merged_columns(
                    table,
                    row_index,
                    {0: "what_is_covered", 1: "what_is_not_covered"},
                )
                if inherited:
                    row["_inherit_columns"] = inherited
                rows.append(row)

            table_definition = {
                    "table_type": "emergency_coverage_details",
                    "columns": [
                        "what_is_covered",
                        "what_is_not_covered",
                        "hec",
                        "hex",
                    ],
                    "rows": rows,
                }
            structured_tables.extend(
                split_table_at_important_notices(
                    table_definition=table_definition,
                    extracted_rows=extracted_rows,
                    table=table,
                    table_rectangle=table_rectangle,
                )
            )
            continue

        if (
            column_count == 3
            and normalised_header[1:] == ["hec", "hex"]
        ):
            rows = []

            for row_index, extracted_row in enumerate(
                extracted_rows[1:],
                start=1,
            ):
                cells = table.rows[row_index].cells
                rows.append(
                    {
                        "cover_level": clean_cell(extracted_row[0]),
                        "cover_status": {
                            "hec": status_in_cell(cells[1], icons),
                            "hex": status_in_cell(cells[2], icons),
                        },
                    }
                )

            structured_tables.append(
                {
                    "content_type": "table",
                    "_x0": table_rectangle.x0,
                    "_y0": table_rectangle.y0,
                    "table_type": "emergency_cover_by_level",
                    "columns": ["cover_level", "hec", "hex"],
                    "rows": rows,
                }
            )
            continue

        structured_tables.append(
            {
                "content_type": "table",
                "_x0": table_rectangle.x0,
                "_y0": table_rectangle.y0,
                "table_type": "generic",
                "columns": header,
                "rows": [
                    [clean_cell(value) for value in row]
                    for row in extracted_rows[1:]
                ],
            }
        )

    return structured_tables, table_rectangles


def extract_text_blocks(
    page: pymupdf.Page,
    excluded_rectangles: list[pymupdf.Rect],
) -> list[dict[str, Any]]:
    """Extract semantic headings and paragraphs in visual reading order."""
    text_blocks = []

    for block in page.get_text("dict", sort=True)["blocks"]:
        # PyMuPDF block type 0 represents text; type 1 represents an image.
        if block.get("type") != 0:
            continue

        x0, y0, x1, y1 = block["bbox"]
        block_rectangle = pymupdf.Rect(block["bbox"])

        if any(
            block_rectangle.intersects(table_rectangle)
            for table_rectangle in excluded_rectangles
        ):
            continue

        lines = block.get("lines", [])
        spans = [
            span
            for line in lines
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]
        block_text = "\n".join(
            "".join(
                span.get("text", "")
                for span in line.get("spans", [])
            ).rstrip()
            for line in lines
        )
        cleaned_text = clean_text(block_text)

        if not cleaned_text:
            continue

        if normalise_heading(cleaned_text) in REDUNDANT_PAGE_TEXT:
            continue

        if is_structural_heading(cleaned_text):
            continue

        maximum_font_size = max(
            (float(span.get("size", 0.0)) for span in spans),
            default=0.0,
        )
        all_emphasised = bool(spans) and all(
            any(weight in span.get("font", "").casefold() for weight in (
                "bold",
                "semibold",
            ))
            for span in spans
        )
        all_semibold = bool(spans) and all(
            "semibold" in span.get("font", "").casefold()
            for span in spans
        )
        uses_heading_colour = any(
            span.get("color") in HEADING_COLOURS
            for span in spans
        )
        uses_only_heading_colours = bool(spans) and all(
            span.get("color") in HEADING_COLOURS
            for span in spans
        )
        is_compact_coloured_label = (
            maximum_font_size >= 9.4
            and uses_only_heading_colours
            and len(cleaned_text) <= 120
            and not cleaned_text.rstrip().endswith((".", "!", "?", ":", ";"))
        )
        is_compact_semibold_label = (
            maximum_font_size >= 9.4
            and all_semibold
            and len(cleaned_text) <= 100
            and not cleaned_text.rstrip().endswith((".", "!", "?", ":", ";"))
        )
        is_heading = (
            maximum_font_size >= 14.0
            or (
                maximum_font_size >= 10.0
                and all_emphasised
            )
            or (
                maximum_font_size >= 9.4
                and all_emphasised
                and uses_heading_colour
            )
            or is_compact_semibold_label
            or is_compact_coloured_label
        )

        if is_heading:
            if maximum_font_size >= 14.0:
                heading_level = 2
            elif (
                maximum_font_size >= 11.0
                or uses_heading_colour
                or is_compact_coloured_label
            ):
                heading_level = 3
            else:
                heading_level = 4

            text_blocks.append(
                {
                    "content_type": "heading",
                    "heading_level": heading_level,
                    "text": cleaned_text,
                    "_x0": x0,
                    "_y0": y0,
                }
            )
            continue

        text_blocks.append(
            {
                "content_type": "paragraph",
                "_x0": x0,
                "_y0": y0,
                "text": cleaned_text,
            }
        )

    return text_blocks


def order_page_content(
    text_blocks: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    notices: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Interleave text and tables in their visual page order."""
    ordered_items = sorted(
        [*text_blocks, *tables, *notices],
        key=lambda item: (item["_y0"], item["_x0"]),
    )
    content = []

    for order, item in enumerate(ordered_items, start=1):
        output_item = {
            key: value
            for key, value in item.items()
            if key not in {"_x0", "_y0"}
        }
        output_item["order"] = order
        content.append(output_item)

    return content


CONTINUATION_START = re.compile(
    r"^\s*\(?\s*cont(?:inued)?\s*\.{0,3}\s*\)?\s*",
    flags=re.IGNORECASE,
)
CONTINUATION_END = re.compile(
    r"\s*\(?\s*cont(?:inued)?\s*\.{1,3}\s*\)?\s*$",
    flags=re.IGNORECASE,
)
CONTINUABLE_TABLE_TYPES = {
    "coverage_by_level",
    "coverage_details",
    "emergency_coverage_details",
}
CONTINUABLE_COLUMNS = ("what_is_covered", "what_is_not_covered")


def starts_with_continuation_marker(value: Any) -> bool:
    """Return whether a cell explicitly starts with a continuation marker."""
    return isinstance(value, str) and CONTINUATION_START.match(value) is not None


def join_continued_cell(previous: str, continuation: str) -> str:
    """Join two parts of a table cell and remove printed cont. markers."""
    previous_without_marker = CONTINUATION_END.sub("", previous).rstrip()
    continuation_without_marker = CONTINUATION_START.sub(
        "",
        continuation,
        count=1,
    ).lstrip()
    return "\n".join(
        part
        for part in (previous_without_marker, continuation_without_marker)
        if part
    )


def row_has_meaningful_content(row: dict[str, Any]) -> bool:
    """Return whether a coverage row still contains text or a status."""
    if any(row.get(column) for column in CONTINUABLE_COLUMNS):
        return True

    statuses = row.get("cover_status")
    return isinstance(statuses, dict) and any(
        status is not None for status in statuses.values()
    )


def consolidate_cross_page_table_continuations(
    pages: list[dict[str, Any]],
) -> None:
    """Merge explicitly marked table continuations into the preceding row.

    A continuation is accepted only from the same table type on the same or
    immediately following PDF page. This keeps the merge deterministic while
    supporting tables split by a page break or an IMPORTANT notice.
    """
    previous_cells: dict[
        tuple[str, str],
        tuple[dict[str, Any], int],
    ] = {}

    for page in pages:
        pdf_page_number = page["pdf_page_number"]
        retained_content = []

        for item in page["content"]:
            table_type = item.get("table_type")
            if (
                item.get("content_type") != "table"
                or table_type not in CONTINUABLE_TABLE_TYPES
            ):
                retained_content.append(item)
                continue

            retained_rows = []
            for row in item.get("rows", []):
                merged = False

                for column in CONTINUABLE_COLUMNS:
                    value = row.get(column)
                    previous_entry = previous_cells.get(
                        (table_type, column)
                    )
                    if (
                        starts_with_continuation_marker(value)
                        and previous_entry is not None
                    ):
                        previous_row, previous_pdf_page = previous_entry
                        previous_value = previous_row.get(column)
                        if (
                            pdf_page_number - previous_pdf_page <= 1
                            and isinstance(previous_value, str)
                        ):
                            previous_row[column] = join_continued_cell(
                                previous_value,
                                value,
                            )
                            previous_row[
                                "continued_through_pdf_page"
                            ] = pdf_page_number
                            previous_row[
                                "continued_through_printed_page"
                            ] = page["printed_page_number"]
                            previous_cells[(table_type, column)] = (
                                previous_row,
                                pdf_page_number,
                            )
                            row[column] = None
                            merged = True

                if row_has_meaningful_content(row):
                    retained_rows.append(row)
                    for column in CONTINUABLE_COLUMNS:
                        if row.get(column):
                            previous_cells[(table_type, column)] = (
                                row,
                                pdf_page_number,
                            )
                elif not merged:
                    retained_rows.append(row)

            if retained_rows:
                item["rows"] = retained_rows
                retained_content.append(item)

        for order, item in enumerate(retained_content, start=1):
            item["order"] = order
        page["content"] = retained_content


def duplicate_vertically_merged_cells(
    pages: list[dict[str, Any]],
) -> None:
    """Copy each completed vertically merged cell into every row it covers."""
    for page in pages:
        for item in page["content"]:
            if item.get("content_type") != "table":
                continue

            previous_values: dict[str, dict[str, Any]] = {}
            for row in item.get("rows", []):
                if not isinstance(row, dict):
                    continue

                inherited_columns = row.pop("_inherit_columns", [])
                for column in inherited_columns:
                    source_row = previous_values.get(column)
                    if source_row is None:
                        continue

                    row[column] = source_row.get(column)
                    for provenance_key in (
                        "continued_through_pdf_page",
                        "continued_through_printed_page",
                    ):
                        if provenance_key in source_row:
                            row[provenance_key] = source_row[provenance_key]

                for column in CONTINUABLE_COLUMNS:
                    if row.get(column):
                        previous_values[column] = row


def extract_pages(
    document: pymupdf.Document,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract every page and collect unexpected low-text warnings."""
    pages = []
    warnings = []

    for page_index, page in enumerate(document):
        pdf_page_number = page_index + 1
        booklet_page_number = printed_page_number(pdf_page_number)
        tables, table_rectangles = extract_tables(page)
        notices, notice_rectangles = extract_standalone_important_notices(
            page,
            table_rectangles,
        )
        excluded_rectangles = [*table_rectangles, *notice_rectangles]
        blocks = extract_text_blocks(page, excluded_rectangles)
        page_content = order_page_content(blocks, tables, notices)

        # Assess extraction health using the raw PDF text rather than the
        # filtered JSON content. Otherwise a page made mostly of deliberately
        # removed structural headings (for example, the contents page) can
        # produce a false low-text warning.
        raw_page_text = page.get_text("text")
        character_count = len(re.sub(r"\s+", "", raw_page_text))

        pages.append(
            {
                "pdf_page_number": pdf_page_number,
                "printed_page_number": booklet_page_number,
                "content": page_content,
            }
        )

        if (
            character_count < MINIMUM_PAGE_CHARACTERS
            and pdf_page_number != 70
        ):
            warnings.append(
                {
                    "pdf_page_number": pdf_page_number,
                    "printed_page_number": booklet_page_number,
                    "character_count": character_count,
                    "reason": "Little or no text was extracted",
                }
            )

    return pages, warnings


def get_subgroup_end_pages() -> dict[int, int]:
    """Calculate each subgroup's end from the next distinct start page."""
    start_pages = [
        start_page
        for group in CONTENTS_GROUPS
        for start_page, _ in group["subgroups"]
    ]
    end_pages = {}

    for index, start_page in enumerate(start_pages):
        if index + 1 < len(start_pages):
            end_pages[start_page] = start_pages[index + 1] - 1
        else:
            end_pages[start_page] = FINAL_PRINTED_PAGE

    return end_pages


def inclusive_page_numbers(start: int, end: int) -> list[int]:
    """Return an inclusive list of page numbers."""
    return list(range(start, end + 1))


def flatten_subsection_content(
    subsection_pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten page content while retaining item-level source provenance."""
    subsection_content = []

    for page in subsection_pages:
        pdf_page_number = page["pdf_page_number"]
        printed_page = page["printed_page_number"]

        for item in page["content"]:
            item.pop("order", None)
            last_pdf_page = pdf_page_number
            last_printed_page = printed_page

            if item.get("content_type") == "table":
                for row in item.get("rows", []):
                    if not isinstance(row, dict):
                        continue

                    row_last_pdf_page = row.pop(
                        "continued_through_pdf_page",
                        pdf_page_number,
                    )
                    row_last_printed_page = row.pop(
                        "continued_through_printed_page",
                        printed_page,
                    )
                    row["source_pdf_pages"] = inclusive_page_numbers(
                        pdf_page_number,
                        row_last_pdf_page,
                    )
                    row["source_printed_pages"] = inclusive_page_numbers(
                        printed_page,
                        row_last_printed_page,
                    )
                    last_pdf_page = max(last_pdf_page, row_last_pdf_page)
                    last_printed_page = max(
                        last_printed_page,
                        row_last_printed_page,
                    )

            item["source_pdf_pages"] = inclusive_page_numbers(
                pdf_page_number,
                last_pdf_page,
            )
            item["source_printed_pages"] = inclusive_page_numbers(
                printed_page,
                last_printed_page,
            )
            item["order"] = len(subsection_content) + 1
            subsection_content.append(item)

    return subsection_content


def group_pages_into_sections(
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create sections with page-independent subsection content."""
    pages_by_printed_number = {
        page["printed_page_number"]: page
        for page in pages
        if (
            page["printed_page_number"] is not None
            and page["pdf_page_number"] not in EXCLUDED_PDF_PAGES
        )
    }
    subgroup_end_pages = get_subgroup_end_pages()
    sections = []

    for group_definition in CONTENTS_GROUPS:
        subsections = []

        for start_page, headings in group_definition["subgroups"]:
            end_page = subgroup_end_pages[start_page]
            subsection_pages = [
                pages_by_printed_number[page_number]
                for page_number in range(start_page, end_page + 1)
                if page_number in pages_by_printed_number
            ]

            subsections.append(
                {
                    "subsection_headings": list(headings),
                    "start_printed_page": start_page,
                    "end_printed_page": end_page,
                    "start_pdf_page": start_page + UNNUMBERED_FRONT_PAGES,
                    "end_pdf_page": end_page + UNNUMBERED_FRONT_PAGES,
                    "content": flatten_subsection_content(subsection_pages),
                }
            )

        sections.append(
            {
                "section_heading": group_definition["group_heading"],
                "subsections": subsections,
            }
        )

    return sections


def validate_page_assignments(
    pages: list[dict[str, Any]],
    excluded_pages: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> None:
    """Ensure the declared subsection ranges account for every source page."""
    assigned_page_numbers = [
        page["pdf_page_number"]
        for page in excluded_pages
    ]
    assigned_page_numbers.extend(
        pdf_page_number
        for section in sections
        for subsection in section["subsections"]
        for pdf_page_number in inclusive_page_numbers(
            subsection["start_pdf_page"],
            subsection["end_pdf_page"],
        )
    )
    expected_page_numbers = [
        page["pdf_page_number"]
        for page in pages
    ]

    if sorted(assigned_page_numbers) != sorted(expected_page_numbers):
        raise ValueError(
            "The contents structure did not assign every PDF page exactly once"
        )

    if len(assigned_page_numbers) != len(set(assigned_page_numbers)):
        raise ValueError(
            "The contents structure assigned at least one PDF page more than once"
        )


def extract_document(pdf_path: Path) -> dict[str, Any]:
    """Extract one HH booklet into the required JSON hierarchy."""
    with pymupdf.open(pdf_path) as document:
        if document.page_count == 0:
            raise ValueError("The PDF contains no pages")

        title = extract_document_title(document)
        pages, warnings = extract_pages(document)
        consolidate_cross_page_table_continuations(pages)
        duplicate_vertically_merged_cells(pages)
        page_count = document.page_count

    excluded_pages = [
        {
            "pdf_page_number": pdf_page_number,
            "reason": reason,
        }
        for pdf_page_number, reason in EXCLUDED_PDF_PAGES.items()
    ]
    sections = group_pages_into_sections(pages)
    validate_page_assignments(pages, excluded_pages, sections)

    return {
        "document_name": pdf_path.name,
        "document_type": "home_insurance_policy_booklet",
        "document_code": extract_document_code(pdf_path),
        "title": title,
        "sha256": calculate_sha256(pdf_path),
        "source_files": [pdf_path.name],
        "page_count": page_count,
        "excluded_pages": excluded_pages,
        "coverage_status_legend": COVERAGE_STATUS_LEGEND,
        "sections": sections,
        "warnings": warnings,
    }


def main() -> int:
    """Extract all matching HH- PDF files."""
    arguments = parse_arguments()
    input_directory = arguments.input_dir.expanduser().resolve()
    output_file = arguments.output_file.expanduser().resolve()

    if not input_directory.is_dir():
        print(
            f"ERROR: Input directory not found: {input_directory}",
            file=sys.stderr,
        )
        return 1

    pdf_paths = find_hh_pdfs(input_directory)

    if not pdf_paths:
        print(
            f"ERROR: No PDF files beginning with HH- found in "
            f"{input_directory}",
            file=sys.stderr,
        )
        return 1

    documents = []
    failures = []

    for pdf_path in pdf_paths:
        print(f"Extracting: {pdf_path.name}")

        try:
            documents.append(extract_document(pdf_path))
        except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
            print(f"ERROR: {pdf_path.name}: {error}", file=sys.stderr)
            failures.append(
                {
                    "document_name": pdf_path.name,
                    "reason": str(error),
                }
            )

    warnings = [
        {
            "document_name": document["document_name"],
            **warning,
        }
        for document in documents
        for warning in document["warnings"]
    ]
    output = {
        "summary": {
            "pdf_files_found": len(pdf_paths),
            "documents_extracted": len(documents),
            "documents_failed": len(failures),
            "pages_extracted": sum(
                document["page_count"]
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
    print(f"PDF files found:     {len(pdf_paths)}")
    print(f"Documents extracted: {len(documents)}")
    print(f"Documents failed:    {len(failures)}")
    print(f"Pages extracted:     {output['summary']['pages_extracted']}")
    print(f"Warnings:            {len(warnings)}")
    print(f"Output:              {output_file}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
