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
    2: "Contents page represented by the section hierarchy",
    70: "Final page containing only the document code",
}
COVERAGE_STATUS_LEGEND = {
    "included": "Included",
    "optional": "Optional",
    "not_included": "Not Included",
}
HEADING_COLOURS = {22180, 23195}
LIST_ITEM_START = re.compile(
    r"^\s*(?P<marker>"
    r"•|"
    r"\(?\d+\)(?:\.)?|"
    r"\(?[a-z]\)(?:\.)?|"
    r"\d+\.|"
    r"[ivxlcdm]+\.|"
    r"[a-z]\."
    r")\s+(?P<text>.+)$",
    flags=re.IGNORECASE | re.DOTALL,
)

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

DEFINED_HEADING_PRINTED_PAGES: dict[str, set[int]] = {}
for group in CONTENTS_GROUPS:
    group_start_page = group["subgroups"][0][0]
    DEFINED_HEADING_PRINTED_PAGES.setdefault(
        normalise_heading(group["group_heading"]),
        set(),
    ).add(group_start_page)
    for start_page, subgroup_headings in group["subgroups"]:
        for subgroup_heading in subgroup_headings:
            DEFINED_HEADING_PRINTED_PAGES.setdefault(
                normalise_heading(subgroup_heading),
                set(),
            ).add(start_page)

# These running page titles are navigation labels used by the PDF. Their
# meaning is already represented by section_heading and subsection_headings,
# so retaining them in page content would duplicate structural context.
RUNNING_PAGE_TITLES = frozenset(
    normalise_heading(title)
    for title in (
        "Guide to your Home Insurance cover",
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

def is_structural_heading(
    text: str,
    printed_page: int | None,
) -> bool:
    """Return whether text is already represented by the JSON hierarchy.

    Some PDF headings occur twice in the same extracted block because the PDF
    contains overlapping text objects. Checking the individual lines removes
    those duplicated heading blocks without discarding a block that also
    contains genuine paragraph content.
    """
    normalised_text = normalise_heading(text)
    if normalised_text in RUNNING_PAGE_TITLES:
        return True
    if (
        normalised_text in DEFINED_HEADINGS
        and printed_page
        in DEFINED_HEADING_PRINTED_PAGES.get(normalised_text, set())
    ):
        return True

    heading_lines = [
        (line, normalise_heading(line))
        for line in text.splitlines()
        if line.strip() and not re.fullmatch(r"\d+", line.strip())
    ]
    return bool(heading_lines) and all(
        normalised_line in RUNNING_PAGE_TITLES
        or (
            normalised_line in DEFINED_HEADINGS
            and (
                printed_page
                in DEFINED_HEADING_PRINTED_PAGES.get(
                    normalised_line,
                    set(),
                )
                or re.search(
                    r"\(cont\.\)\s*$",
                    original_line,
                    flags=re.IGNORECASE,
                )
                is not None
            )
        )
        for original_line, normalised_line in heading_lines
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


def extract_document_summary(
    document: pymupdf.Document,
) -> str | None:
    """Extract only the text inside the bordered title-page summary box."""
    page = document[0]
    candidate_boxes = []

    for drawing in page.get_drawings():
        rectangle = drawing.get("rect")
        colour = drawing.get("color")

        if rectangle is None or colour is None:
            continue

        red, green, blue = colour
        is_summary_border = (
            rectangle.width > page.rect.width * 0.50
            and rectangle.height > page.rect.height * 0.15
            and red > 0.70
            and green < 0.20
            and blue > 0.20
        )

        if is_summary_border:
            candidate_boxes.append(rectangle)

    if not candidate_boxes:
        return None

    summary_box = max(candidate_boxes, key=lambda rectangle: rectangle.get_area())
    summary = clean_text(page.get_text("text", clip=summary_box, sort=True))
    return summary or None


def create_document_summary_section(summary: str) -> dict[str, Any]:
    """Represent title-page summary text using the standard section schema."""
    return {
        "section_heading": "Document summary",
        "subsections": [
            {
                "subsection_headings": [],
                "start_printed_page": None,
                "end_printed_page": None,
                "start_pdf_page": 1,
                "end_pdf_page": 1,
                "content": [
                    {
                        "content_type": "paragraph",
                        "text": summary,
                        "source_pdf_pages": [1],
                        "source_printed_pages": [],
                        "order": 1,
                    }
                ],
            }
        ],
    }


def clean_text(text: str) -> str:
    """Normalise whitespace while retaining meaningful line boundaries."""
    cleaned_lines = []
    pending_bullet = False

    for line in text.splitlines():
        line = line.replace("\x07", "")
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        # This PDF's embedded font maps its bullet glyph to a lowercase "y".
        # Depending on the extraction route, the glyph is returned either at
        # the beginning of the bullet text or on a line by itself.  Defer a
        # standalone marker and attach it to the next non-empty line.  Exact
        # marker matching avoids changing words such as "you" or "year".
        if cleaned_line == "y":
            pending_bullet = True
            continue

        cleaned_line = re.sub(r"^y\s+", "• ", cleaned_line)

        if cleaned_line:
            if pending_bullet:
                cleaned_line = f"• {cleaned_line}"
                pending_bullet = False
            cleaned_lines.append(cleaned_line)

    if pending_bullet:
        cleaned_lines.append("•")

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


def extract_table_row_heading(
    page: pymupdf.Page,
    cell: Any,
) -> str | None:
    """Extract a visually styled heading at the start of a table cell.

    Coverage-table row headings use the booklet's blue heading colour and a
    bold or semibold font.  Reading those style attributes avoids treating an
    ordinary first sentence as a heading.  Consecutive styled lines are joined
    so a heading that wraps visually is still stored as one value.
    """
    if cell is None:
        return None

    text = page.get_text(
        "dict",
        clip=pymupdf.Rect(cell),
        sort=True,
    )
    heading_lines: list[str] = []

    for block in text.get("blocks", []):
        for line in block.get("lines", []):
            spans = [
                span
                for span in line.get("spans", [])
                if span.get("text", "").strip()
            ]
            if not spans:
                continue

            is_emphasised = all(
                any(
                    weight in span.get("font", "").casefold()
                    for weight in ("bold", "semibold")
                )
                for span in spans
            )
            uses_heading_colour = all(
                span.get("color") in HEADING_COLOURS
                for span in spans
            )

            if not (is_emphasised and uses_heading_colour):
                return (
                    re.sub(r"\s+", " ", " ".join(heading_lines)).strip()
                    or None
                )

            line_text = clean_text(
                "".join(span.get("text", "") for span in spans)
            )
            if line_text:
                heading_lines.append(line_text)

    return re.sub(r"\s+", " ", " ".join(heading_lines)).strip() or None


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

    # Use only stroked vector lines when constructing the table grid.  The
    # booklet contains coloured heading backgrounds and other filled shapes;
    # the default ``lines`` strategy also treats rectangle borders from those
    # shapes as table edges.  That can corrupt the detected grid and cause two
    # adjacent coverage rows to be returned as one cell (for example,
    # "Household removal and temporary storage" followed by
    # "Contents temporarily away from home").
    table_finder = page.find_tables(
        vertical_strategy="lines_strict",
        horizontal_strategy="lines_strict",
    )

    for table in table_finder.tables:
        # ``Table.extract()`` orders the characters in a cell using their
        # horizontal coordinates.  In this booklet some font glyphs overlap
        # slightly (notably ``fi``), so coordinate ordering can transpose
        # otherwise valid words: ``find`` becomes ``fnid``, ``office`` becomes
        # ``ofcfie``, and so on.  The normal page text engine preserves the
        # PDF's logical character order.  Keep the table detector's cell
        # geometry, but read each cell through that text engine instead.
        extracted_rows = [
            [
                (
                    page.get_textbox(pymupdf.Rect(cell)).strip()
                    if cell is not None
                    else None
                )
                for cell in row.cells
            ]
            for row in table.rows
        ]

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
                row_heading = extract_table_row_heading(page, cells[0])
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
                if row_heading:
                    row = {"row_heading": row_heading, **row}
                inherited = vertically_merged_columns(
                    table,
                    row_index,
                    {0: "what_is_covered", 1: "what_is_not_covered"},
                )
                inherited = [
                    column for column in inherited if not row.get(column)
                ]
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
                cells = table.rows[row_index].cells
                row_heading = extract_table_row_heading(page, cells[0])
                row = {
                    "what_is_covered": clean_cell(extracted_row[0]),
                    "what_is_not_covered": clean_cell(extracted_row[1]),
                }
                if row_heading:
                    row = {"row_heading": row_heading, **row}
                inherited = vertically_merged_columns(
                    table,
                    row_index,
                    {0: "what_is_covered", 1: "what_is_not_covered"},
                )
                inherited = [
                    column for column in inherited if not row.get(column)
                ]
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
                row_heading = extract_table_row_heading(page, cells[0])
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
                if row_heading:
                    row = {"row_heading": row_heading, **row}
                inherited = vertically_merged_columns(
                    table,
                    row_index,
                    {0: "what_is_covered", 1: "what_is_not_covered"},
                )
                inherited = [
                    column for column in inherited if not row.get(column)
                ]
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

        if is_structural_heading(
            cleaned_text,
            printed_page_number(page.number + 1),
        ):
            continue

        # A definition heading and its first enumerated item can share one PDF
        # text block even though they occupy separate visual columns.  Split a
        # semibold left-column first line from a right-column list item before
        # applying whole-block heading classification.
        if len(lines) >= 2:
            first_line = lines[0]
            first_line_spans = [
                span
                for span in first_line.get("spans", [])
                if span.get("text", "").strip()
            ]
            first_line_text = clean_text(
                "".join(
                    span.get("text", "")
                    for span in first_line.get("spans", [])
                )
            )
            remaining_text = clean_text(
                "\n".join(
                    "".join(
                        span.get("text", "")
                        for span in line.get("spans", [])
                    ).rstrip()
                    for line in lines[1:]
                )
            )
            remaining_list_match = LIST_ITEM_START.match(remaining_text)
            first_line_is_emphasised = bool(first_line_spans) and all(
                any(
                    weight in span.get("font", "").casefold()
                    for weight in ("bold", "semibold")
                )
                for span in first_line_spans
            )
            first_line_size = max(
                (
                    float(span.get("size", 0.0))
                    for span in first_line_spans
                ),
                default=0.0,
            )
            first_line_uses_heading_colour = bool(
                first_line_spans
            ) and all(
                span.get("color") in HEADING_COLOURS
                for span in first_line_spans
            )
            remaining_spans = [
                span
                for line in lines[1:]
                for span in line.get("spans", [])
                if span.get("text", "").strip()
            ]
            remaining_maximum_font_size = max(
                (
                    float(span.get("size", 0.0))
                    for span in remaining_spans
                ),
                default=0.0,
            )
            first_line_x0, first_line_y0, first_line_x1, first_line_y1 = (
                first_line["bbox"]
            )
            remaining_x0 = min(line["bbox"][0] for line in lines[1:])
            remaining_y0 = min(line["bbox"][1] for line in lines[1:])
            remaining_x1 = max(line["bbox"][2] for line in lines[1:])
            remaining_y1 = max(line["bbox"][3] for line in lines[1:])
            is_mixed_definition_block = (
                first_line_is_emphasised
                and first_line_text
                and first_line_x1 <= 170.0
                and remaining_x0 > first_line_x1
            )
            is_stacked_heading_block = (
                first_line_is_emphasised
                and first_line_uses_heading_colour
                and first_line_size >= 10.0
                and remaining_maximum_font_size
                < first_line_size - 0.25
                and len(first_line_text) <= 120
                and not first_line_text.rstrip().endswith(
                    (".", "!", "?", ":", ";")
                )
            )

            if is_mixed_definition_block or is_stacked_heading_block:
                heading_level = 3 if first_line_size >= 11.0 else 4
                if remaining_list_match is not None:
                    definition_item = {
                        "content_type": "list_item",
                        "list_marker": remaining_list_match.group("marker"),
                        "text": remaining_list_match.group("text").strip(),
                        "_x0": remaining_x0,
                        "_y0": remaining_y0,
                        "_x1": remaining_x1,
                        "_y1": remaining_y1,
                    }
                else:
                    definition_item = {
                        "content_type": "paragraph",
                        "text": remaining_text,
                        "_x0": remaining_x0,
                        "_y0": remaining_y0,
                        "_x1": remaining_x1,
                        "_y1": remaining_y1,
                    }
                text_blocks.extend(
                    (
                        {
                            "content_type": "heading",
                            "heading_level": heading_level,
                            "text": first_line_text,
                            "_font_size": first_line_size,
                            "_font_colour": first_line_spans[0].get("color"),
                            "_x0": first_line_x0,
                            "_y0": first_line_y0,
                            "_x1": first_line_x1,
                            "_y1": first_line_y1,
                        },
                        definition_item,
                    )
                )
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
                    "_font_size": maximum_font_size,
                    "_font_colour": spans[0].get("color"),
                    "_x0": x0,
                    "_y0": y0,
                    "_x1": x1,
                    "_y1": y1,
                }
            )
            continue

        list_item_match = LIST_ITEM_START.match(cleaned_text)
        if list_item_match:
            text_blocks.append(
                {
                    "content_type": "list_item",
                    "list_marker": list_item_match.group("marker"),
                    "text": list_item_match.group("text").strip(),
                    "_x0": x0,
                    "_y0": y0,
                    "_x1": x1,
                    "_y1": y1,
                }
            )
            continue

        text_blocks.append(
            {
                "content_type": "paragraph",
                "_x0": x0,
                "_y0": y0,
                "_x1": x1,
                "_y1": y1,
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
    # On some PyMuPDF/platform combinations, each visual line of a narrow
    # left-column definition heading is returned as a separate block.  Merge
    # vertically adjacent heading fragments before paragraphs are interleaved;
    # otherwise a right-column definition can be sorted between the two lines
    # of headings such as "Reasonable prospects".
    retained_text_blocks = []
    consumed_heading_ids = set()
    headings = sorted(
        (
            item for item in text_blocks
            if item.get("content_type") == "heading"
        ),
        key=lambda item: (item.get("_y0", 0.0), item.get("_x0", 0.0)),
    )

    for heading_index, heading in enumerate(headings):
        if id(heading) in consumed_heading_ids:
            continue

        for continuation in headings[heading_index + 1:]:
            vertical_gap = continuation.get("_y0", 0.0) - heading.get(
                "_y1", 0.0
            )
            # Lines belonging to one wrapped heading sit almost directly
            # beneath each other.  A larger allowance can accidentally join
            # two consecutive one-line definition terms (for example,
            # "Vehicle cloning" and "We, us, our, ARAG").
            if vertical_gap > 4.0:
                break
            if vertical_gap < -1.0:
                continue
            if (
                heading.get("heading_level")
                == continuation.get("heading_level")
                and (
                    "\n" not in heading.get("text", "")
                    or vertical_gap <= 4.0
                )
                and abs(
                    heading.get("_x0", 0.0)
                    - continuation.get("_x0", 0.0)
                ) <= 35.0
                and max(
                    heading.get("_x1", 0.0),
                    continuation.get("_x1", 0.0),
                ) <= 170.0
            ):
                heading["text"] = "\n".join(
                    (heading["text"].rstrip(), continuation["text"].lstrip())
                )
                heading["_x0"] = min(
                    heading.get("_x0", 0.0),
                    continuation.get("_x0", 0.0),
                )
                heading["_x1"] = max(
                    heading.get("_x1", 0.0),
                    continuation.get("_x1", 0.0),
                )
                heading["_y1"] = max(
                    heading.get("_y1", 0.0),
                    continuation.get("_y1", 0.0),
                )
                consumed_heading_ids.add(id(continuation))

    retained_text_blocks = [
        item for item in text_blocks
        if id(item) not in consumed_heading_ids
    ]
    vertically_sorted_items = sorted(
        [*retained_text_blocks, *tables, *notices],
        key=lambda item: (item["_y0"], item["_x0"]),
    )
    ordered_items = []
    visual_band = []
    band_y = None

    for item in vertically_sorted_items:
        item_y = item.get("_y0", 0.0)
        if band_y is None or item_y - band_y <= 3.0:
            visual_band.append(item)
            if band_y is None:
                band_y = item_y
            continue

        ordered_items.extend(
            sorted(visual_band, key=lambda value: value.get("_x0", 0.0))
        )
        visual_band = [item]
        band_y = item_y

    if visual_band:
        ordered_items.extend(
            sorted(visual_band, key=lambda value: value.get("_x0", 0.0))
        )

    # Some platforms place a right-column definition between two visual lines
    # of its narrow left-column heading.  Recognise the heading/definition/
    # heading pattern from column positions and overlapping vertical ranges,
    # then join the heading lines without moving the definition text itself.
    reconstructed_items = []
    item_index = 0
    while item_index < len(ordered_items):
        if item_index + 2 < len(ordered_items):
            first_heading = ordered_items[item_index]
            definition = ordered_items[item_index + 1]
            second_heading = ordered_items[item_index + 2]
            is_split_definition_heading = (
                first_heading.get("content_type") == "heading"
                and definition.get("content_type") == "paragraph"
                and second_heading.get("content_type") == "heading"
                and first_heading.get("heading_level")
                == second_heading.get("heading_level")
                and max(
                    first_heading.get("_x1", 0.0),
                    second_heading.get("_x1", 0.0),
                ) <= 170.0
                and definition.get("_x0", 0.0)
                > max(
                    first_heading.get("_x1", 0.0),
                    second_heading.get("_x1", 0.0),
                )
                and second_heading.get("_y0", 0.0)
                <= definition.get("_y1", 0.0)
            )
            if is_split_definition_heading:
                first_heading["text"] = "\n".join(
                    (
                        first_heading["text"].rstrip(),
                        second_heading["text"].lstrip(),
                    )
                )
                first_heading["_x0"] = min(
                    first_heading.get("_x0", 0.0),
                    second_heading.get("_x0", 0.0),
                )
                first_heading["_x1"] = max(
                    first_heading.get("_x1", 0.0),
                    second_heading.get("_x1", 0.0),
                )
                first_heading["_y1"] = max(
                    first_heading.get("_y1", 0.0),
                    second_heading.get("_y1", 0.0),
                )
                reconstructed_items.extend((first_heading, definition))
                item_index += 3
                continue

        reconstructed_items.append(ordered_items[item_index])
        item_index += 1

    ordered_items = reconstructed_items
    merged_items = []

    for item_index, item in enumerate(ordered_items):
        previous = merged_items[-1] if merged_items else None
        next_item = (
            ordered_items[item_index + 1]
            if item_index + 1 < len(ordered_items)
            else None
        )
        previous_text = previous.get("text", "") if previous else ""
        current_text = item.get("text", "")
        is_same_column_continuation = (
            abs(previous.get("_x0", 0.0) - item.get("_x0", 0.0)) <= 2.0
            and 0.0
            <= item.get("_y0", 0.0) - previous.get("_y1", 0.0)
            <= 4.0
        ) if previous else False
        current_starts_list_item = re.match(
            r"^(?:\(?[a-z0-9]+[.)]\s)",
            current_text.lstrip(),
            flags=re.IGNORECASE,
        ) is not None
        is_short_sentence_ending_fragment = (
            len(previous_text.strip()) >= 60
            and 0 < len(current_text.strip()) <= 80
            and current_text.rstrip().endswith((".", "!", "?"))
            and not current_text.lstrip().startswith(("•", "-"))
            and not current_starts_list_item
        )
        first_current_letter = re.search(r"[A-Za-z]", current_text)
        is_lowercase_continuation = (
            first_current_letter is not None
            and first_current_letter.group(0).islower()
            and not current_text.lstrip().startswith(("•", "-"))
            and not current_starts_list_item
        )
        is_relaxed_list_column_continuation = (
            previous is not None
            and previous.get("content_type") == "list_item"
            and item.get("content_type") == "paragraph"
            and abs(
                previous.get("_x0", 0.0) - item.get("_x0", 0.0)
            ) <= 30.0
            and -2.0
            <= item.get("_y0", 0.0) - previous.get("_y1", 0.0)
            <= 6.0
        )
        is_list_item_continuation = (
            previous is not None
            and previous.get("content_type") == "list_item"
            and item.get("content_type") == "paragraph"
            and not current_starts_list_item
            and first_current_letter is not None
            and first_current_letter.group(0).islower()
        )
        is_paragraph_between_list_items = (
            previous is not None
            and previous.get("content_type") == "list_item"
            and item.get("content_type") == "paragraph"
            and next_item is not None
            and next_item.get("content_type") == "list_item"
            and not current_text.rstrip().endswith(":")
        )
        is_pipe_delimited_list_continuation = (
            previous is not None
            and previous.get("content_type") == "list_item"
            and item.get("content_type") in {"paragraph", "heading"}
            and previous_text.rstrip().endswith("|")
        )
        previous_is_unfinished = (
            not previous_text.rstrip().endswith(
                (".", "!", "?", ":", ";")
            )
            or is_paragraph_between_list_items
            or is_relaxed_list_column_continuation
            or is_pipe_delimited_list_continuation
        )
        can_join_paragraph_fragment = (
            previous is not None
            and previous.get("content_type") in {"paragraph", "list_item"}
            and (
                item.get("content_type") == "paragraph"
                or is_pipe_delimited_list_continuation
            )
            and previous_is_unfinished
            and (
                is_same_column_continuation
                or is_short_sentence_ending_fragment
                or is_lowercase_continuation
                or is_list_item_continuation
                or is_paragraph_between_list_items
                or is_relaxed_list_column_continuation
                or is_pipe_delimited_list_continuation
            )
        )

        if can_join_paragraph_fragment:
            previous["text"] = "\n".join(
                (previous["text"].rstrip(), item["text"].lstrip())
            )
            previous["_x1"] = max(
                previous.get("_x1", 0.0),
                item.get("_x1", 0.0),
            )
            previous["_y1"] = item.get("_y1", previous.get("_y1"))
            continue

        merged_items.append(item)

    # Borderless glossaries use bold terms in a narrow left column and their
    # definitions in a consistent right column.  PyMuPDF does not identify
    # these as tables because there are no drawn grid lines.  Convert runs of
    # two or more term/definition pairs into a semantic two-column table so a
    # term does not behave like an active document heading.
    glossary_items = []
    item_index = 0

    while item_index < len(merged_items):
        rows = []
        run_index = item_index

        while run_index < len(merged_items):
            term = merged_items[run_index]
            is_term = (
                term.get("content_type") == "heading"
                and term.get("heading_level") == 4
                and term.get("_x1", float("inf")) <= 175.0
            )
            if not is_term:
                break

            definition_index = run_index + 1
            definition_items = []
            while definition_index < len(merged_items):
                definition_item = merged_items[definition_index]
                if definition_item.get("content_type") == "heading":
                    break
                if (
                    definition_item.get("content_type")
                    not in {"paragraph", "list_item"}
                    or definition_item.get("_x0", 0.0) < 155.0
                ):
                    definition_items = []
                    break
                definition_items.append(definition_item)
                definition_index += 1

            if not definition_items:
                break

            definition_parts = []
            for definition_item in definition_items:
                definition_text = definition_item.get("text", "").strip()
                if definition_item.get("content_type") == "list_item":
                    marker = definition_item.get("list_marker") or "•"
                    definition_text = f"{marker} {definition_text}".strip()
                if definition_text:
                    definition_parts.append(definition_text)

            rows.append(
                {
                    "term": re.sub(
                        r"\s+",
                        " ",
                        term.get("text", ""),
                    ).strip(),
                    "definition": "\n".join(definition_parts),
                }
            )
            run_index = definition_index

        if len(rows) >= 2:
            run_items = merged_items[item_index:run_index]
            glossary_items.append(
                {
                    "content_type": "table",
                    "table_type": "definitions",
                    "columns": ["term", "definition"],
                    "rows": rows,
                    "_x0": min(item.get("_x0", 0.0) for item in run_items),
                    "_y0": min(item.get("_y0", 0.0) for item in run_items),
                    "_x1": max(item.get("_x1", 0.0) for item in run_items),
                    "_y1": max(item.get("_y1", 0.0) for item in run_items),
                }
            )
            item_index = run_index
            continue

        glossary_items.append(merged_items[item_index])
        item_index += 1

    merged_items = glossary_items
    content = []

    for order, item in enumerate(merged_items, start=1):
        output_item = {
            key: value
            for key, value in item.items()
            if key not in {"_x0", "_y0", "_x1", "_y1"}
        }
        output_item["order"] = order
        content.append(output_item)

    return content


def apply_nested_heading_levels(
    page_content: list[dict[str, Any]],
    hierarchy_state: dict[str, Any],
) -> None:
    """Derive nested headings from numbering and relative typography.

    A numbered level-3 heading can have same-sized unnumbered children. An
    unnumbered level-3 heading can also act as a parent when the headings that
    follow use a smaller font. A heading at least as large as the active
    unnumbered parent starts a new level-3 group. The supplied state is retained
    between pages so a child heading can follow its parent across a page break.
    """
    parent_font_size = hierarchy_state.get("parent_font_size")
    parent_font_colour = hierarchy_state.get("parent_font_colour")
    parent_is_numbered = hierarchy_state.get(
        "parent_is_numbered",
        False,
    )
    level_4_parent_font_size = hierarchy_state.get(
        "level_4_parent_font_size"
    )
    level_4_parent_font_colour = hierarchy_state.get(
        "level_4_parent_font_colour"
    )
    level_4_parent_is_numbered = hierarchy_state.get(
        "level_4_parent_is_numbered",
        False,
    )
    level_4_parent_number = hierarchy_state.get("level_4_parent_number")
    parent_started_on_page = False
    previous_item_was_heading = False

    for item in page_content:
        if item.get("content_type") != "heading":
            previous_item_was_heading = False
            continue

        normalised_text = normalise_heading(item.get("text", ""))
        numbered_match = re.match(r"^(\d+)\.\s+", normalised_text)
        is_numbered = numbered_match is not None
        heading_number = (
            int(numbered_match.group(1)) if numbered_match else None
        )
        font_size = float(item.get("_font_size", 0.0))
        font_colour = item.get("_font_colour")
        uses_level_5_style = (
            level_4_parent_font_size is not None
            # PyMuPDF can report the same visual font as 9.5 on one
            # platform and 10.0 on another. Allow a small tolerance while
            # still requiring the child's contrasting font colour.
            and font_size <= level_4_parent_font_size + 0.25
            and font_colour != level_4_parent_font_colour
        )

        if is_numbered and item.get("heading_level") == 3:
            continues_numbered_level_4 = (
                not parent_started_on_page
                and not parent_is_numbered
                and level_4_parent_is_numbered
                and level_4_parent_number is not None
                and heading_number == level_4_parent_number + 1
            )
            is_numbered_child = (
                (parent_started_on_page or continues_numbered_level_4)
                and parent_font_size is not None
                and not parent_is_numbered
                and font_size < parent_font_size - 0.5
            )
            if is_numbered_child:
                # A smaller numbered heading can sit below an unnumbered
                # level-3 heading. Normally the parent must be on the same
                # page; across a page boundary, require the numbering to
                # continue consecutively so unrelated sections do not inherit
                # stale hierarchy state.
                item["heading_level"] = 4
                level_4_parent_font_size = font_size
                level_4_parent_font_colour = font_colour
                level_4_parent_is_numbered = True
                level_4_parent_number = heading_number
            else:
                parent_font_size = font_size
                parent_font_colour = font_colour
                parent_is_numbered = True
                parent_started_on_page = False
                level_4_parent_font_size = None
                level_4_parent_font_colour = None
                level_4_parent_is_numbered = False
                level_4_parent_number = None
        elif (
            parent_font_size is not None
            and item.get("heading_level") == 3
        ):
            is_smaller_than_parent = font_size < parent_font_size - 0.5
            is_same_size_numbered_child = (
                parent_is_numbered
                and font_size <= parent_font_size + 0.5
            )
            is_consecutive_same_style_unnumbered_child = (
                previous_item_was_heading
                and parent_started_on_page
                and not parent_is_numbered
                and not is_numbered
                and level_4_parent_font_size is None
                and abs(font_size - parent_font_size) <= 0.25
                and font_colour == parent_font_colour
            )

            if (
                is_smaller_than_parent
                or is_same_size_numbered_child
                or is_consecutive_same_style_unnumbered_child
            ):
                if uses_level_5_style or level_4_parent_is_numbered:
                    # On some platforms this lower-emphasis heading is
                    # initially detected as level 3. Move it directly below
                    # the existing level-4 parent instead of demoting it only
                    # once to level 4. An unnumbered heading following a
                    # numbered level-4 parent is also its semantic child even
                    # when the PDF uses the same typography for both levels.
                    item["heading_level"] = 5
                else:
                    item["heading_level"] = 4
                    level_4_parent_font_size = font_size
                    level_4_parent_font_colour = font_colour
                    level_4_parent_is_numbered = False
                    level_4_parent_number = None
            else:
                parent_font_size = font_size
                parent_font_colour = font_colour
                parent_is_numbered = False
                parent_started_on_page = True
                level_4_parent_font_size = None
                level_4_parent_font_colour = None
                level_4_parent_is_numbered = False
                level_4_parent_number = None
        elif item.get("heading_level") == 3:
            parent_font_size = font_size
            parent_font_colour = font_colour
            parent_is_numbered = False
            parent_started_on_page = True
            level_4_parent_font_size = None
            level_4_parent_font_colour = None
            level_4_parent_is_numbered = False
            level_4_parent_number = None
        elif (
            item.get("heading_level") == 4
            and uses_level_5_style
        ):
            item["heading_level"] = 5

        previous_item_was_heading = True

    for item in page_content:
        item.pop("_font_size", None)
        item.pop("_font_colour", None)

    hierarchy_state["parent_font_size"] = parent_font_size
    hierarchy_state["parent_font_colour"] = parent_font_colour
    hierarchy_state["parent_is_numbered"] = parent_is_numbered
    hierarchy_state["level_4_parent_font_size"] = level_4_parent_font_size
    hierarchy_state["level_4_parent_font_colour"] = level_4_parent_font_colour
    hierarchy_state["level_4_parent_is_numbered"] = (
        level_4_parent_is_numbered
    )
    hierarchy_state["level_4_parent_number"] = level_4_parent_number


CONTINUATION_START = re.compile(
    r"^\s*\(?\s*cont(?:inued)?\b\s*\.{0,3}\s*\)?\s*",
    flags=re.IGNORECASE,
)
CONTINUATION_END = re.compile(
    r"\s*\(?\s*cont(?:inued)?\b\s*\.{1,3}\s*\)?\s*$",
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
                    if (
                        column == "what_is_covered"
                        and source_row.get("row_heading")
                    ):
                        row["row_heading"] = source_row["row_heading"]
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
    heading_hierarchy_state: dict[str, Any] = {}
    subgroup_start_pdf_pages = {
        printed_start + UNNUMBERED_FRONT_PAGES
        for group in CONTENTS_GROUPS
        for printed_start, _ in group["subgroups"]
    }

    for page_index, page in enumerate(document):
        pdf_page_number = page_index + 1
        if (
            pdf_page_number in EXCLUDED_PDF_PAGES
            or pdf_page_number in subgroup_start_pdf_pages
        ):
            heading_hierarchy_state.clear()
        booklet_page_number = printed_page_number(pdf_page_number)
        tables, table_rectangles = extract_tables(page)
        notices, notice_rectangles = extract_standalone_important_notices(
            page,
            table_rectangles,
        )
        excluded_rectangles = [*table_rectangles, *notice_rectangles]
        blocks = extract_text_blocks(page, excluded_rectangles)
        page_content = order_page_content(blocks, tables, notices)
        apply_nested_heading_levels(
            page_content,
            heading_hierarchy_state,
        )

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


MERGEABLE_CROSS_PAGE_TABLE_TYPES = {
    "coverage_by_level",
    "coverage_details",
    "definitions",
    "emergency_coverage_details",
}


def flatten_subsection_content(
    subsection_pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten page content while retaining item-level source provenance.

    Some logical tables are extracted as one table object per PDF page because
    layout coordinates are page-local. If one page ends with a mergeable table
    and the same or immediately following page continues with the same table
    type and columns, join their rows into a single logical table. Same-page
    handoffs can occur after a row continuation is consolidated. Row provenance
    remains page-specific, while the table records its full page range.
    """
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

            previous_item = (
                subsection_content[-1] if subsection_content else None
            )
            continues_cross_page_table = (
                item.get("content_type") == "table"
                and item.get("table_type")
                in MERGEABLE_CROSS_PAGE_TABLE_TYPES
                and isinstance(previous_item, dict)
                and previous_item.get("content_type") == "table"
                and previous_item.get("table_type")
                == item.get("table_type")
                and previous_item.get("columns") == item.get("columns")
                and previous_item.get("source_pdf_pages")
                and pdf_page_number
                - previous_item["source_pdf_pages"][-1]
                in {0, 1}
            )
            if continues_cross_page_table:
                previous_item.setdefault("rows", []).extend(
                    item.get("rows", [])
                )
                previous_item["source_pdf_pages"] = sorted(
                    set(previous_item["source_pdf_pages"])
                    | set(item["source_pdf_pages"])
                )
                previous_item["source_printed_pages"] = sorted(
                    set(previous_item["source_printed_pages"])
                    | set(item["source_printed_pages"])
                )
                continue

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


def apply_pure_numbered_subsection_levels(
    sections: list[dict[str, Any]],
) -> None:
    """Make a complete numbered subsection sequence level 4.

    A printed subsection title is already represented by
    ``subsection_headings`` and therefore acts as the level-3 parent. When all
    headings inside that subsection form one uninterrupted sequence beginning
    at 1, they are direct children of the subsection rather than independent
    level-3 boundaries. Mixed hierarchies are left unchanged.
    """
    for section in sections:
        for subsection in section.get("subsections", []):
            headings = [
                item
                for item in subsection.get("content", [])
                if item.get("content_type") == "heading"
            ]
            if len(headings) < 3:
                continue

            heading_numbers = []
            for heading in headings:
                match = re.match(
                    r"^(\d+)\.\s+",
                    normalise_heading(heading.get("text", "")),
                )
                if (
                    match is None
                    or heading.get("heading_level") != 3
                ):
                    break
                heading_numbers.append(int(match.group(1)))
            else:
                expected_numbers = list(range(1, len(headings) + 1))
                if heading_numbers == expected_numbers:
                    for heading in headings:
                        heading["heading_level"] = 4


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
        document_summary = extract_document_summary(document)
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
    apply_pure_numbered_subsection_levels(sections)

    if document_summary:
        sections.insert(0, create_document_summary_section(document_summary))

    validate_page_assignments(pages, excluded_pages, sections)

    return {
        "document_name": pdf_path.name,
        "document_type": "home_insurance_policy_booklet",
        "document_code": extract_document_code(pdf_path),
        "title": title,
        "sha256": calculate_sha256(pdf_path),
        "fca_reference_number": None,
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
