#!/usr/bin/env python3
"""Extract HH- home-insurance booklets into contents-grouped JSON.

Usage:
    python3 scripts/prepare_data/extract_hh_pdfs.py \
      --input-dir data/raw \
      --output-file data/processed/hh_documents.json

Dependency:
    python3 -m pip install pymupdf
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pymupdf

from insurance_rag.files import calculate_sha256
from insurance_rag.logging_setup import add_logging_arguments, configure_logging

logger = logging.getLogger(__name__)

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

# The booklet's contents structure is data rather than code, so it lives in a
# JSON file next to this script. See load_contents_groups() for the format.
CONTENTS_GROUPS_FILE = Path(__file__).resolve().with_name("hh_contents_groups.json")

ContentsGroup = dict[str, Any]


def load_contents_groups(
    path: Path = CONTENTS_GROUPS_FILE,
) -> tuple[ContentsGroup, ...]:
    """Load the booklet's contents groups from JSON.

    Each group has a ``group_heading`` and ``subgroups``. Each subgroup is
    returned as a ``(start_page, (heading, ...))`` tuple, where ``start_page``
    is the printed page number. Headings that start on the same page share one
    subgroup so their page content is not duplicated.

    Raises:
        ValueError: If the file is missing required fields or has invalid types.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    groups: list[ContentsGroup] = []
    for group in data["contents_groups"]:
        subgroups = tuple(
            (int(subgroup["start_page"]), tuple(subgroup["headings"]))
            for subgroup in group["subgroups"]
        )
        if not isinstance(group.get("group_heading"), str) or not subgroups:
            raise ValueError(f"Invalid contents group in {path}: {group!r}")
        groups.append({"group_heading": group["group_heading"], "subgroups": subgroups})
    return tuple(groups)


CONTENTS_GROUPS = load_contents_groups()


def normalise_heading(text: str) -> str:
    """Normalise a block for comparison with a defined contents heading."""
    without_page_numbers = "\n".join(
        line for line in text.splitlines() if not re.fullmatch(r"\d+", line.strip())
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
        and printed_page in DEFINED_HEADING_PRINTED_PAGES.get(normalised_text, set())
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
        description=("Extract PDFs beginning with HH- into contents-grouped JSON.")
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
    add_logging_arguments(parser)
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

    for raw_line in text.splitlines():
        line = raw_line.replace("\x07", "")
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

        if not (10 <= rectangle.width <= 25 and 10 <= rectangle.height <= 25):
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
        if (x0 <= icon["centre_x"] <= x1 and y0 <= icon["centre_y"] <= y1)
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
                span for span in line.get("spans", []) if span.get("text", "").strip()
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
                span.get("color") in HEADING_COLOURS for span in spans
            )

            if not (is_emphasised and uses_heading_colour):
                return re.sub(r"\s+", " ", " ".join(heading_lines)).strip() or None

            line_text = clean_text("".join(span.get("text", "") for span in spans))
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
    conditions_text, separator, notes_text = details.partition("\nGood to know\n")

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
        if not is_standalone_notice_box(drawing, table_rectangles):
            continue

        notice = build_important_notice(rectangle, text_inside(rectangle, text_blocks))
        if notice is None:
            continue
        notices.append(notice)
        notice_rectangles.append(rectangle)

    return notices, notice_rectangles


def is_standalone_notice_box(
    drawing: dict[str, Any], table_rectangles: list[pymupdf.Rect]
) -> bool:
    """Return whether a drawing is a large pink notice box outside any table."""
    rectangle = drawing.get("rect")
    if (
        rectangle is None
        or not is_important_border_colour(drawing.get("color"))
        or rectangle.width < 400
        or rectangle.height < 40
    ):
        return False
    return not any(
        rectangle.intersects(table_rectangle) for table_rectangle in table_rectangles
    )


def text_inside(rectangle: pymupdf.Rect, text_blocks: list[tuple[Any, ...]]) -> str:
    """Return the cleaned text of the blocks centred inside ``rectangle``.

    Each block's text is separated by a blank line.
    """
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
    return "\n\n".join(contained_text)


def build_important_notice(
    rectangle: pymupdf.Rect, notice_text: str
) -> dict[str, Any] | None:
    """Return a notice item, or None unless the text starts with "IMPORTANT".

    A short second line without closing punctuation becomes the title.
    """
    if not notice_text:
        return None

    notice_lines = [line.strip() for line in notice_text.splitlines() if line.strip()]
    if notice_lines[0].strip().casefold() != "important":
        return None

    remaining_lines = notice_lines[1:]
    title = None
    if (
        remaining_lines
        and len(remaining_lines[0]) <= 80
        and not remaining_lines[0].endswith((".", ":"))
    ):
        title = remaining_lines.pop(0)

    return {
        "content_type": "important_notice",
        "_x0": rectangle.x0,
        "_y0": rectangle.y0,
        "label": "IMPORTANT",
        "title": title,
        "text": "\n".join(remaining_lines) or None,
    }


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
            first_cell = next(cell for cell in first_row.cells if cell is not None)
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
        notice_cell = next(cell for cell in notice_row.cells if cell is not None)
        content_items.append(
            {
                "content_type": "important_notice",
                "_x0": notice_cell[0],
                "_y0": notice_cell[1],
                "label": notice_lines[0] if notice_lines else "IMPORTANT",
                "title": notice_lines[1] if len(notice_lines) > 1 else None,
                "text": (
                    "\n".join(notice_lines[2:]) if len(notice_lines) > 2 else None
                ),
            }
        )
        segment_start = notice_index + 1

    if segment_start < len(data_rows):
        first_row = table.rows[segment_start + 1]
        first_cell = next(cell for cell in first_row.cells if cell is not None)
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
    row_top = min(cell[1] for cell in cells if cell is not None)
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
        extracted_rows = read_table_cells(page, table)
        if not extracted_rows:
            continue

        table_rectangle = pymupdf.Rect(table.bbox)
        table_rectangles.append(table_rectangle)
        structured_tables.extend(
            structure_table(page, table, extracted_rows, table_rectangle, icons)
        )

    return structured_tables, table_rectangles


def read_table_cells(page: pymupdf.Page, table: Any) -> list[list[str | None]]:
    """Read the text of every cell in a detected table, row by row.

    ``Table.extract()`` orders the characters in a cell using their
    horizontal coordinates.  In this booklet some font glyphs overlap
    slightly (notably ``fi``), so coordinate ordering can transpose
    otherwise valid words: ``find`` becomes ``fnid``, ``office`` becomes
    ``ofcfie``, and so on.  The normal page text engine preserves the
    PDF's logical character order.  Keep the table detector's cell
    geometry, but read each cell through that text engine instead.
    """
    return [
        [
            page.get_textbox(pymupdf.Rect(cell)).strip() if cell is not None else None
            for cell in row.cells
        ]
        for row in table.rows
    ]


def structure_table(
    page: pymupdf.Page,
    table: Any,
    extracted_rows: list[list[str | None]],
    table_rectangle: pymupdf.Rect,
    icons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recognise the kind of table from its header and return its content items.

    Most tables become one item; coverage tables interrupted by an IMPORTANT
    notice become several items.
    """
    header = [clean_cell(value) for value in extracted_rows[0]]
    normalised_header = [normalise_heading(value or "") for value in header]
    column_count = len(header)
    position = {
        "content_type": "table",
        "_x0": table_rectangle.x0,
        "_y0": table_rectangle.y0,
    }

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
        return [{**position, **change_notification_table(extracted_rows)}]
    if is_policy_change_permissions_table(extracted_rows, column_count):
        return [
            {
                **position,
                **policy_change_permissions_table(table, extracted_rows, icons),
            }
        ]
    if column_count == 5 and has_coverage_text_columns:
        status_columns = {
            level: index for index, level in enumerate(COVER_LEVELS, start=2)
        }
        table_definition = {
            "table_type": "coverage_by_level",
            "columns": ["what_is_covered", "what_is_not_covered", *COVER_LEVELS],
            "rows": build_coverage_rows(
                page, table, extracted_rows, icons, status_columns
            ),
        }
        return split_table_at_important_notices(
            table_definition=table_definition,
            extracted_rows=extracted_rows,
            table=table,
            table_rectangle=table_rectangle,
        )
    if column_count == 2 and has_coverage_text_columns:
        return [
            {
                **position,
                "table_type": "coverage_details",
                "columns": ["what_is_covered", "what_is_not_covered"],
                "rows": build_coverage_rows(page, table, extracted_rows, icons, None),
            }
        ]
    if (
        column_count == 4
        and has_coverage_text_columns
        and normalised_header[2:] == ["hec", "hex"]
    ):
        table_definition = {
            "table_type": "emergency_coverage_details",
            "columns": ["what_is_covered", "what_is_not_covered", "hec", "hex"],
            "rows": build_coverage_rows(
                page, table, extracted_rows, icons, {"hec": 2, "hex": 3}
            ),
        }
        return split_table_at_important_notices(
            table_definition=table_definition,
            extracted_rows=extracted_rows,
            table=table,
            table_rectangle=table_rectangle,
        )
    if column_count == 3 and normalised_header[1:] == ["hec", "hex"]:
        return [
            {**position, **emergency_cover_by_level_table(table, extracted_rows, icons)}
        ]
    return [
        {
            **position,
            "table_type": "generic",
            "columns": header,
            "rows": [
                [clean_cell(value) for value in row] for row in extracted_rows[1:]
            ],
        }
    ]


def is_policy_change_permissions_table(
    extracted_rows: list[list[str | None]], column_count: int
) -> bool:
    """Return whether the table lists who can make each type of policy change."""
    return (
        column_count == 8
        and len(extracted_rows) >= 4
        and normalise_heading(extracted_rows[0][1] or "") == "type of change"
        and normalise_heading(extracted_rows[2][0] or "") == "who can make a change"
    )


def change_notification_table(
    extracted_rows: list[list[str | None]],
) -> dict[str, Any]:
    """Return the change-notification requirements table."""
    return {
        "table_type": "change_notification_requirements",
        "columns": ["notification_timing", "requirements"],
        "rows": [
            parse_change_notification_row(
                clean_cell(extracted_row[0]),
                clean_cell(extracted_row[1]),
            )
            for extracted_row in extracted_rows
        ],
    }


def policy_change_permissions_table(
    table: Any,
    extracted_rows: list[list[str | None]],
    icons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the table of who can make each type of policy change.

    A tick icon in a cell means that role can make that change.
    """
    permission_names = [
        re.sub(r"\s+", "_", normalise_heading(value or ""))
        for value in extracted_rows[2][1:]
    ]
    rows = []
    for row_index, extracted_row in enumerate(extracted_rows[3:], start=3):
        cells = table.rows[row_index].cells
        rows.append(
            {
                "role": re.sub(r"\s+", " ", clean_cell(extracted_row[0]) or "").strip(),
                "permissions": {
                    permission_name: (
                        status_in_cell(cells[column_index], icons) == "included"
                    )
                    for column_index, permission_name in enumerate(
                        permission_names, start=1
                    )
                },
            }
        )

    return {
        "table_type": "policy_change_permissions",
        "permission_groups": [
            {
                "heading": ("Single policies or MultiCover policies"),
                "permissions": permission_names[:4],
            },
            {
                "heading": "MultiCover policies only",
                "permissions": permission_names[4:],
            },
        ],
        "rows": rows,
    }


def emergency_cover_by_level_table(
    table: Any,
    extracted_rows: list[list[str | None]],
    icons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the table of Home Emergency cover included at each policy level."""
    rows = []
    for row_index, extracted_row in enumerate(extracted_rows[1:], start=1):
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
    return {
        "table_type": "emergency_cover_by_level",
        "columns": ["cover_level", "hec", "hex"],
        "rows": rows,
    }


def build_coverage_rows(
    page: pymupdf.Page,
    table: Any,
    extracted_rows: list[list[str | None]],
    icons: list[dict[str, Any]],
    status_columns: dict[str, int] | None,
) -> list[dict[str, Any]]:
    """Build the rows of a "What is covered / What isn't covered" table.

    Args:
        page: The page, used to read styled row headings.
        table: The detected table, used for cell geometry.
        extracted_rows: The cell text, including the header row.
        icons: Status icons on the page.
        status_columns: Maps each cover status name to the column holding its
            icon, or None if the table has no status columns.
    """
    rows = []
    for row_index, extracted_row in enumerate(extracted_rows[1:], start=1):
        cells = table.rows[row_index].cells
        row_heading = extract_table_row_heading(page, cells[0])
        row: dict[str, Any] = {
            "what_is_covered": clean_cell(extracted_row[0]),
            "what_is_not_covered": clean_cell(extracted_row[1]),
        }
        if status_columns is not None:
            row["cover_status"] = {
                name: status_in_cell(cells[column_index], icons)
                for name, column_index in status_columns.items()
            }
        if row_heading:
            row = {"row_heading": row_heading, **row}
        inherited = vertically_merged_columns(
            table,
            row_index,
            {0: "what_is_covered", 1: "what_is_not_covered"},
        )
        inherited = [column for column in inherited if not row.get(column)]
        if inherited:
            row["_inherit_columns"] = inherited
        rows.append(row)
    return rows


def extract_text_blocks(
    page: pymupdf.Page,
    excluded_rectangles: list[pymupdf.Rect],
) -> list[dict[str, Any]]:
    """Extract semantic headings and paragraphs in visual reading order."""
    text_blocks = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        text_blocks.extend(block_content_items(page, block, excluded_rectangles))
    return text_blocks


def block_content_items(
    page: pymupdf.Page,
    block: dict[str, Any],
    excluded_rectangles: list[pymupdf.Rect],
) -> list[dict[str, Any]]:
    """Classify one PDF text block, returning no items if it should be skipped.

    Blocks are skipped when they are images, lie inside a table or notice,
    are empty, or only repeat structural text such as running page titles.
    """
    # PyMuPDF block type 0 represents text; type 1 represents an image.
    if block.get("type") != 0:
        return []

    x0, y0, x1, y1 = block["bbox"]
    bbox = (x0, y0, x1, y1)
    block_rectangle = pymupdf.Rect(block["bbox"])
    if any(
        block_rectangle.intersects(table_rectangle)
        for table_rectangle in excluded_rectangles
    ):
        return []

    lines = block.get("lines", [])
    cleaned_text = clean_text(lines_text(lines))
    if not cleaned_text:
        return []
    if normalise_heading(cleaned_text) in REDUNDANT_PAGE_TEXT:
        return []
    if is_structural_heading(cleaned_text, printed_page_number(page.number + 1)):
        return []

    # A definition heading and its first enumerated item can share one PDF
    # text block even though they occupy separate visual columns.  Split a
    # semibold left-column first line from a right-column list item before
    # applying whole-block heading classification.
    if len(lines) >= 2:
        split_items = split_heading_from_definition(lines)
        if split_items is not None:
            return split_items

    return [classify_text_block(cleaned_text, text_spans(lines), bbox)]


def line_text(line: dict[str, Any]) -> str:
    """Return the raw text of one PDF line."""
    return "".join(span.get("text", "") for span in line.get("spans", []))


def lines_text(lines: list[dict[str, Any]]) -> str:
    """Return the text of several PDF lines, one per line."""
    return "\n".join(line_text(line).rstrip() for line in lines)


def text_spans(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the spans in the lines that contain non-whitespace text."""
    return [
        span
        for line in lines
        for span in line.get("spans", [])
        if span.get("text", "").strip()
    ]


def all_spans_emphasised(spans: list[dict[str, Any]]) -> bool:
    """Return whether there are spans and all use a bold or semibold font."""
    return bool(spans) and all(
        any(
            weight in span.get("font", "").casefold() for weight in ("bold", "semibold")
        )
        for span in spans
    )


def all_spans_use_heading_colours(spans: list[dict[str, Any]]) -> bool:
    """Return whether there are spans and all use a heading colour."""
    return bool(spans) and all(span.get("color") in HEADING_COLOURS for span in spans)


def largest_font_size(spans: list[dict[str, Any]]) -> float:
    """Return the largest font size among the spans, or 0.0 if there are none."""
    return max((float(span.get("size", 0.0)) for span in spans), default=0.0)


def ends_like_sentence(text: str) -> bool:
    """Return whether text ends with sentence or clause punctuation."""
    return text.rstrip().endswith((".", "!", "?", ":", ";"))


def list_item(match: re.Match[str], bbox: tuple[float, ...]) -> dict[str, Any]:
    """Return a list item from a ``LIST_ITEM_START`` match and its bounding box."""
    x0, y0, x1, y1 = bbox
    return {
        "content_type": "list_item",
        "list_marker": match.group("marker"),
        "text": match.group("text").strip(),
        "_x0": x0,
        "_y0": y0,
        "_x1": x1,
        "_y1": y1,
    }


def split_heading_from_definition(
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Split a block whose first line is a heading and the rest its definition.

    Returns:
        The heading and its definition as two items, or None if the block is
        not one of the two recognised shapes: a narrow emphasised term with
        its definition to the right, or a coloured heading stacked above
        smaller text.
    """
    first_line = lines[0]
    first_line_spans = text_spans([first_line])
    first_line_text = clean_text(line_text(first_line))
    remaining_text = clean_text(lines_text(lines[1:]))
    first_line_is_emphasised = all_spans_emphasised(first_line_spans)
    first_line_size = largest_font_size(first_line_spans)
    remaining_maximum_font_size = largest_font_size(text_spans(lines[1:]))
    first_line_x0, first_line_y0, first_line_x1, first_line_y1 = first_line["bbox"]
    remaining_bbox = (
        min(line["bbox"][0] for line in lines[1:]),
        min(line["bbox"][1] for line in lines[1:]),
        max(line["bbox"][2] for line in lines[1:]),
        max(line["bbox"][3] for line in lines[1:]),
    )

    is_mixed_definition_block = (
        first_line_is_emphasised
        and first_line_text
        and first_line_x1 <= 170.0
        and remaining_bbox[0] > first_line_x1
    )
    is_stacked_heading_block = (
        first_line_is_emphasised
        and all_spans_use_heading_colours(first_line_spans)
        and first_line_size >= 10.0
        and remaining_maximum_font_size < first_line_size - 0.25
        and len(first_line_text) <= 120
        and not ends_like_sentence(first_line_text)
    )
    if not (is_mixed_definition_block or is_stacked_heading_block):
        return None

    remaining_list_match = LIST_ITEM_START.match(remaining_text)
    if remaining_list_match is not None:
        definition_item = list_item(remaining_list_match, remaining_bbox)
    else:
        x0, y0, x1, y1 = remaining_bbox
        definition_item = {
            "content_type": "paragraph",
            "text": remaining_text,
            "_x0": x0,
            "_y0": y0,
            "_x1": x1,
            "_y1": y1,
        }
    heading = {
        "content_type": "heading",
        "heading_level": 3 if first_line_size >= 11.0 else 4,
        "text": first_line_text,
        "_font_size": first_line_size,
        "_font_colour": first_line_spans[0].get("color"),
        "_x0": first_line_x0,
        "_y0": first_line_y0,
        "_x1": first_line_x1,
        "_y1": first_line_y1,
    }
    return [heading, definition_item]


def classify_text_block(
    cleaned_text: str,
    spans: list[dict[str, Any]],
    bbox: tuple[float, ...],
) -> dict[str, Any]:
    """Return a whole text block as a heading, list item or paragraph.

    Headings are recognised from font size, weight and colour; the heading
    level is 2 for very large text, 3 for large or coloured headings, and 4
    otherwise.
    """
    x0, y0, x1, y1 = bbox
    maximum_font_size = largest_font_size(spans)
    all_emphasised = all_spans_emphasised(spans)
    all_semibold = bool(spans) and all(
        "semibold" in span.get("font", "").casefold() for span in spans
    )
    uses_heading_colour = any(span.get("color") in HEADING_COLOURS for span in spans)
    is_compact_coloured_label = (
        maximum_font_size >= 9.4
        and all_spans_use_heading_colours(spans)
        and len(cleaned_text) <= 120
        and not ends_like_sentence(cleaned_text)
    )
    is_compact_semibold_label = (
        maximum_font_size >= 9.4
        and all_semibold
        and len(cleaned_text) <= 100
        and not ends_like_sentence(cleaned_text)
    )
    is_heading = (
        maximum_font_size >= 14.0
        or (maximum_font_size >= 10.0 and all_emphasised)
        or (maximum_font_size >= 9.4 and all_emphasised and uses_heading_colour)
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
        return {
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

    list_item_match = LIST_ITEM_START.match(cleaned_text)
    if list_item_match:
        return list_item(list_item_match, bbox)

    return {
        "content_type": "paragraph",
        "_x0": x0,
        "_y0": y0,
        "_x1": x1,
        "_y1": y1,
        "text": cleaned_text,
    }


LAYOUT_KEYS = frozenset({"_x0", "_y0", "_x1", "_y1"})


def order_page_content(
    text_blocks: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    notices: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Interleave text and tables in their visual page order.

    The page is processed in stages: wrapped headings are merged, items are
    ordered by visual rows, split definition headings are rejoined, paragraph
    fragments are merged, borderless glossaries become tables, and finally the
    layout coordinates are replaced by an ``order`` number.
    """
    retained_text_blocks = merge_wrapped_headings(text_blocks)
    ordered_items = order_by_visual_rows([*retained_text_blocks, *tables, *notices])
    ordered_items = rejoin_split_definition_headings(ordered_items)
    merged_items = merge_paragraph_fragments(ordered_items)
    merged_items = convert_glossary_runs(merged_items)
    return strip_layout_fields(merged_items)


def join_heading_lines(target: dict[str, Any], other: dict[str, Any]) -> None:
    """Append ``other``'s text to ``target`` and widen ``target``'s bounding box."""
    target["text"] = "\n".join((target["text"].rstrip(), other["text"].lstrip()))
    target["_x0"] = min(target.get("_x0", 0.0), other.get("_x0", 0.0))
    target["_x1"] = max(target.get("_x1", 0.0), other.get("_x1", 0.0))
    target["_y1"] = max(target.get("_y1", 0.0), other.get("_y1", 0.0))


def is_wrapped_heading_line(
    heading: dict[str, Any], continuation: dict[str, Any], vertical_gap: float
) -> bool:
    """Return whether ``continuation`` is the next line of a narrow wrapped heading."""
    return (
        heading.get("heading_level") == continuation.get("heading_level")
        and ("\n" not in heading.get("text", "") or vertical_gap <= 4.0)
        and abs(heading.get("_x0", 0.0) - continuation.get("_x0", 0.0)) <= 35.0
        and max(heading.get("_x1", 0.0), continuation.get("_x1", 0.0)) <= 170.0
    )


def merge_wrapped_headings(text_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge vertically adjacent fragments of the same heading.

    On some PyMuPDF/platform combinations, each visual line of a narrow
    left-column definition heading is returned as a separate block. Merge
    vertically adjacent heading fragments before paragraphs are interleaved;
    otherwise a right-column definition can be sorted between the two lines
    of headings such as "Reasonable prospects".

    Headings are updated in place; the returned list omits the merged-in lines.
    """
    consumed_heading_ids = set()
    headings = sorted(
        (item for item in text_blocks if item.get("content_type") == "heading"),
        key=lambda item: (item.get("_y0", 0.0), item.get("_x0", 0.0)),
    )

    for heading_index, heading in enumerate(headings):
        if id(heading) in consumed_heading_ids:
            continue

        for continuation in headings[heading_index + 1 :]:
            vertical_gap = continuation.get("_y0", 0.0) - heading.get("_y1", 0.0)
            # Lines belonging to one wrapped heading sit almost directly
            # beneath each other.  A larger allowance can accidentally join
            # two consecutive one-line definition terms (for example,
            # "Vehicle cloning" and "We, us, our, ARAG").
            if vertical_gap > 4.0:
                break
            if vertical_gap < -1.0:
                continue
            if is_wrapped_heading_line(heading, continuation, vertical_gap):
                join_heading_lines(heading, continuation)
                consumed_heading_ids.add(id(continuation))

    return [item for item in text_blocks if id(item) not in consumed_heading_ids]


def order_by_visual_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order items top to bottom, then left to right within each visual row.

    Items whose tops are within 3 points of the first item in a row are treated
    as one visual row, so side-by-side columns are read left to right.
    """
    vertically_sorted_items = sorted(items, key=lambda item: (item["_y0"], item["_x0"]))
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
    return ordered_items


def is_split_definition_heading(
    first_heading: dict[str, Any],
    definition: dict[str, Any],
    second_heading: dict[str, Any],
) -> bool:
    """Return whether a definition sits between two lines of its own heading."""
    return (
        first_heading.get("content_type") == "heading"
        and definition.get("content_type") == "paragraph"
        and second_heading.get("content_type") == "heading"
        and first_heading.get("heading_level") == second_heading.get("heading_level")
        and max(first_heading.get("_x1", 0.0), second_heading.get("_x1", 0.0)) <= 170.0
        and definition.get("_x0", 0.0)
        > max(first_heading.get("_x1", 0.0), second_heading.get("_x1", 0.0))
        and second_heading.get("_y0", 0.0) <= definition.get("_y1", 0.0)
    )


def rejoin_split_definition_headings(
    ordered_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rejoin heading lines that ended up either side of their definition.

    Some platforms place a right-column definition between two visual lines
    of its narrow left-column heading.  Recognise the heading/definition/
    heading pattern from column positions and overlapping vertical ranges,
    then join the heading lines without moving the definition text itself.
    """
    reconstructed_items = []
    item_index = 0
    while item_index < len(ordered_items):
        if item_index + 2 < len(ordered_items):
            first_heading, definition, second_heading = ordered_items[
                item_index : item_index + 3
            ]
            if is_split_definition_heading(first_heading, definition, second_heading):
                join_heading_lines(first_heading, second_heading)
                reconstructed_items.extend((first_heading, definition))
                item_index += 3
                continue

        reconstructed_items.append(ordered_items[item_index])
        item_index += 1
    return reconstructed_items


def starts_list_item(text: str) -> bool:
    """Return whether text starts with a list marker such as ``a)`` or ``(1)``."""
    return (
        re.match(r"^(?:\(?[a-z0-9]+[.)]\s)", text.lstrip(), flags=re.IGNORECASE)
        is not None
    )


def starts_lowercase(text: str) -> bool:
    """Return whether the first letter in ``text`` is lowercase."""
    first_letter = re.search(r"[A-Za-z]", text)
    return first_letter is not None and first_letter.group(0).islower()


def can_join_paragraph_fragment(
    previous: dict[str, Any] | None,
    item: dict[str, Any],
    next_item: dict[str, Any] | None,
) -> bool:
    """Return whether ``item`` continues the text of the ``previous`` item.

    PDF extraction can split one paragraph or list item into several blocks.
    A fragment is joined when the previous item looks unfinished and the
    fragment sits in the same column, starts in lowercase, briefly finishes
    the sentence, or continues a list item.
    """
    if previous is None:
        return False

    previous_text = previous.get("text", "")
    current_text = item.get("text", "")
    current_starts_list_item = starts_list_item(current_text)
    current_starts_bullet = current_text.lstrip().startswith(("•", "-"))
    previous_is_list_item = previous.get("content_type") == "list_item"
    item_is_paragraph = item.get("content_type") == "paragraph"
    horizontal_offset = abs(previous.get("_x0", 0.0) - item.get("_x0", 0.0))
    vertical_gap = item.get("_y0", 0.0) - previous.get("_y1", 0.0)

    is_same_column_continuation = (
        horizontal_offset <= 2.0 and 0.0 <= vertical_gap <= 4.0
    )
    is_short_sentence_ending_fragment = (
        len(previous_text.strip()) >= 60
        and 0 < len(current_text.strip()) <= 80
        and current_text.rstrip().endswith((".", "!", "?"))
        and not current_starts_bullet
        and not current_starts_list_item
    )
    is_lowercase_continuation = (
        starts_lowercase(current_text)
        and not current_starts_bullet
        and not current_starts_list_item
    )
    is_relaxed_list_column_continuation = (
        previous_is_list_item
        and item_is_paragraph
        and horizontal_offset <= 30.0
        and -2.0 <= vertical_gap <= 6.0
    )
    is_list_item_continuation = (
        previous_is_list_item
        and item_is_paragraph
        and not current_starts_list_item
        and starts_lowercase(current_text)
    )
    is_paragraph_between_list_items = (
        previous_is_list_item
        and item_is_paragraph
        and next_item is not None
        and next_item.get("content_type") == "list_item"
        and not current_text.rstrip().endswith(":")
    )
    is_pipe_delimited_list_continuation = (
        previous_is_list_item
        and item.get("content_type") in {"paragraph", "heading"}
        and previous_text.rstrip().endswith("|")
    )
    previous_is_unfinished = (
        not previous_text.rstrip().endswith((".", "!", "?", ":", ";"))
        or is_paragraph_between_list_items
        or is_relaxed_list_column_continuation
        or is_pipe_delimited_list_continuation
    )
    return (
        previous.get("content_type") in {"paragraph", "list_item"}
        and (item_is_paragraph or is_pipe_delimited_list_continuation)
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


def merge_paragraph_fragments(
    ordered_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join paragraph and list-item fragments back into the item they continue."""
    merged_items: list[dict[str, Any]] = []

    for item_index, item in enumerate(ordered_items):
        previous = merged_items[-1] if merged_items else None
        next_item = (
            ordered_items[item_index + 1]
            if item_index + 1 < len(ordered_items)
            else None
        )

        if previous is not None and can_join_paragraph_fragment(
            previous, item, next_item
        ):
            previous["text"] = "\n".join(
                (previous["text"].rstrip(), item["text"].lstrip())
            )
            previous["_x1"] = max(previous.get("_x1", 0.0), item.get("_x1", 0.0))
            previous["_y1"] = item.get("_y1", previous.get("_y1"))
            continue

        merged_items.append(item)
    return merged_items


def read_glossary_row(
    items: list[dict[str, Any]], term_index: int
) -> tuple[dict[str, str] | None, int]:
    """Read one glossary term and its definition starting at ``term_index``.

    Returns:
        The ``{"term", "definition"}`` row and the index after its definition,
        or ``(None, term_index)`` if no term/definition pair starts there.
    """
    term = items[term_index]
    is_term = (
        term.get("content_type") == "heading"
        and term.get("heading_level") == 4
        and term.get("_x1", float("inf")) <= 175.0
    )
    if not is_term:
        return None, term_index

    definition_index = term_index + 1
    definition_items = []
    while definition_index < len(items):
        definition_item = items[definition_index]
        if definition_item.get("content_type") == "heading":
            break
        if (
            definition_item.get("content_type") not in {"paragraph", "list_item"}
            or definition_item.get("_x0", 0.0) < 155.0
        ):
            definition_items = []
            break
        definition_items.append(definition_item)
        definition_index += 1

    if not definition_items:
        return None, term_index

    definition_parts = []
    for definition_item in definition_items:
        definition_text = definition_item.get("text", "").strip()
        if definition_item.get("content_type") == "list_item":
            marker = definition_item.get("list_marker") or "•"
            definition_text = f"{marker} {definition_text}".strip()
        if definition_text:
            definition_parts.append(definition_text)

    row = {
        "term": re.sub(r"\s+", " ", term.get("text", "")).strip(),
        "definition": "\n".join(definition_parts),
    }
    return row, definition_index


def convert_glossary_runs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace runs of two or more glossary term/definition pairs with a table.

    Borderless glossaries use bold terms in a narrow left column and their
    definitions in a consistent right column.  PyMuPDF does not identify
    these as tables because there are no drawn grid lines.  Convert runs of
    two or more term/definition pairs into a semantic two-column table so a
    term does not behave like an active document heading.
    """
    glossary_items = []
    item_index = 0

    while item_index < len(items):
        rows = []
        run_index = item_index
        while run_index < len(items):
            row, next_index = read_glossary_row(items, run_index)
            if row is None:
                break
            rows.append(row)
            run_index = next_index

        if len(rows) >= 2:
            run_items = items[item_index:run_index]
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

        glossary_items.append(items[item_index])
        item_index += 1
    return glossary_items


def strip_layout_fields(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copies of the items without layout coordinates, numbered in order."""
    content = []
    for order, item in enumerate(items, start=1):
        output_item = {
            key: value for key, value in item.items() if key not in LAYOUT_KEYS
        }
        output_item["order"] = order
        content.append(output_item)
    return content


@dataclass
class HeadingHierarchy:
    """The active level-3 and level-4 parent headings while reading pages.

    The first seven fields are carried between pages through the
    ``hierarchy_state`` dict. ``parent_started_on_page`` and
    ``previous_item_was_heading`` only apply within one page.
    """

    parent_font_size: float | None = None
    parent_font_colour: Any = None
    parent_is_numbered: bool = False
    level_4_parent_font_size: float | None = None
    level_4_parent_font_colour: Any = None
    level_4_parent_is_numbered: bool = False
    level_4_parent_number: int | None = None
    parent_started_on_page: bool = False
    previous_item_was_heading: bool = False

    STATE_KEYS: ClassVar[tuple[str, ...]] = (
        "parent_font_size",
        "parent_font_colour",
        "parent_is_numbered",
        "level_4_parent_font_size",
        "level_4_parent_font_colour",
        "level_4_parent_is_numbered",
        "level_4_parent_number",
    )

    @classmethod
    def from_state(cls, hierarchy_state: dict[str, Any]) -> HeadingHierarchy:
        """Restore the hierarchy carried over from the previous page."""
        defaults = cls()
        return cls(
            **{
                key: hierarchy_state.get(key, getattr(defaults, key))
                for key in cls.STATE_KEYS
            }
        )

    def save_to(self, hierarchy_state: dict[str, Any]) -> None:
        """Store the hierarchy so the next page can continue it."""
        for key in self.STATE_KEYS:
            hierarchy_state[key] = getattr(self, key)

    def set_parent(
        self, font_size: float, font_colour: Any, numbered: bool, started_on_page: bool
    ) -> None:
        """Start a new level-3 parent and clear any level-4 parent."""
        self.parent_font_size = font_size
        self.parent_font_colour = font_colour
        self.parent_is_numbered = numbered
        self.parent_started_on_page = started_on_page
        self.level_4_parent_font_size = None
        self.level_4_parent_font_colour = None
        self.level_4_parent_is_numbered = False
        self.level_4_parent_number = None

    def set_level_4_parent(
        self, font_size: float, font_colour: Any, number: int | None
    ) -> None:
        """Record a heading as the active level-4 parent."""
        self.level_4_parent_font_size = font_size
        self.level_4_parent_font_colour = font_colour
        self.level_4_parent_is_numbered = number is not None
        self.level_4_parent_number = number

    def uses_level_5_style(self, font_size: float, font_colour: Any) -> bool:
        """Return whether a heading is styled as a child of the level-4 parent."""
        return (
            self.level_4_parent_font_size is not None
            # PyMuPDF can report the same visual font as 9.5 on one
            # platform and 10.0 on another. Allow a small tolerance while
            # still requiring the child's contrasting font colour.
            and font_size <= self.level_4_parent_font_size + 0.25
            and font_colour != self.level_4_parent_font_colour
        )

    def apply(self, item: dict[str, Any]) -> None:
        """Adjust one heading's level and update the hierarchy."""
        normalised_text = normalise_heading(item.get("text", ""))
        numbered_match = re.match(r"^(\d+)\.\s+", normalised_text)
        heading_number = int(numbered_match.group(1)) if numbered_match else None
        font_size = float(item.get("_font_size", 0.0))
        font_colour = item.get("_font_colour")
        level = item.get("heading_level")

        if numbered_match is not None and level == 3:
            self.apply_numbered_level_3(item, font_size, font_colour, heading_number)
        elif self.parent_font_size is not None and level == 3:
            self.apply_level_3_below_parent(
                item,
                font_size,
                font_colour,
                parent_font_size=self.parent_font_size,
                is_numbered=numbered_match is not None,
            )
        elif level == 3:
            self.set_parent(
                font_size, font_colour, numbered=False, started_on_page=True
            )
        elif level == 4 and self.uses_level_5_style(font_size, font_colour):
            item["heading_level"] = 5

    def apply_numbered_level_3(
        self,
        item: dict[str, Any],
        font_size: float,
        font_colour: Any,
        heading_number: int | None,
    ) -> None:
        """Handle a numbered heading detected as level 3.

        A smaller numbered heading can sit below an unnumbered level-3
        heading. Normally the parent must be on the same page; across a page
        boundary, require the numbering to continue consecutively so unrelated
        sections do not inherit stale hierarchy state.
        """
        continues_numbered_level_4 = (
            not self.parent_started_on_page
            and not self.parent_is_numbered
            and self.level_4_parent_is_numbered
            and self.level_4_parent_number is not None
            and heading_number == self.level_4_parent_number + 1
        )
        is_numbered_child = (
            (self.parent_started_on_page or continues_numbered_level_4)
            and self.parent_font_size is not None
            and not self.parent_is_numbered
            and font_size < self.parent_font_size - 0.5
        )
        if is_numbered_child:
            item["heading_level"] = 4
            self.set_level_4_parent(font_size, font_colour, number=heading_number)
        else:
            self.set_parent(
                font_size, font_colour, numbered=True, started_on_page=False
            )

    def apply_level_3_below_parent(
        self,
        item: dict[str, Any],
        font_size: float,
        font_colour: Any,
        parent_font_size: float,
        is_numbered: bool,
    ) -> None:
        """Handle a level-3 heading while a level-3 parent is active."""
        is_smaller_than_parent = font_size < parent_font_size - 0.5
        is_same_size_numbered_child = (
            self.parent_is_numbered and font_size <= parent_font_size + 0.5
        )
        is_consecutive_same_style_unnumbered_child = (
            self.previous_item_was_heading
            and self.parent_started_on_page
            and not self.parent_is_numbered
            and not is_numbered
            and self.level_4_parent_font_size is None
            and abs(font_size - parent_font_size) <= 0.25
            and font_colour == self.parent_font_colour
        )
        is_child = (
            is_smaller_than_parent
            or is_same_size_numbered_child
            or is_consecutive_same_style_unnumbered_child
        )
        if not is_child:
            self.set_parent(
                font_size, font_colour, numbered=False, started_on_page=True
            )
        elif (
            self.uses_level_5_style(font_size, font_colour)
            or self.level_4_parent_is_numbered
        ):
            # On some platforms this lower-emphasis heading is initially
            # detected as level 3. Move it directly below the existing level-4
            # parent instead of demoting it only once to level 4. An
            # unnumbered heading following a numbered level-4 parent is also
            # its semantic child even when the PDF uses the same typography
            # for both levels.
            item["heading_level"] = 5
        else:
            item["heading_level"] = 4
            self.set_level_4_parent(font_size, font_colour, number=None)


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
    hierarchy = HeadingHierarchy.from_state(hierarchy_state)

    for item in page_content:
        if item.get("content_type") != "heading":
            hierarchy.previous_item_was_heading = False
            continue
        hierarchy.apply(item)
        hierarchy.previous_item_was_heading = True

    for item in page_content:
        item.pop("_font_size", None)
        item.pop("_font_colour", None)

    hierarchy.save_to(hierarchy_state)


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
        part for part in (previous_without_marker, continuation_without_marker) if part
    )


def row_has_meaningful_content(row: dict[str, Any]) -> bool:
    """Return whether a coverage row still contains text or a status."""
    if any(row.get(column) for column in CONTINUABLE_COLUMNS):
        return True

    statuses = row.get("cover_status")
    return isinstance(statuses, dict) and any(
        status is not None for status in statuses.values()
    )


# Maps (table_type, column) to the last row with text in that column and the
# PDF page it was on, so a continuation can be appended to it.
PreviousCells = dict[tuple[str, str], tuple[dict[str, Any], int]]


def consolidate_cross_page_table_continuations(
    pages: list[dict[str, Any]],
) -> None:
    """Merge explicitly marked table continuations into the preceding row.

    A continuation is accepted only from the same table type on the same or
    immediately following PDF page. This keeps the merge deterministic while
    supporting tables split by a page break or an IMPORTANT notice.
    """
    previous_cells: PreviousCells = {}

    for page in pages:
        pdf_page_number = page["pdf_page_number"]
        retained_content = []
        for item in page["content"]:
            if (
                item.get("content_type") != "table"
                or item.get("table_type") not in CONTINUABLE_TABLE_TYPES
            ):
                retained_content.append(item)
                continue

            retained_rows = consolidate_table_rows(
                item, page, pdf_page_number, previous_cells
            )
            if retained_rows:
                item["rows"] = retained_rows
                retained_content.append(item)

        for order, item in enumerate(retained_content, start=1):
            item["order"] = order
        page["content"] = retained_content


def consolidate_table_rows(
    table: dict[str, Any],
    page: dict[str, Any],
    pdf_page_number: int,
    previous_cells: PreviousCells,
) -> list[dict[str, Any]]:
    """Merge continuation cells in a table's rows and return the rows to keep.

    A row whose cells were all merged into earlier rows is dropped.
    """
    table_type = table.get("table_type")
    retained_rows = []
    for row in table.get("rows", []):
        merged = merge_continuation_cells(
            row, table_type, page, pdf_page_number, previous_cells
        )
        if row_has_meaningful_content(row):
            retained_rows.append(row)
            for column in CONTINUABLE_COLUMNS:
                if row.get(column):
                    previous_cells[(table_type, column)] = (row, pdf_page_number)
        elif not merged:
            retained_rows.append(row)
    return retained_rows


def merge_continuation_cells(
    row: dict[str, Any],
    table_type: str,
    page: dict[str, Any],
    pdf_page_number: int,
    previous_cells: PreviousCells,
) -> bool:
    """Append this row's "cont." cells to the earlier cells they continue.

    Merged cells are cleared in ``row``. Returns whether any cell was merged.
    """
    merged = False
    for column in CONTINUABLE_COLUMNS:
        value = row.get(column)
        previous_entry = previous_cells.get((table_type, column))
        if not starts_with_continuation_marker(value) or previous_entry is None:
            continue

        previous_row, previous_pdf_page = previous_entry
        previous_value = previous_row.get(column)
        if pdf_page_number - previous_pdf_page > 1 or not isinstance(
            previous_value, str
        ):
            continue

        previous_row[column] = join_continued_cell(previous_value, value)
        previous_row["continued_through_pdf_page"] = pdf_page_number
        previous_row["continued_through_printed_page"] = page["printed_page_number"]
        previous_cells[(table_type, column)] = (previous_row, pdf_page_number)
        row[column] = None
        merged = True
    return merged


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
                copy_inherited_cells(row, previous_values)
                for column in CONTINUABLE_COLUMNS:
                    if row.get(column):
                        previous_values[column] = row


def copy_inherited_cells(
    row: dict[str, Any], previous_values: dict[str, dict[str, Any]]
) -> None:
    """Fill the row's ``_inherit_columns`` from the rows that last had them."""
    for column in row.pop("_inherit_columns", []):
        source_row = previous_values.get(column)
        if source_row is None:
            continue

        row[column] = source_row.get(column)
        if column == "what_is_covered" and source_row.get("row_heading"):
            row["row_heading"] = source_row["row_heading"]
        for provenance_key in (
            "continued_through_pdf_page",
            "continued_through_printed_page",
        ):
            if provenance_key in source_row:
                row[provenance_key] = source_row[provenance_key]


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

        if character_count < MINIMUM_PAGE_CHARACTERS and pdf_page_number != 70:
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
        start_page for group in CONTENTS_GROUPS for start_page, _ in group["subgroups"]
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

            previous_item = subsection_content[-1] if subsection_content else None
            continues_cross_page_table = (
                item.get("content_type") == "table"
                and item.get("table_type") in MERGEABLE_CROSS_PAGE_TABLE_TYPES
                and isinstance(previous_item, dict)
                and previous_item.get("content_type") == "table"
                and previous_item.get("table_type") == item.get("table_type")
                and previous_item.get("columns") == item.get("columns")
                and previous_item.get("source_pdf_pages")
                and pdf_page_number - previous_item["source_pdf_pages"][-1] in {0, 1}
            )
            if continues_cross_page_table:
                previous_item.setdefault("rows", []).extend(item.get("rows", []))
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
                if match is None or heading.get("heading_level") != 3:
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
    assigned_page_numbers = [page["pdf_page_number"] for page in excluded_pages]
    assigned_page_numbers.extend(
        pdf_page_number
        for section in sections
        for subsection in section["subsections"]
        for pdf_page_number in inclusive_page_numbers(
            subsection["start_pdf_page"],
            subsection["end_pdf_page"],
        )
    )
    expected_page_numbers = [page["pdf_page_number"] for page in pages]

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
    configure_logging(arguments.verbose, arguments.quiet)
    input_directory = arguments.input_dir.expanduser().resolve()
    output_file = arguments.output_file.expanduser().resolve()

    if not input_directory.is_dir():
        logger.error("Input directory not found: %s", input_directory)
        return 1

    pdf_paths = find_hh_pdfs(input_directory)

    if not pdf_paths:
        logger.error("No PDF files beginning with HH- found in %s", input_directory)
        return 1

    documents = []
    failures = []

    for pdf_path in pdf_paths:
        logger.info("Extracting: %s", pdf_path.name)

        try:
            documents.append(extract_document(pdf_path))
        except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
            logger.error("%s: %s", pdf_path.name, error)
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
            "pages_extracted": sum(document["page_count"] for document in documents),
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

    print("Extraction summary")
    print(f"PDF files found:     {len(pdf_paths)}")
    print(f"Documents extracted: {len(documents)}")
    print(f"Documents failed:    {len(failures)}")
    print(f"Pages extracted:     {output['summary']['pages_extracted']}")
    print(f"Warnings:            {len(warnings)}")
    print(f"Output:              {output_file}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
