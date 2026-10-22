#!/usr/bin/env python3
"""Generate a grounded answer from chunks stored in Azure AI Search."""

from __future__ import annotations

import argparse
import logging

from insurance_rag.answering import (
    DEFAULT_TOP,
    answer_question,
    create_answer_pipeline,
    format_answer,
    save_answer,
)
from insurance_rag.logging_setup import add_logging_arguments, configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for generating an answer."""
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve policy chunks with vector search and generate a cited answer."
        )
    )
    parser.add_argument("question", help="The user's insurance question")
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP})",
    )
    parser.add_argument(
        "--document-code",
        help="Optional exact document_code filter, for example IP-HO-2-012",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Include Azure Search scores in the source list",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()

    if not args.question.strip():
        parser.error("question cannot be empty")
    if not 1 <= args.top <= 50:
        parser.error("--top must be between 1 and 50")
    return args


def main() -> int:
    """Retrieve relevant chunks, generate a cited answer and print it.

    Returns:
        The process exit code.
    """
    args = parse_args()
    configure_logging(args.verbose, args.quiet)
    try:
        pipeline = create_answer_pipeline()
        answer = answer_question(
            pipeline, args.question, top=args.top, document_code=args.document_code
        )
        output_path = save_answer(
            answer.question, answer.result.answer, answer.chunks, args.show_scores
        )
    except Exception as exc:  # Keep CLI errors concise and actionable.
        logger.error("%s", exc)
        return 1

    print(format_answer(answer, args.show_scores))
    logger.info("Answer saved to: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
