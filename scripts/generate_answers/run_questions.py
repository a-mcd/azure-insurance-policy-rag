#!/usr/bin/env python3
"""Generate a grounded answer for every question in a JSON file."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from insurance_rag.answering import (
    AnswerPipeline,
    answer_question,
    create_answer_pipeline,
    format_answer,
    save_answer,
)
from insurance_rag.logging_setup import add_logging_arguments, configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for running a batch of questions."""
    parser = argparse.ArgumentParser(
        description="Read questions from JSON and generate an answer for each one."
    )
    parser.add_argument(
        "questions_file",
        type=Path,
        help=(
            "JSON file containing a list of questions or a {'questions': [...]} object."
        ),
    )
    parser.add_argument(
        "--question-id",
        action="append",
        dest="question_ids",
        help="Run only this question ID. May be supplied more than once.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if answering a question fails.",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    """Load JSON from ``path``, converting read and parse errors to ``ValueError``."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"Questions file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path} at line {error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ) from error


def extract_questions(data: Any) -> list[tuple[str, str]]:
    """Return ``(question_id, question)`` pairs from the loaded JSON.

    Accepts a list, or an object with a ``questions`` list, whose items are
    strings or objects with a ``question`` and optional ID. Items without an
    ID are numbered ``Q001``, ``Q002`` and so on.
    """
    if isinstance(data, dict):
        data = data.get("questions")

    if not isinstance(data, list):
        raise ValueError(
            "The JSON must be a list or an object containing a 'questions' list."
        )

    questions: list[tuple[str, str]] = []

    for position, item in enumerate(data, start=1):
        default_id = f"Q{position:03d}"

        if isinstance(item, str):
            question_id = default_id
            question = item.strip()
        elif isinstance(item, dict):
            raw_id = item.get("question_id", item.get("id", default_id))
            raw_question = item.get("question")

            if not isinstance(raw_id, str) or not raw_id.strip():
                raise ValueError(f"Question {position} has an invalid question ID.")
            if not isinstance(raw_question, str):
                raise ValueError(
                    f"Question {raw_id!r} must contain a string 'question' field."
                )

            question_id = raw_id.strip()
            question = raw_question.strip()
        else:
            raise ValueError(
                f"Question {position} must be a string or an object, "
                f"not {type(item).__name__}."
            )

        if not question:
            raise ValueError(f"Question {question_id!r} is empty.")

        questions.append((question_id, question))

    if not questions:
        raise ValueError("The questions file does not contain any questions.")

    return questions


def select_questions(
    questions: list[tuple[str, str]], requested_ids: list[str] | None
) -> list[tuple[str, str]]:
    """Return only the requested questions, or all of them if none are requested."""
    if not requested_ids:
        return questions

    requested = set(requested_ids)
    available = {question_id for question_id, _ in questions}
    missing = requested - available

    if missing:
        raise ValueError(f"Question ID(s) not found: {', '.join(sorted(missing))}")

    return [item for item in questions if item[0] in requested]


def run_questions(
    pipeline: AnswerPipeline,
    questions: list[tuple[str, str]],
    stop_on_error: bool,
) -> int:
    """Answer each question, save it and print it, continuing past failures.

    Returns:
        0 if every question succeeded, otherwise 1.
    """
    failures: list[str] = []
    total = len(questions)

    for number, (question_id, question) in enumerate(questions, start=1):
        # The header labels the answer that follows, so it is part of the output.
        print(f"\n[{number}/{total}] {question_id}: {question}", flush=True)

        try:
            answer = answer_question(pipeline, question)
            output_path = save_answer(
                answer.question, answer.result.answer, answer.chunks, show_scores=False
            )
        except Exception as exc:  # Record the failure and move on.
            failures.append(question_id)
            logger.error("Failed %s: %s", question_id, exc)
            if stop_on_error:
                break
            continue

        print(format_answer(answer, show_scores=False), flush=True)
        logger.info("Answer saved to: %s", output_path)
        logger.info("Completed %s", question_id)

    completed = total - len(failures)
    print(f"\nFinished: {completed}/{total} succeeded.")

    if failures:
        logger.error("Failed question ID(s): %s", ", ".join(failures))
        return 1

    return 0


def main() -> int:
    """Load the questions and run them.

    Returns:
        The process exit code.
    """
    args = parse_args()
    configure_logging(args.verbose, args.quiet)

    try:
        data = load_json(args.questions_file.expanduser().resolve())
        questions = extract_questions(data)
        questions = select_questions(questions, args.question_ids)
    except ValueError as error:
        logger.error("%s", error)
        return 2

    try:
        pipeline = create_answer_pipeline()
    except ValueError as error:
        logger.error("%s", error)
        return 1

    return run_questions(pipeline, questions, args.stop_on_error)


if __name__ == "__main__":
    raise SystemExit(main())
