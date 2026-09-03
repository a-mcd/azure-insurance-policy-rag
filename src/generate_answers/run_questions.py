#!/usr/bin/env python3
"""Run generate_answer.py once for every question in a JSON file."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Read questions from JSON and run generate_answer.py for each one."
    )
    parser.add_argument(
        "questions_file",
        type=Path,
        help="JSON file containing a list of questions or a {'questions': [...]} object.",
    )
    parser.add_argument(
        "--generate-script",
        type=Path,
        default=script_dir / "generate_answer.py",
        help="Path to generate_answer.py (default: next to this script).",
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
        help="Stop immediately if generate_answer.py fails for a question.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
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
    if not requested_ids:
        return questions

    requested = set(requested_ids)
    available = {question_id for question_id, _ in questions}
    missing = requested - available

    if missing:
        raise ValueError(f"Question ID(s) not found: {', '.join(sorted(missing))}")

    return [item for item in questions if item[0] in requested]


def validate_generate_script(path: Path) -> Path:
    resolved_path = path.expanduser().resolve()

    if not resolved_path.is_file():
        raise ValueError(f"generate_answer.py not found: {resolved_path}")

    return resolved_path


def run_questions(
    questions: list[tuple[str, str]],
    generate_script: Path,
    stop_on_error: bool,
) -> int:
    failures: list[str] = []
    total = len(questions)

    for number, (question_id, question) in enumerate(questions, start=1):
        print(f"\n[{number}/{total}] {question_id}: {question}", flush=True)

        result = subprocess.run(
            [sys.executable, str(generate_script), question],
            cwd=generate_script.parent,
            check=False,
        )

        if result.returncode == 0:
            print(f"Completed {question_id}", flush=True)
            continue

        failures.append(question_id)
        print(
            f"Failed {question_id}: generate_answer.py exited with "
            f"status {result.returncode}",
            file=sys.stderr,
            flush=True,
        )

        if stop_on_error:
            break

    completed = total - len(failures)
    print(f"\nFinished: {completed}/{total} succeeded.")

    if failures:
        print(f"Failed question ID(s): {', '.join(failures)}", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    args = parse_args()

    try:
        data = load_json(args.questions_file.expanduser().resolve())
        questions = extract_questions(data)
        questions = select_questions(questions, args.question_ids)
        generate_script = validate_generate_script(args.generate_script)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    return run_questions(questions, generate_script, args.stop_on_error)


if __name__ == "__main__":
    raise SystemExit(main())
