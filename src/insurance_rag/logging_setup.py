"""Consistent logging for the command-line scripts.

Results a script was asked to produce (answers, search results, summaries)
are printed to stdout. Progress, warnings and errors are logged to stderr, so
stdout can be redirected or piped without status messages mixed in.
"""

from __future__ import annotations

import argparse
import logging


class _LevelPrefixFormatter(logging.Formatter):
    """Show info messages as plain text and prefix warnings and errors."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{record.levelname}: {message}"
        return message


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive ``--verbose`` and ``--quiet`` options to a parser."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed debug messages as well as progress.",
    )
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Hide progress messages and show only warnings and errors.",
    )


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Send log messages to stderr at a level chosen by the verbosity flags.

    The default level is INFO, which shows progress messages. ``verbose``
    lowers it to DEBUG and ``quiet`` raises it to WARNING. Messages from
    third-party libraries are limited to warnings unless ``verbose`` is set.
    """
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO

    handler = logging.StreamHandler()  # Writes to stderr.
    handler.setFormatter(_LevelPrefixFormatter("%(message)s"))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.WARNING)

    # Our own loggers: the package, and scripts run as ``__main__``.
    for name in ("insurance_rag", "__main__"):
        logging.getLogger(name).setLevel(level)
