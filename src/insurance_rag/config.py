"""Project paths, search index constants and settings loaded from ``.env``."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# The package is installed in editable mode from ``src/``, so the project root
# is two levels above this package directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# These must match the index created by scripts/chunks/index_document_chunks.py.
VECTOR_FIELD = "content_vector"
VECTOR_DIMENSIONS = 1536

OPENAI_MAX_RETRIES = 5

# Chat completion defaults, suited to GPT-5 reasoning models. GPT-5 counts
# hidden reasoning and visible answer tokens against the same completion
# limit; minimal reasoning is sufficient for grounded RAG and leaves more of
# the allowance for the cited answer. Override them in .env for other models.
DEFAULT_REASONING_EFFORT = "minimal"
DEFAULT_MAX_COMPLETION_TOKENS = 2000


@dataclass(frozen=True)
class OpenAISettings:
    """Azure OpenAI endpoint, key and model deployment names."""

    endpoint: str
    api_key: str
    embedding_deployment: str


@dataclass(frozen=True)
class ChatSettings:
    """The chat model deployment and the options sent with each request."""

    deployment: str
    # None means the reasoning_effort parameter is not sent, which is required
    # for chat models that do not support reasoning.
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS


@dataclass(frozen=True)
class SearchSettings:
    """Azure AI Search endpoint, key and index name."""

    endpoint: str
    key: str
    index_name: str


def load_environment() -> None:
    """Load variables from the project's ``.env`` file without overriding the shell."""
    load_dotenv(ENV_FILE)


def require_environment(names: Iterable[str]) -> dict[str, str]:
    """Return the named environment variables, failing if any are missing or empty.

    Raises:
        ValueError: If any of the variables is missing or blank.
    """
    values: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            values[name] = value
        else:
            missing.append(name)
    if missing:
        raise ValueError("Missing environment variables: " + ", ".join(missing))
    return values


def load_openai_settings() -> OpenAISettings:
    """Load the Azure OpenAI endpoint, key and embedding deployment.

    Raises:
        ValueError: If a required environment variable is missing.
    """
    values = require_environment(
        (
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        )
    )
    return OpenAISettings(
        endpoint=values["AZURE_OPENAI_ENDPOINT"],
        api_key=values["AZURE_OPENAI_API_KEY"],
        embedding_deployment=values["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    )


def load_chat_settings() -> ChatSettings:
    """Load the chat deployment and its request options.

    ``AZURE_OPENAI_REASONING_EFFORT`` defaults to ``minimal`` when it is not
    set. Setting it to an empty value omits the parameter, which non-reasoning
    models such as GPT-4o require. ``AZURE_OPENAI_MAX_COMPLETION_TOKENS``
    defaults to 2000.

    Raises:
        ValueError: If the chat deployment is missing or the token limit is not
            a positive whole number.
    """
    deployment = require_environment(("AZURE_OPENAI_CHAT_DEPLOYMENT",))[
        "AZURE_OPENAI_CHAT_DEPLOYMENT"
    ]

    reasoning_effort = os.getenv("AZURE_OPENAI_REASONING_EFFORT")
    if reasoning_effort is None:
        reasoning_effort = DEFAULT_REASONING_EFFORT
    reasoning_effort = reasoning_effort.strip() or None

    raw_max_tokens = os.getenv("AZURE_OPENAI_MAX_COMPLETION_TOKENS", "").strip()
    try:
        max_completion_tokens = (
            int(raw_max_tokens) if raw_max_tokens else DEFAULT_MAX_COMPLETION_TOKENS
        )
    except ValueError:
        max_completion_tokens = 0
    if max_completion_tokens <= 0:
        raise ValueError(
            "AZURE_OPENAI_MAX_COMPLETION_TOKENS must be a positive whole number, "
            f"not {raw_max_tokens!r}."
        )

    return ChatSettings(
        deployment=deployment,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
    )


def load_search_settings(*, admin: bool = False) -> SearchSettings:
    """Load Azure AI Search settings.

    Args:
        admin: Require the admin key, which is needed to create indexes and
            upload documents. Otherwise the read-only query key is preferred and
            the admin key is only used as a fallback.

    Raises:
        ValueError: If a required environment variable is missing.
    """
    values = require_environment(("AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_INDEX_NAME"))
    if admin:
        key = require_environment(("AZURE_SEARCH_ADMIN_KEY",))["AZURE_SEARCH_ADMIN_KEY"]
    else:
        key = (
            os.getenv("AZURE_SEARCH_QUERY_KEY", "").strip()
            or os.getenv("AZURE_SEARCH_ADMIN_KEY", "").strip()
        )
        if not key:
            raise ValueError(
                "Missing AZURE_SEARCH_QUERY_KEY or AZURE_SEARCH_ADMIN_KEY "
                "environment variable."
            )
    return SearchSettings(
        endpoint=values["AZURE_SEARCH_ENDPOINT"],
        key=key,
        index_name=values["AZURE_SEARCH_INDEX_NAME"],
    )
