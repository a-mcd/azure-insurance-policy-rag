#!/usr/bin/env python3
"""Create embeddings for policy chunks and upload them to Azure AI Search."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ExhaustiveKnnAlgorithmConfiguration,
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from openai import OpenAI

from insurance_rag.clients import (
    create_index_client,
    create_openai_client,
    create_search_client,
)
from insurance_rag.config import (
    DATA_DIR,
    VECTOR_DIMENSIONS,
    VECTOR_FIELD,
    load_environment,
    load_openai_settings,
    load_search_settings,
)
from insurance_rag.logging_setup import add_logging_arguments, configure_logging

logger = logging.getLogger(__name__)

DEFAULT_INPUTS = (
    DATA_DIR / "processed" / "ip_document_chunks.jsonl",
    DATA_DIR / "processed" / "hh_document_chunks.jsonl",
)
VECTOR_PROFILE = "insurance-vector-profile"
DEFAULT_VECTOR_ALGORITHM = "hnsw"
ENGLISH_ANALYZER = "en.microsoft"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for indexing document chunks."""
    parser = argparse.ArgumentParser(
        description=(
            "Read JSONL document chunks, create Azure OpenAI embeddings, "
            "and upload the chunks to Azure AI Search."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_INPUTS),
        help="Chunk JSONL files (defaults to the IP and HH processed files).",
    )
    parser.add_argument(
        "--vector-algorithm",
        choices=("hnsw", "eknn"),
        default=DEFAULT_VECTOR_ALGORITHM,
        help=(
            "Vector search algorithm used by the index: hnsw or exhaustive "
            f"KNN (eknn) (default: {DEFAULT_VECTOR_ALGORITHM})."
        ),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Texts sent in each embedding request (default: 16).",
    )
    parser.add_argument(
        "--upload-batch-size",
        type=int,
        default=100,
        help="Documents sent in each Azure AI Search upload (default: 100).",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete and recreate the index. This removes all existing index data.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the input files without contacting Azure.",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    if args.embedding_batch_size < 1 or args.upload_batch_size < 1:
        parser.error("Batch sizes must be at least 1.")
    return args


def load_chunks(paths: list[Path]) -> list[dict[str, Any]]:
    """Load chunks from JSONL files, validating required fields and unique IDs."""
    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for path in paths:
        file_count = 0
        for line_number, chunk in read_jsonl_objects(path):
            chunk_id = validate_chunk(chunk, f"{path}:{line_number}", seen_ids)
            seen_ids.add(chunk_id)
            chunks.append(chunk)
            file_count += 1
        logger.info("Read %d chunks from %s", file_count, path)

    if not chunks:
        raise ValueError("No chunks were found in the input files.")
    return chunks


def read_jsonl_objects(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, object)`` for each non-blank line of a JSONL file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line is not valid JSON or not a JSON object.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            yield line_number, value


def validate_chunk(chunk: dict[str, Any], location: str, seen_ids: set[str]) -> str:
    """Check a chunk's required fields and return its ID.

    Args:
        chunk: The chunk read from a JSONL file.
        location: ``path:line`` used in error messages.
        seen_ids: Chunk IDs already loaded, to detect duplicates.

    Raises:
        ValueError: If a required field is missing or the chunk ID is duplicated.
    """
    chunk_id = chunk.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError(f"Missing chunk_id in {location}")
    if chunk_id in seen_ids:
        raise ValueError(f"Duplicate chunk_id {chunk_id!r} in {location}")
    for field in ("embedding_text", "text"):
        value = chunk.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing {field} in {location}")
    return chunk_id


def create_index_definition(index_name: str, vector_algorithm: str) -> SearchIndex:
    """Build the Azure AI Search index schema, including the vector search profile."""
    string = SearchFieldDataType.String
    integer = SearchFieldDataType.Int32
    boolean = SearchFieldDataType.Boolean

    fields = [
        SimpleField(name="chunk_id", type=string, key=True, filterable=True),
        SimpleField(name="chunk_sequence", type=integer, sortable=True),
        SimpleField(name="chunk_type", type=string, filterable=True),
        SimpleField(name="document_code", type=string, filterable=True),
        SimpleField(name="document_name", type=string, filterable=True),
        SimpleField(name="document_sha256", type=string, filterable=True),
        SimpleField(name="document_type", type=string, filterable=True),
        SearchField(
            name="title",
            type=string,
            searchable=True,
            filterable=True,
            analyzer_name=ENGLISH_ANALYZER,
        ),
        SimpleField(name="fca_reference_number", type=string, filterable=True),
        SearchField(
            name="section_heading",
            type=string,
            searchable=True,
            filterable=True,
            analyzer_name=ENGLISH_ANALYZER,
        ),
        SearchField(
            name="subsection_headings",
            type=SearchFieldDataType.Collection(string),
            searchable=True,
            filterable=True,
            analyzer_name=ENGLISH_ANALYZER,
        ),
        SearchField(
            name="semantic_heading",
            type=string,
            searchable=True,
            filterable=True,
            analyzer_name=ENGLISH_ANALYZER,
        ),
        SimpleField(name="semantic_heading_level", type=integer, filterable=True),
        SearchField(
            name="text",
            type=string,
            searchable=True,
            analyzer_name=ENGLISH_ANALYZER,
        ),
        SearchField(
            name="embedding_text",
            type=string,
            searchable=True,
            analyzer_name=ENGLISH_ANALYZER,
        ),
        SimpleField(name="token_count", type=integer, filterable=True, sortable=True),
        SimpleField(
            name="start_pdf_page", type=integer, filterable=True, sortable=True
        ),
        SimpleField(name="end_pdf_page", type=integer, filterable=True, sortable=True),
        SimpleField(
            name="start_printed_page", type=integer, filterable=True, sortable=True
        ),
        SimpleField(
            name="end_printed_page", type=integer, filterable=True, sortable=True
        ),
        SearchField(
            name="source_pdf_pages",
            type=SearchFieldDataType.Collection(integer),
            filterable=True,
        ),
        SearchField(
            name="source_printed_pages",
            type=SearchFieldDataType.Collection(integer),
            filterable=True,
        ),
        SimpleField(name="source_start_order", type=integer, sortable=True),
        SimpleField(name="source_end_order", type=integer, sortable=True),
        SimpleField(name="table_type", type=string, filterable=True),
        SimpleField(name="contains_complete_table", type=boolean, filterable=True),
        SimpleField(name="contains_complete_rows", type=boolean, filterable=True),
        SimpleField(name="table_is_split", type=boolean, filterable=True),
        SimpleField(name="table_chunk_index", type=integer),
        SimpleField(name="table_chunk_count", type=integer),
        SimpleField(name="table_row_start", type=integer),
        SimpleField(name="table_row_end", type=integer),
        SearchField(
            name=VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
    ]

    algorithm_name = f"insurance-{vector_algorithm}"
    if vector_algorithm == "hnsw":
        algorithm_configuration = HnswAlgorithmConfiguration(name=algorithm_name)
    elif vector_algorithm == "eknn":
        algorithm_configuration = ExhaustiveKnnAlgorithmConfiguration(
            name=algorithm_name
        )
    else:
        raise ValueError(f"Unsupported vector algorithm: {vector_algorithm}")

    vector_search = VectorSearch(
        algorithms=[algorithm_configuration],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE,
                algorithm_configuration_name=algorithm_name,
            )
        ],
    )

    return SearchIndex(name=index_name, fields=fields, vector_search=vector_search)


def create_or_update_index(
    client: SearchIndexClient,
    index: SearchIndex,
    recreate: bool,
) -> None:
    """Create or update the search index, deleting it first when ``recreate`` is set."""
    if recreate:
        try:
            client.delete_index(index.name)
            logger.info("Deleted existing index: %s", index.name)
        except ResourceNotFoundError:
            # Nothing to delete on the first run or after a manual delete.
            logger.info("No existing index to delete: %s", index.name)

    client.create_or_update_index(index)
    logger.info("Index is ready: %s", index.name)


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    """Yield successive slices of ``items`` containing at most ``size`` elements."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def add_embeddings(
    chunks: list[dict[str, Any]], client: OpenAI, deployment: str, batch_size: int
) -> None:
    """Generate an embedding for each chunk in batches and store it on the chunk.

    Raises:
        RuntimeError: If the response size, vector dimensions or values are invalid.
    """
    completed = 0
    for batch in batched(chunks, batch_size):
        response = client.embeddings.create(
            model=deployment,
            input=[chunk["embedding_text"] for chunk in batch],
        )
        returned = sorted(response.data, key=lambda item: item.index)
        if len(returned) != len(batch):
            raise RuntimeError(
                f"Embedding response returned {len(returned)} items "
                f"for a batch of {len(batch)}."
            )
        for chunk, item in zip(batch, returned, strict=True):
            vector = item.embedding
            if len(vector) != VECTOR_DIMENSIONS:
                raise RuntimeError(
                    f"Chunk {chunk['chunk_id']} returned {len(vector)} dimensions; "
                    f"expected {VECTOR_DIMENSIONS}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError(
                    f"Chunk {chunk['chunk_id']} returned a non-finite vector."
                )
            chunk[VECTOR_FIELD] = vector
        completed += len(batch)
        logger.info("Created embeddings: %d/%d", completed, len(chunks))


def prepare_search_document(
    chunk: dict[str, Any], allowed_fields: set[str]
) -> dict[str, Any]:
    """Return a copy of the chunk containing only fields defined in the index."""
    # Azure Search rejects properties that are not defined in the index.
    return {key: value for key, value in chunk.items() if key in allowed_fields}


def upload_chunks(
    chunks: list[dict[str, Any]],
    client: SearchClient,
    batch_size: int,
    allowed_fields: set[str],
) -> None:
    """Upload chunks to Azure AI Search in batches, failing on any rejected document."""
    completed = 0
    for batch in batched(chunks, batch_size):
        documents = [prepare_search_document(chunk, allowed_fields) for chunk in batch]
        results = client.upload_documents(documents=documents)
        failures = [result for result in results if not result.succeeded]
        if failures:
            details = "; ".join(
                f"{result.key}: {result.error_message}" for result in failures
            )
            raise RuntimeError(f"Azure AI Search rejected documents: {details}")
        completed += len(batch)
        logger.info("Uploaded chunks: %d/%d", completed, len(chunks))


def main() -> int:
    """Validate the chunks, then embed and upload them to Azure AI Search.

    Returns:
        The process exit code.
    """
    load_environment()
    args = parse_args()
    configure_logging(args.verbose, args.quiet)
    chunks = load_chunks(args.inputs)
    logger.info("Validated %d unique chunks", len(chunks))

    if args.validate_only:
        logger.info("Validation completed; Azure was not contacted.")
        return 0

    openai_settings = load_openai_settings()
    search_settings = load_search_settings(admin=True)

    index = create_index_definition(search_settings.index_name, args.vector_algorithm)
    create_or_update_index(
        create_index_client(search_settings),
        index,
        args.recreate_index,
    )

    add_embeddings(
        chunks,
        create_openai_client(openai_settings),
        openai_settings.embedding_deployment,
        args.embedding_batch_size,
    )

    allowed_fields = {field.name for field in index.fields}
    upload_chunks(
        chunks,
        create_search_client(search_settings),
        args.upload_batch_size,
        allowed_fields,
    )

    print("Indexing completed successfully")
    print(f"Files processed: {len(args.inputs)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Embedding dimensions: {VECTOR_DIMENSIONS}")
    print(f"Vector algorithm: {args.vector_algorithm}")
    print(f"Index: {search_settings.index_name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
