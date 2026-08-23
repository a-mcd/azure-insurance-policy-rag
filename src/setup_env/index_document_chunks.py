#!/usr/bin/env python3
"""Create embeddings for policy chunks and upload them to Azure AI Search."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI

from azure.core.credentials import AzureKeyCredential
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


DEFAULT_INPUTS = (
    Path("data/processed/ip_document_chunks.jsonl"),
    Path("data/processed/hh_document_chunks.jsonl"),
)
VECTOR_DIMENSIONS = 1536
VECTOR_FIELD = "content_vector"
VECTOR_PROFILE = "insurance-vector-profile"
DEFAULT_VECTOR_ALGORITHM = "hnsw"
ENGLISH_ANALYZER = "en.microsoft"


def parse_args() -> argparse.Namespace:
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
    args = parser.parse_args()
    if args.embedding_batch_size < 1 or args.upload_batch_size < 1:
        parser.error("Batch sizes must be at least 1.")
    return args


def require_environment(names: Iterable[str]) -> dict[str, str]:
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


def load_chunks(paths: list[Path]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

        file_count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                if not isinstance(chunk, dict):
                    raise ValueError(f"Expected a JSON object in {path}:{line_number}")

                chunk_id = chunk.get("chunk_id")
                embedding_text = chunk.get("embedding_text")
                text = chunk.get("text")
                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    raise ValueError(f"Missing chunk_id in {path}:{line_number}")
                if chunk_id in seen_ids:
                    raise ValueError(f"Duplicate chunk_id {chunk_id!r} in {path}:{line_number}")
                if not isinstance(embedding_text, str) or not embedding_text.strip():
                    raise ValueError(f"Missing embedding_text in {path}:{line_number}")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Missing text in {path}:{line_number}")

                seen_ids.add(chunk_id)
                chunks.append(chunk)
                file_count += 1
        print(f"Read {file_count} chunks from {path}")

    if not chunks:
        raise ValueError("No chunks were found in the input files.")
    return chunks


def create_index_definition(index_name: str, vector_algorithm: str) -> SearchIndex:
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
        SimpleField(name="start_pdf_page", type=integer, filterable=True, sortable=True),
        SimpleField(name="end_pdf_page", type=integer, filterable=True, sortable=True),
        SimpleField(name="start_printed_page", type=integer, filterable=True, sortable=True),
        SimpleField(name="end_printed_page", type=integer, filterable=True, sortable=True),
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
    index_name: str,
    recreate: bool,
    vector_algorithm: str,
) -> None:
    if recreate:
        try:
            client.delete_index(index_name)
            print(f"Deleted existing index: {index_name}")
        except Exception as exc:  # Azure returns ResourceNotFoundError if absent.
            if exc.__class__.__name__ != "ResourceNotFoundError":
                raise

    client.create_or_update_index(
        create_index_definition(index_name, vector_algorithm)
    )
    print(f"Index is ready: {index_name} ({vector_algorithm})")


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def add_embeddings(
    chunks: list[dict[str, Any]], client: AzureOpenAI, deployment: str, batch_size: int
) -> None:
    completed = 0
    for batch in batched(chunks, batch_size):
        response = client.embeddings.create(
            model=deployment,
            input=[chunk["embedding_text"] for chunk in batch],
        )
        returned = sorted(response.data, key=lambda item: item.index)
        if len(returned) != len(batch):
            raise RuntimeError(
                f"Embedding response returned {len(returned)} items for a batch of {len(batch)}."
            )
        for chunk, item in zip(batch, returned):
            vector = item.embedding
            if len(vector) != VECTOR_DIMENSIONS:
                raise RuntimeError(
                    f"Chunk {chunk['chunk_id']} returned {len(vector)} dimensions; "
                    f"expected {VECTOR_DIMENSIONS}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError(f"Chunk {chunk['chunk_id']} returned a non-finite vector.")
            chunk[VECTOR_FIELD] = vector
        completed += len(batch)
        print(f"Created embeddings: {completed}/{len(chunks)}")


def prepare_search_document(chunk: dict[str, Any], allowed_fields: set[str]) -> dict[str, Any]:
    # Azure Search rejects properties that are not defined in the index.
    return {key: value for key, value in chunk.items() if key in allowed_fields}


def upload_chunks(
    chunks: list[dict[str, Any]],
    client: SearchClient,
    batch_size: int,
    allowed_fields: set[str],
) -> None:
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
        print(f"Uploaded chunks: {completed}/{len(chunks)}")


def main() -> int:
    load_dotenv()
    args = parse_args()
    chunks = load_chunks(args.inputs)
    print(f"Validated {len(chunks)} unique chunks")

    if args.validate_only:
        print("Validation completed; Azure was not contacted.")
        return 0

    config = require_environment(
        (
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_ADMIN_KEY",
            "AZURE_SEARCH_INDEX_NAME",
        )
    )

    search_credential = AzureKeyCredential(config["AZURE_SEARCH_ADMIN_KEY"])
    index_client = SearchIndexClient(
        endpoint=config["AZURE_SEARCH_ENDPOINT"], credential=search_credential
    )
    create_or_update_index(
        index_client,
        config["AZURE_SEARCH_INDEX_NAME"],
        args.recreate_index,
        args.vector_algorithm,
    )

    openai_client = AzureOpenAI(
        azure_endpoint=config["AZURE_OPENAI_ENDPOINT"],
        api_key=config["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        max_retries=5,
    )
    add_embeddings(
        chunks,
        openai_client,
        config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        args.embedding_batch_size,
    )

    index = create_index_definition(
        config["AZURE_SEARCH_INDEX_NAME"], args.vector_algorithm
    )
    allowed_fields = {field.name for field in index.fields}
    search_client = SearchClient(
        endpoint=config["AZURE_SEARCH_ENDPOINT"],
        index_name=config["AZURE_SEARCH_INDEX_NAME"],
        credential=search_credential,
    )
    upload_chunks(chunks, search_client, args.upload_batch_size, allowed_fields)

    print("\nIndexing completed successfully")
    print(f"Files processed: {len(args.inputs)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Embedding dimensions: {VECTOR_DIMENSIONS}")
    print(f"Vector algorithm: {args.vector_algorithm}")
    print(f"Index: {config['AZURE_SEARCH_INDEX_NAME']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
