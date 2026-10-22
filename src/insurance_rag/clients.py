"""Factories for the Azure OpenAI and Azure AI Search clients."""

from __future__ import annotations

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import OpenAI

from insurance_rag.config import OPENAI_MAX_RETRIES, OpenAISettings, SearchSettings


def create_openai_client(settings: OpenAISettings) -> OpenAI:
    """Create an OpenAI client for the Azure OpenAI v1 endpoint.

    The v1 endpoint serves both embeddings and chat completions and needs no
    ``api_version``, so one client type is used everywhere.
    """
    return OpenAI(
        base_url=settings.endpoint.rstrip("/") + "/openai/v1/",
        api_key=settings.api_key,
        max_retries=OPENAI_MAX_RETRIES,
    )


def create_search_client(settings: SearchSettings) -> SearchClient:
    """Create a client for querying and uploading documents to the index."""
    return SearchClient(
        endpoint=settings.endpoint,
        index_name=settings.index_name,
        credential=AzureKeyCredential(settings.key),
    )


def create_index_client(settings: SearchSettings) -> SearchIndexClient:
    """Create a client for creating, updating and deleting search indexes."""
    return SearchIndexClient(
        endpoint=settings.endpoint,
        credential=AzureKeyCredential(settings.key),
    )
