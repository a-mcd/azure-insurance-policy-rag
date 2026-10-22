#!/usr/bin/env python3
"""Check the Azure OpenAI embedding deployment by requesting one test embedding."""

from insurance_rag.clients import create_openai_client
from insurance_rag.config import load_environment, load_openai_settings


def main() -> None:
    """Create a test embedding and print its size and first values."""
    load_environment()
    settings = load_openai_settings()
    client = create_openai_client(settings)

    response = client.embeddings.create(
        model=settings.embedding_deployment,
        input="Storm damage to gates, fences and hedges.",
    )

    embedding = response.data[0].embedding

    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First five values: {embedding[:5]}")


if __name__ == "__main__":
    main()
