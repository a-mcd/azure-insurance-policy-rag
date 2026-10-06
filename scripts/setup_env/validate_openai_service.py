import os

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
)

response = client.embeddings.create(
    model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    input="Storm damage to gates, fences and hedges.",
)

embedding = response.data[0].embedding

print(f"Embedding dimensions: {len(embedding)}")
print(f"First five values: {embedding[:5]}")