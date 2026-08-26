# Home Insurance RAG (Retrieval → Augmentation → Generation)

An end-to-end retrieval-augmented generation (RAG) pipeline for answering questions about home insurance documents using Azure OpenAI and Azure AI Search.

The pipeline:

1. extracts structured content from PDF documents

2. divides the extracted content into retrieval-friendly chunks

3. creates vector embeddings for each chunk

4. indexes the chunks in Azure AI Search

5. retrieves relevant chunks for a user's question; and

6. generates a grounded answer with source citations.


## RAG workflow

- Retrieval: keyword, vector, or hybrid search finds relevant policy chunks.
- Augmentation: the retrieved chunks are added to the language-model prompt.
- Generation: the model answers using only the supplied chunks.


## Supported documents

Insurance Product Information (IP) documents: concise product summaries covering the main benefits, exclusions, restrictions, and customer obligations.

Home insurance handbook (HH) documents: detailed policy wording, cover information, conditions, and exclusions.

## Project structure

```text
.
├── data/
│   ├── raw/                              # Source PDF documents
│   └── processed/                        # Extracted JSON and chunk JSONL files
├── setup_env/
│   └── setup_azure_openai.sh             # Creates and configures Azure resources
└── src/
    ├── prep_data/
    │   ├── extract_ip_pdfs.py            # Extracts IP PDFs into structured JSON
    │   ├── extract_hh_pdfs.py            # Extracts the HH PDF into structured JSON
    │   └── chunk_documents.py             # Creates retrieval chunks
    ├── setup_env/
    │   ├── index_document_chunks.py      # Embeds and indexes chunks
    │   └── test_openai_service.py         # Tests the embedding deployment
    ├── search_chunks/
    │   ├── search_document_chunks.py     # Runs ad hoc retrieval queries
    │   └── evaluate_retrieval.py          # Evaluates retrieval performance
    └── generate_answers/
        └── generate_answer.py             # Produces grounded, cited answers
```


## Requirements

- Python 3
- Azure CLI
- An Azure subscription. Login with `az login`
- Source PDFs placed in data/raw
- Project dependencies installed from requirements.txt file, run: `python3 -m pip install -r requirements.txt`


## Setup Environment

Run all commands from the project root.

1. Extract the PDF documents

Extract the Insurance Product Information documents:

```bash
python3 src/prep_data/extract_ip_pdfs.py \
  --input-dir data/raw \
  --output-file data/processed/ip_documents.json
```
```bash
python3 src/prep_data/extract_hh_pdfs.py \
  --input-dir data/raw \
  --output-file data/processed/hh_documents.json
```

> [!NOTE]
> These scripts were developed and tested on macOS. PDF text positioning and line boundaries may differ across operating systems or dependency versions, which can affect how successfully the documents are parsed.
>
> For consistent and reproducible extraction, consider running the scripts in a Docker container with fixed Python and package versions. The parsing boundaries can then be configured and tested against that stable environment.

After extraction, compare the generated JSON with the source PDFs and confirm that:

- headings and paragraphs are identified correctly
- bullet points remain grouped with their parent content
- tables contain the correct rows and columns
- words split across PDF lines are reconstructed correctly
- page references are accurate
- spelling is correct
- no content is missing, duplicated, or incorrectly merged

An LLM can help identify potential discrepancies, but verify every reported difference against the source PDF before changing the extraction logic.


2. Create document chunks

Chunk the Insurance Product Information documents:

```bash
python3 src/prep_data/chunk_documents.py  \
  data/processed/ip_documents.json \
  --output data/processed/ip_document_chunks.jsonl
```

Chunk the home insurance handbook:
```bash
python3 src/prep_data/chunk_documents.py \
  data/processed/hh_documents.json \
  --output data/processed/hh_document_chunks.jsonl
```


3. Create the Azure environment and index the chunks

Run the setup script to:

- Create the Azure resource group.
- Create an Azure OpenAI resource.
- Deploys the embedding and chat models
- Create a Free-tier Azure AI Search service.
- Add the Azure OpenAI and Azure AI Search credentials to .env.
- Test the Azure OpenAI embedding deployment.
- Check that the processed chunk files exist.
- Create the search index and upload the document chunks.


Replace ******** with a unique resource prefix:

If no --vector-algorithm is supplied then it defualts to HNSW. EKNN can also be supplied.


```bash
./setup_env/setup_azure_openai.sh --prefix ********

./setup_env/setup_azure_openai.sh \
  --prefix aug17 \
  --vector-algorithm hnsw/eknn
```

HNSW is suitable as the production default because it scales efficiently as the index grows. Exhaustive KNN is useful as an exact baseline when evaluating whether approximate search reduces retrieval quality.


This bash script uses the python scripts in folder src/setup_env

*Updating the indexed chunks*

If updating the chunks or the vector algorithm use the following command which generates the new embeddings, deletes and recreates the existing search index, and uploads the latest chunks.

```bash
python3 src/setup_env/index_document_chunks.py \
  data/processed/ip_document_chunks.jsonl \
  data/processed/hh_document_chunks.jsonl \
  --vector-algorithm hnsw/eknn \
  --recreate-index
```


## Search Indexed Chunks and Evaluate Retrieval

### search_document_chunks.py

Use search_document_chunks.py to inspect the chunks retrieved for an ad hoc questions.

- Decide which is the best retreval type vector/keyword/hybrid
- Do the chunks needs to be updated?
- Which vector algorithm is most suitable hnsw or eknn

```bash
python3 src/search_chunks/search_document_chunks.py \
  "QUESTION" \
  [--mode hybrid|vector|keyword] \
  [--top NUMBER] \
  [--document-code CODE] \
  [--document-type TYPE] \
  [--json]
```

| Parameter | Required? | Default | Description |
| --- | --- | --- | --- |
| `question` | Yes | — | Question or search text. It must not be empty. |
| `--mode` | No | `hybrid` | Retrieval method: <br /> - vector: Vector search retrieves chunks based on semantic similarity. <br /> keyword: Keyword search retrieves chunks using matching words and phrases.<br />, hybrid: combines vector similarity with keyword matching. It is the default retrieval method. |
| `--top` | No | `5` | Number of chunks to return. Accepted range: 1–50. |
| `--document-code` | No | All documents | Restricts results to an exact document code, such as `IP-HO-2-012`. |
| `--document-type` | No | All document types | Restricts results to a document type, such as `insurance_product_information` or `home_insurance_policy_booklet`. |
| `--json` | No | Disabled | Prints machine-readable JSON instead of formatted text. |


### evaluate_retrieval.py

Evaluates retrieved chunk IDs against the expected chunk IDs in the labelled question set documented in retrieval_questions.json. This script has no required command-line parameters.

```bash
python3 src/search_chunks/evaluate_retrieval.py \
  [--input PATH] \
  [--output PATH] \
  [--modes MODE [MODE ...]] \
  [--top NUMBER] \
  [--question-id ID]
```

| Parameter | Required? | Default | Description |
| --- | --- | --- | --- |
| `--input` | No | `src/search_chunks/retrieval_questions.json` | Evaluation questions JSON file. |
| `--output` | No | `src/search_chunks/retrieval_results.json` | Destination for detailed JSON results. |
| `--modes` | No | `keyword vector hybrid` | One or more retrieval modes. Accepted values: `keyword`, `vector`, and `hybrid`. |
| `--top` | No | `5` | Number of chunks retrieved for each question. Accepted range: 1–50. |
| `--question-id` | No | All questions | Evaluates only the specified question, such as `Q006`. |

Compare multiple selected modes:

```bash
python3 src/search_chunks/evaluate_retrieval.py \
  --modes vector hybrid \
  --top 10
```

Evaluate one question only:

```bash
python3 src/search_chunks/evaluate_retrieval.py \
  --question-id Q006 \
  --modes vector \
  --top 10
```


## Generate a grounded answer

Retrieves policy chunks using vector search and generates a grounded answer with numbered citations.


```bash
python3 src/generate_answers/generate_answer.py \
  "QUESTION" \
  [--top NUMBER] \
  [--document-code CODE] \
  [--vector-field FIELD] \
  [--show-scores]
```

| Parameter | Required? | Default | Description |
| --- | --- | --- | --- |
| `question` | Yes | — | Insurance question to answer. It must not be empty. |
| `--top` | No | `10` | Number of chunks supplied to the chat model. Accepted range: 1–50. |
| `--document-code` | No | All documents | Restricts retrieval to an exact document code, such as `IP-HO-2-012`. |
| `--vector-field` | No | Automatically detected | Selects the Azure AI Search vector field. Normally this does not need to be supplied. |
| `--show-scores` | No | Disabled | Includes Azure AI Search scores in the printed source list. |

Only `question` is required:

```bash
python3 src/generate_answers/generate_answer.py \
  "Do I need to list my £3,000 violin separately?"
```

Example using optional parameters:

```bash
python3 src/generate_answers/generate_answer.py \
  "Are fences covered for storm damage?" \
  --document-code IP-HO-2-012 \
  --top 10 \
  --show-scores
```


## Costs

### Deploy text-embedding-3-small
£0.00 per month while unused

### Running index_document_chunks.py
Cost to embed all chunks by running index script, using text-embedding-3-small

| Chunk file                 | Number of chunks | Total tokens |
| -------------------------- | ---------------: | -----------: |
| `ip_document_chunks.jsonl` |              228 |       15,182 |
| `hh_document_chunks.jsonl` |              256 |       49,067 |
| **Total**                  |          **484** |   **64,249** |

- £0.000019 	Per 1,000 tokens
- 20 ÷ 1,000 × £0.000019 = £0.00000038

### Store indexed chunks

| Azure AI Search Free-tier service                | Limit | Used |
| -------------------------- | ---------------: | -----------: |
| Storage |              50 MB |       9.91 MB |
| Vector index quota usage |              25 MB8.58 MB |       8.58 MB |


### Overall GPT-5-mini question costs

GPT-5-mini Global pricing:

- £0 while deployed but unused
- Input: £0.19 per 1 million tokens
- Cached input: £0.02 per 1 million tokens
- Output and reasoning: £1.51 per 1 million tokens
- Maximum completion tokens: 2,000 per request
- Reasoning effort: Minimal

The following results use the actual token usage reported by Azure for five questions. Each request included the system prompt, question, source metadata and 10 retrieved policy chunks. Cached-input pricing has not been applied.

| Question | Embedding tokens | Chat input tokens | Chat completion tokens | Total tokens | Total cost |
|---|---:|---:|---:|---:|---:|
| Do I need to list my £3,000 violin separately? | 13 | 1,050 | 120 total<br>0 reasoning<br>120 visible | 1,183 | **£0.000380947** |
| If my home became unsafe to live in after a fire, how much would each level of cover pay for somewhere else for my family and pets to stay? | 31 | 1,783 | 487 total<br>0 reasoning<br>487 visible | 2,301 | **£0.001074729** |
| I have garden furniture worth £2,000. Will any policies cover it being left in the garden? | 21 | 1,674 | 569 total<br>0 reasoning<br>569 visible | 2,264 | **£0.001177649** |
| If someone broke into my detached garage and stole my tools, which levels of home insurance would cover them, and how much could I claim? | 28 | 2,891 | 397 total<br>0 reasoning<br>397 visible | 3,316 | **£0.001149292** |
| I have the Gold policy. Is my phone covered if I take it on holiday? | 17 | 1,005 | 229 total<br>0 reasoning<br>229 visible | 1,251 | **£0.000537063** |
| **Total** | **110** | **8,403** | **1,802 total**<br>**0 reasoning**<br>**1,802 visible** | **10,315** | **£0.004319680** |

Across the five measured questions:

- Total tokens processed: **10,315**
- Total embedding cost: **£0.000002090**
- Total GPT-5-mini input cost: **£0.001596570**
- Total GPT-5-mini output cost: **£0.002721020**
- Total cost: **£0.004319680**
- Average cost per question: **£0.000863936**

The five questions cost approximately **£0.00432 in total**, equivalent to approximately **0.432 pence**. The average cost was approximately **£0.000864 per question**, equivalent to approximately **0.086 pence**.

| Number of questions | Projected cost at measured average |
|---:|---:|
| 100 | £0.09 |
| 1,000 | £0.86 |
| 10,000 | £8.64 |
| 100,000 | £86.39 |

The calculations use:

```text
Embedding cost =
    embedding tokens ÷ 1,000 × £0.000019

Chat input cost =
    chat input tokens ÷ 1,000,000 × £0.19

Output cost =
    completion tokens ÷ 1,000,000 × £1.51

Total request cost =
    embedding cost + chat input cost + output cost
```

## Further enhancements

### Decompose complex questions into focused retrieval queries

Some user questions contain several topics or conditions and may retrieve better results when they are broken down into smaller, focused queries.

A future enhancement could add a query-decomposition stage before vector retrieval. The application would:

1. Send the original question to the chat model.
2. Ask the model to generate two to four focused retrieval queries.
3. Create an embedding for each query.
4. Add a VectorizedQuery object for each embedding to vector_queries.
5. Let Azure AI Search combine the ranked result lists.
6. Deduplicate the returned chunks.
7. Pass the highest-ranked chunks and the original question to the answer generator.

Azure AI Search supports multiple vector queries in a single search request. It combines their ranked result lists using Reciprocal Rank Fusion (RRF), so a chunk that ranks well for several focused queries can rise in the final ranking.

This approach can improve retrieval for multi-part questions, but it should be evaluated against the existing retrieval question set. Query decomposition adds an extra chat-model request and may introduce unnecessary or misleading subqueries for simple questions, so the application could apply it only to questions identified as complex.

