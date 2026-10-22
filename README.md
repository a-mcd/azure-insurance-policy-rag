# Home Insurance RAG (Retrieval → Augmentation → Generation)

An end-to-end retrieval-augmented generation (RAG) pipeline that answers questions about home insurance policy documents using Azure OpenAI and Azure AI Search.

The pipeline:

1. extracts structured content from the policy PDFs
2. divides the extracted content into retrieval-friendly chunks
3. creates a vector embedding for each chunk
4. indexes the chunks in Azure AI Search
5. retrieves the chunks most relevant to a question
6. generates a grounded answer with numbered source citations.

## Contents

- [How it works](#how-it-works)
- [Supported documents](#supported-documents)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Set up the local environment](#set-up-the-local-environment)
- [Configuration](#configuration)
- [Script output and logging](#script-output-and-logging)
- [1. Prepare the data](#1-prepare-the-data)
- [2. Set up Azure and index the chunks](#2-set-up-azure-and-index-the-chunks)
- [3. Search and evaluate retrieval](#3-search-and-evaluate-retrieval)
- [4. Generate grounded answers](#4-generate-grounded-answers)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Costs](#costs)
- [Possible further enhancements](#possible-further-enhancements)


## How it works

- **Retrieval:** vector search (the default), keyword search or hybrid search finds the policy chunks most relevant to the question.
- **Augmentation:** the retrieved chunks are added to the language-model prompt, each labelled with its document, section and subsection headings and page numbers.
- **Generation:** the model answers using only the supplied chunks and cites them by number.


## Supported documents

- **Insurance Product Information (IP) documents:** concise product summaries covering the main benefits, exclusions, restrictions and customer obligations.
- **Home insurance handbook (HH) documents:** detailed policy wording, cover information, conditions and exclusions.


## Project structure

```text
azure-insurance-policy-rag/
├── .env.example                      # Template for the Azure settings in .env
├── pyproject.toml                    # Package metadata, pinned dependencies and ruff settings
├── requirements.txt                  # Installs the project in editable mode
├── retrieval_questions.json          # Labelled questions for evaluating retrieval
│
├── data/
│   ├── raw/                          # Source PDF documents
│   │   ├── Gold-IP-HO-2-012.pdf
│   │   ├── HH-010-018-Guide-to-your-Home-Insurance-Cover.pdf
│   │   ├── HomeEmergency-IP-HE-1-008.pdf
│   │   ├── HomeEmergencyExtra-IP-HEE-1-005.pdf
│   │   ├── HomeLegal-IP-FL-1-008.pdf
│   │   ├── Platinum-IP-HO-3-010.pdf
│   │   └── Standard-IP-HO-1-012.pdf
│   └── processed/                    # Extracted JSON and chunk JSONL files
│
├── outputs/
│   ├── answers/                      # Answers saved by the answer scripts
│   └── retrieved_chunks/             # Retrieval evaluation results
│
├── src/
│   └── insurance_rag/                # Shared code imported by the scripts
│       ├── answering.py              # Answer generation, source lists and saving answers
│       ├── clients.py                # Azure OpenAI and Azure AI Search clients
│       ├── config.py                 # Paths, index constants and .env settings
│       ├── files.py                  # File helpers shared by the PDF extractors
│       ├── logging_setup.py          # Logging and --verbose/--quiet options
│       └── retrieval.py              # Query embedding and keyword, vector or hybrid search
│
└── scripts/
    ├── prepare_data/
    │   ├── extract_ip_pdfs.py        # Extracts the IP PDFs into structured JSON
    │   ├── extract_hh_pdfs.py        # Extracts the HH PDF into structured JSON
    │   ├── hh_contents_groups.json   # HH booklet contents structure used by extract_hh_pdfs.py
    │   └── chunk_documents.py        # Creates retrieval chunks
    │
    ├── setup_env/
    │   ├── setup_azure_openai.sh     # Creates and configures the Azure resources
    │   └── validate_openai_service.py  # Checks the embedding deployment works
    │
    ├── chunks/
    │   ├── index_document_chunks.py  # Embeds the chunks and uploads them to the index
    │   ├── search_document_chunks.py # Runs ad hoc retrieval queries
    │   └── evaluate_retrieval.py     # Measures retrieval against the labelled questions
    │
    └── generate_answers/
        ├── generate_answer.py        # Answers one question with citations
        └── run_questions.py          # Answers every question in a JSON file
```


## Requirements

- Python 3.10 or later
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli), signed in with `az login`
- An Azure subscription
- The source PDFs in `data/raw` (included in this repository)


## Set up the local environment

Run all commands from the project root.

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

2. Activate it: `source .venv/bin/activate` on macOS or Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. On Windows, use `python` wherever this README says `python3`.

3. Install the project. Choose one:

   - To run the pipeline:

     ```bash
     python3 -m pip install -r requirements.txt
     ```

   - To change the code, which also installs the development tools such as ruff (see [Development](#development)):

     ```bash
     python3 -m pip install -e ".[dev]"
     ```

   Both install the pinned dependencies from `pyproject.toml` and the `insurance_rag` package in `src/` in editable mode, so the scripts can import it and code changes take effect without reinstalling. Rerun the same command after pulling changes that add dependencies. To add a dependency, list it in `pyproject.toml`, not `requirements.txt`: runtime dependencies under `dependencies`, and development tools under `dev`.

4. Sign in to Azure:

   ```bash
   az login
   ```


## Configuration

The scripts read their Azure settings from a `.env` file in the project root. The [setup script](#2-set-up-azure-and-index-the-chunks) creates it for you; you can also copy `.env.example` and fill it in by hand. Values already set in your shell take precedence over `.env`.

| Variable | Required | Description |
| --- | --- | --- |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI resource endpoint. |
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI key. |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Yes | Embedding model deployment name. |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | For answers | Chat model deployment name. |
| `AZURE_SEARCH_ENDPOINT` | Yes | Azure AI Search service endpoint. |
| `AZURE_SEARCH_INDEX_NAME` | Yes | Name of the search index. |
| `AZURE_SEARCH_ADMIN_KEY` | For indexing | Admin key, needed to create the index and upload chunks. Also used for searching when no query key is set. |
| `AZURE_SEARCH_QUERY_KEY` | No | Read-only key. Searching and answering use it in preference to the admin key. |
| `AZURE_OPENAI_REASONING_EFFORT` | No | Reasoning effort for the chat model (default `minimal`). Set it to an empty value for models without reasoning, such as GPT-4o. See [Generate grounded answers](#4-generate-grounded-answers). |
| `AZURE_OPENAI_MAX_COMPLETION_TOKENS` | No | Maximum tokens the chat model may generate (default `2000`). |

Keep `.env` out of version control; it is listed in `.gitignore`.


## Script output and logging

Each script prints the results you asked for (answers, search results, evaluation tables and summaries) to standard output. Progress messages, such as `Created embeddings: 64/484`, and any warnings or errors go to standard error. This means you can save or pipe the results without status messages mixed in, for example:

```bash
python3 scripts/chunks/search_document_chunks.py "storm damage" --json > results.json
```

To capture everything in one file, add `2>&1`.

All scripts except `validate_openai_service.py` accept these options:

| Option | Effect |
| --- | --- |
| `-v`, `--verbose` | Also show debug messages, such as the chunk IDs retrieved for an answer. |
| `-q`, `--quiet` | Hide progress messages and show only warnings and errors. |


## 1. Prepare the data

The processed JSON and chunk files are already included in `data/processed`, so you only need this step if you change the PDFs, the extractors or the chunking.

### Extract the PDF documents

Extract the Insurance Product Information documents:

```bash
python3 scripts/prepare_data/extract_ip_pdfs.py \
  --input-dir data/raw \
  --output-file data/processed/ip_documents.json
```

Extract the home insurance handbook:

```bash
python3 scripts/prepare_data/extract_hh_pdfs.py \
  --input-dir data/raw \
  --output-file data/processed/hh_documents.json
```

The handbook's contents structure (its sections, subsections and their start pages) is read from `scripts/prepare_data/hh_contents_groups.json`. Update that file if you use a different edition of the booklet.

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
- no content is missing, duplicated or incorrectly merged.

If you only changed code, `git diff data/processed` is a quick way to confirm the output is unchanged. An LLM can help identify potential discrepancies, but verify every reported difference against the source PDF before changing the extraction logic.

### Create document chunks

Chunk the Insurance Product Information documents:

```bash
python3 scripts/prepare_data/chunk_documents.py \
  data/processed/ip_documents.json \
  --output data/processed/ip_document_chunks.jsonl
```

Chunk the home insurance handbook:

```bash
python3 scripts/prepare_data/chunk_documents.py \
  data/processed/hh_documents.json \
  --output data/processed/hh_document_chunks.jsonl
```

| Parameter | Required? | Default | Description |
| --- | --- | --- | --- |
| `input` | Yes | — | Extracted JSON file. |
| `--output`, `-o` | Yes | — | JSONL file to create. |
| `--max-tokens` | No | `500` | Target maximum tokens per chunk, including its metadata header. A single table row, or a level-four block with no level-five subheadings to split at, is kept whole even when it is longer. |
| `--encoding` | No | `cl100k_base` | tiktoken encoding used to count tokens. |

> [!IMPORTANT]
> Each chunk ID includes a hash of the chunk's text, so changing the extracted text, `--max-tokens`, the encoding or the chunking code changes the chunk IDs. If that happens, update the `expected_chunk_ids` in `retrieval_questions.json` before evaluating retrieval, and re-index the chunks.


## 2. Set up Azure and index the chunks

### Create the Azure resources

The setup script:

- creates the Azure resource group
- creates an Azure OpenAI resource and deploys the embedding and chat models
- creates a Free-tier Azure AI Search service
- writes the Azure OpenAI and Azure AI Search settings to `.env`
- tests the Azure OpenAI embedding deployment
- checks that the processed chunk files exist
- creates the search index and uploads the document chunks.

Run it with a unique prefix for the resource names, for example:

```bash
./scripts/setup_env/setup_azure_openai.sh --prefix aug17
```

| Option | Default | Description |
| --- | --- | --- |
| `--prefix NAME` | — (required) | Prefix used to name the Azure resources. |
| `--location LOCATION` | `norwayeast` | Azure region for the resources. |
| `--vector-algorithm ALG` | `hnsw` | Vector index algorithm: `hnsw` or `eknn`. |
| `--chat-model NAME` | `gpt-5-mini` | Chat model to deploy. |
| `--chat-model-version V` | `2025-08-07` | Chat model version. |
| `--chat-deployment NAME` | `gpt-5-mini` | Chat deployment name. |
| `--capacity NUMBER` | `10` | Embedding deployment capacity, in thousands of tokens per minute. |
| `--chat-capacity NUMBER` | `10` | Chat deployment capacity, in thousands of tokens per minute. |
| `--env-file PATH` | `.env` | Environment file to update. |

HNSW is suitable as the production default because it scales efficiently as the index grows. Exhaustive KNN (`eknn`) is useful as an exact baseline when evaluating whether approximate search reduces retrieval quality.

The setup script uses `scripts/setup_env/validate_openai_service.py` and `scripts/chunks/index_document_chunks.py`, and checks first that the project has been installed.

### Update the indexed chunks

After changing the chunks or the vector algorithm, generate new embeddings, delete and recreate the index, and upload the latest chunks:

```bash
python3 scripts/chunks/index_document_chunks.py --recreate-index
```

| Parameter | Required? | Default | Description |
| --- | --- | --- | --- |
| `inputs` | No | Both files in `data/processed` | Chunk JSONL files to index. |
| `--vector-algorithm` | No | `hnsw` | `hnsw` or `eknn`. |
| `--recreate-index` | No | Disabled | Delete and recreate the index first. This removes all existing index data. |
| `--validate-only` | No | Disabled | Check the chunk files without contacting Azure. |
| `--embedding-batch-size` | No | `16` | Chunks sent in each embedding request. |
| `--upload-batch-size` | No | `100` | Chunks sent in each Azure AI Search upload. |


## 3. Search and evaluate retrieval

### search_document_chunks.py

Use `search_document_chunks.py` to inspect the chunks retrieved for an ad hoc question, for example to:

- compare the vector, keyword and hybrid retrieval modes
- decide whether the chunks need to be changed
- compare the HNSW and exhaustive KNN vector algorithms.

```bash
python3 scripts/chunks/search_document_chunks.py \
  "QUESTION" \
  [--mode vector|hybrid|keyword] \
  [--top NUMBER] \
  [--document-code CODE] \
  [--document-type TYPE] \
  [--json]
```

| Parameter | Required? | Default | Description |
| --- | --- | --- | --- |
| `question` | Yes | — | Question or search text. It must not be empty. |
| `--mode` | No | `vector` | Retrieval method: <br />- `vector`: retrieves chunks by semantic similarity to the question. This is the default.<br />- `keyword`: retrieves chunks using matching words and phrases.<br />- `hybrid`: combines vector similarity with keyword matching. |
| `--top` | No | `5` | Number of chunks to return. Accepted range: 1–50. |
| `--document-code` | No | All documents | Restricts results to an exact document code, such as `IP-HO-2-012`. |
| `--document-type` | No | All document types | Restricts results to a document type, such as `insurance_product_information` or `home_insurance_policy_booklet`. |
| `--json` | No | Disabled | Prints machine-readable JSON instead of formatted text. |

Vector search is the default because it outperformed both other modes on every metric when evaluated against the 21 labelled questions in `retrieval_questions.json` with `--top 10`:

| Mode | Mean reciprocal rank | Hit@1 | Recall@10 |
| --- | ---: | ---: | ---: |
| `vector` | 0.81 | 0.71 | 0.96 |
| `hybrid` | 0.67 | 0.52 | 0.71 |
| `keyword` | 0.45 | 0.33 | 0.55 |

The detailed results are in `outputs/retrieved_chunks/retrieval_results.json`. `generate_answer.py` uses vector search for the same reason. Rerun the evaluation, and update these figures, after changing the chunks or the index.

### evaluate_retrieval.py

Evaluates retrieval against the labelled questions in `retrieval_questions.json`, each of which lists the chunk IDs that should be retrieved. It reports hit rate, recall and mean reciprocal rank for each mode and saves the detailed results. It has no required parameters.

```bash
python3 scripts/chunks/evaluate_retrieval.py \
  [--input PATH] \
  [--output PATH] \
  [--modes MODE [MODE ...]] \
  [--top NUMBER] \
  [--question-id ID]
```

| Parameter | Required? | Default | Description |
| --- | --- | --- | --- |
| `--input` | No | `retrieval_questions.json` | Evaluation questions JSON file. |
| `--output` | No | `outputs/retrieved_chunks/retrieval_results.json` | Destination for detailed JSON results. |
| `--modes` | No | `keyword vector hybrid` | One or more retrieval modes. Accepted values: `keyword`, `vector` and `hybrid`. |
| `--top` | No | `5` | Number of chunks retrieved for each question. Accepted range: 1–50. |
| `--question-id` | No | All questions | Evaluates only the specified question, such as `Q006`. |

The default paths are relative to the project root, so the script works from any directory. Paths you pass are relative to the current directory.

Compare selected modes:

```bash
python3 scripts/chunks/evaluate_retrieval.py \
  --modes vector hybrid \
  --top 10
```

Evaluate one question only:

```bash
python3 scripts/chunks/evaluate_retrieval.py \
  --question-id Q006 \
  --modes vector \
  --top 10
```


## 4. Generate grounded answers

### generate_answer.py

Retrieves policy chunks using vector search and generates a grounded answer with numbered citations.

```bash
python3 scripts/generate_answers/generate_answer.py \
  "QUESTION" \
  [--top NUMBER] \
  [--document-code CODE] \
  [--show-scores]
```

| Parameter | Required? | Default | Description |
| --- | --- | --- | --- |
| `question` | Yes | — | Insurance question to answer. It must not be empty. |
| `--top` | No | `10` | Number of chunks supplied to the chat model. Accepted range: 1–50. |
| `--document-code` | No | All documents | Restricts retrieval to an exact document code, such as `IP-HO-2-012`. |
| `--show-scores` | No | Disabled | Includes Azure AI Search scores in the printed source list. |

Only `question` is required:

```bash
python3 scripts/generate_answers/generate_answer.py \
  "Do I need to list my £3,000 violin separately?"
```

Example using optional parameters:

```bash
python3 scripts/generate_answers/generate_answer.py \
  "Are fences covered for storm damage?" \
  --document-code IP-HO-2-012 \
  --top 10 \
  --show-scores
```

The script prints the answer, the sources it cited, the retrieved sources it did not cite, and the token usage. Each source is labelled with its document title and code, section, subsection and semantic headings, chunk ID and page numbers, for example:

```text
[1] Gold Home Insurance Policy | IP-HO-2-012 | What is insured? | Buildings (Main features, if selected) | chunk: ip-ho-2-012-0004-42f299b9dace | PDF page(s): 1 | printed page(s): 1
```

The answer and source lists are also saved as Markdown in `outputs/answers/`, named after the question. If a file of that name exists, a number is added rather than overwriting it.

The chat request options can be set in `.env` (see `.env.example`):

| Variable | Default | Description |
| --- | --- | --- |
| `AZURE_OPENAI_REASONING_EFFORT` | `minimal` | Reasoning effort for GPT-5 reasoning models. Set it to an empty value (`AZURE_OPENAI_REASONING_EFFORT=`) for models without reasoning, such as GPT-4o, which reject this parameter. |
| `AZURE_OPENAI_MAX_COMPLETION_TOKENS` | `2000` | Maximum tokens the model may generate. For reasoning models this includes hidden reasoning tokens. |

### run_questions.py

Answers every question in a JSON file, saving and printing each answer in turn. A failed question is reported and the remaining questions still run.

```bash
python3 scripts/generate_answers/run_questions.py retrieval_questions.json
```

| Parameter | Required? | Default | Description |
| --- | --- | --- | --- |
| `questions_file` | Yes | — | JSON file containing a list of questions, or an object with a `questions` list. Each question can be a string or an object with `question` and an optional `question_id`. |
| `--question-id` | No | All questions | Answer only this question ID. Repeat it to answer several. |
| `--stop-on-error` | No | Disabled | Stop at the first question that fails. |

The script exits with status 1 if any question failed.


## Development

If you installed with `requirements.txt`, install the development tools as well by running this from the project root:

```bash
python3 -m pip install -e ".[dev]"
```

This installs everything `requirements.txt` does, plus the tools in the `dev` group of `pyproject.toml`. Running it after `requirements.txt` is safe; it only adds what is missing. The quotes stop shells such as zsh, the default on macOS, from treating the square brackets as a file pattern.

Code style is checked with [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`. Run it from the project root:

```bash
ruff check .          # lint
ruff format .         # format
```

ruff is pinned to the version the code was last formatted with, so everyone gets the same formatting. When upgrading it, update the version in `pyproject.toml` and run `ruff format .` in the same commit.

Shared code belongs in the `insurance_rag` package in `src/`; the scripts in `scripts/` are command-line entry points that import from it.


## Troubleshooting

| Problem | Fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'insurance_rag'` | Activate the virtual environment and run `python3 -m pip install -r requirements.txt` from the project root. |
| `ModuleNotFoundError: No module named 'tiktoken'` (or another dependency) | As above. `chunk_documents.py` requires tiktoken because chunk IDs depend on the exact tokenizer. |
| `ERROR: Missing environment variables: …` | Add the named variables to `.env`; see [Configuration](#configuration). |
| The chat request is rejected because of `reasoning_effort` | The chat model does not support reasoning. Set `AZURE_OPENAI_REASONING_EFFORT=` (empty) in `.env`. |
| Retrieval evaluation finds none of the expected chunks | The chunk IDs probably changed after re-chunking; see [Create document chunks](#create-document-chunks). |


## Costs

Prices and measurements below were recorded at the time of testing. Check current Azure pricing before relying on them.

### Deploy text-embedding-3-small

£0.00 per month while unused.

### Query and index chunks

text-embedding-3-small costs £0.000019 per 1,000 tokens.

Cost to embed all chunks by running the index script:

| Chunk file | Number of chunks | Total tokens | Embedding cost |
|---|---:|---:|---:|
| `ip_document_chunks.jsonl` | 228 | 15,182 | £0.000288458 |
| `hh_document_chunks.jsonl` | 256 | 49,067 | £0.000932273 |
| **Total** | **484** | **64,249** | **£0.001220731** |

For example, 15,182 ÷ 1,000 × £0.000019 = £0.000288458.

Cost to embed a question when retrieving chunks:

| Question | Embedding tokens | Cost |
|---|---:|---:|
| Do I need to list my £3,000 violin separately? | 13 | **£0.000000247** |
| If my home became unsafe to live in after a fire, how much would each level of cover pay for somewhere else for my family and pets to stay? | 31 | **£0.000000589** |
| I have garden furniture worth £2,000. Will any policies cover it being left in the garden? | 21 | **£0.000000399** |
| If someone broke into my detached garage and stole my tools, which levels of home insurance would cover them, and how much could I claim? | 28 | **£0.000000532** |
| I have the Gold policy. Is my phone covered if I take it on holiday? | 17 | **£0.000000323** |
| **Total** | **110** | **£0.000002090** |

For example, 13 ÷ 1,000 × £0.000019 = £0.000000247.

### Store indexed chunks

| Azure AI Search Free-tier service | Limit | Used |
| --- | ---: | ---: |
| Storage | 50 MB | 9.91 MB |
| Vector index quota usage | 25 MB | 8.58 MB |

### Agentic retrieval

The first 50 million tokens per month are free. This is not yet used, but could be if questions were broken down into smaller queries (see [Possible further enhancements](#possible-further-enhancements)).

### GPT-5-mini question costs

GPT-5-mini Global pricing:

- £0 while deployed but unused
- Input: £0.19 per 1 million tokens
- Cached input: £0.02 per 1 million tokens
- Output and reasoning: £1.51 per 1 million tokens

These measurements used a maximum of 2,000 completion tokens per request and minimal reasoning effort (the defaults; see [Configuration](#configuration)).

The following results show only the GPT-5-mini cost. Each request included the system prompt, question, source metadata and 10 retrieved policy chunks. Question-embedding costs are excluded from this section, and cached-input pricing has not been applied. Source labels have since gained subsection headings, so current prompts are slightly longer.

| Question | Input tokens | Completion tokens | Total GPT tokens | Input cost | Output cost | GPT-5-mini cost |
|---|---:|---:|---:|---:|---:|---:|
| Do I need to list my £3,000 violin separately? | 1,530 | 132 total<br>0 reasoning<br>132 visible | 1,662 | £0.000290700 | £0.000199320 | **£0.000490020** |
| If my home became unsafe to live in after a fire, how much would each level of cover pay for somewhere else for my family and pets to stay? | 2,263 | 217 total<br>0 reasoning<br>217 visible | 2,480 | £0.000429970 | £0.000327670 | **£0.000757640** |
| I have garden furniture worth £2,000. Will any policies cover it being left in the garden? | 2,154 | 410 total<br>0 reasoning<br>410 visible | 2,564 | £0.000409260 | £0.000619100 | **£0.001028360** |
| If someone broke into my detached garage and stole my tools, which levels of home insurance would cover them, and how much could I claim? | 3,371 | 64 total<br>0 reasoning<br>64 visible | 3,435 | £0.000640490 | £0.000096640 | **£0.000737130** |
| I have the Gold policy. Is my phone covered if I take it on holiday? | 1,485 | 172 total<br>0 reasoning<br>172 visible | 1,657 | £0.000282150 | £0.000259720 | **£0.000541870** |
| **Total** | **10,803** | **995 total**<br>**0 reasoning**<br>**995 visible** | **11,798** | **£0.002052570** | **£0.001502450** | **£0.003555020** |

### End-to-end question costs

Question
→ create a question embedding with text-embedding-3-small
→ retrieve 10 chunks from Azure AI Search
→ generate an answer with GPT-5-mini

| Question | Embedding cost | GPT-5-mini cost | End-to-end cost |
|---|---:|---:|---:|
| Do I need to list my £3,000 violin separately? | £0.000000247 | £0.000490020 | **£0.000490267** |
| If my home became unsafe to live in after a fire, how much would each level of cover pay for somewhere else for my family and pets to stay? | £0.000000589 | £0.000757640 | **£0.000758229** |
| I have garden furniture worth £2,000. Will any policies cover it being left in the garden? | £0.000000399 | £0.001028360 | **£0.001028759** |
| If someone broke into my detached garage and stole my tools, which levels of home insurance would cover them, and how much could I claim? | £0.000000532 | £0.000737130 | **£0.000737662** |
| I have the Gold policy. Is my phone covered if I take it on holiday? | £0.000000323 | £0.000541870 | **£0.000542193** |
| **Total** | **£0.000002090** | **£0.003555020** | **£0.003557110** |
| **Average per question** | **£0.000000418** | **£0.000711004** | **£0.000711422** |


## Possible further enhancements

The key weakness of this RAG pipeline is retrieval. Some required chunks are not returned, and some irrelevant chunks are. This is largely due to vocabulary mismatch between the questions and the policy wording. The improvements below could be tested against the labelled question set.

### Decompose complex questions into focused retrieval queries using Azure agentic retrieval

Some questions contain several topics or conditions and may retrieve better results when they are broken down into smaller, focused queries.

A query-decomposition stage could be added before vector retrieval. The application would:

1. Send the original question to the chat model.
2. Ask the model to generate two to four focused retrieval queries.
3. Create an embedding for each query.
4. Add a `VectorizedQuery` object for each embedding to `vector_queries`.
5. Let Azure AI Search combine the ranked result lists.
6. Deduplicate the returned chunks.
7. Pass the highest-ranked chunks and the original question to the answer generator.

Azure AI Search supports multiple vector queries in a single search request. It combines their ranked result lists using Reciprocal Rank Fusion (RRF), so a chunk that ranks well for several focused queries can rise in the final ranking.

This approach can improve retrieval for multi-part questions, but it should be evaluated against the existing retrieval question set. Query decomposition adds an extra chat-model request and may introduce unnecessary or misleading subqueries for simple questions, so the application could apply it only to questions identified as complex.

### Expand policy terminology before retrieval

Vector search matches chunks by semantic similarity, not by explicit insurance-domain reasoning. The embedding model will probably recognise that:

- phone is related to mobile phone
- holiday is related to travel
- cover is related to insurance.

However, it may not reliably know or prioritise that, in these particular policy documents:

- a mobile phone is defined as a high-risk item
- holiday cover is described as cover away from home
- high-risk items are covered through personal possessions or specified-item cover.

These relationships may be established across multiple chunks. Standard vector search compares the question with each chunk independently and does not join facts across chunks. It therefore cannot reason:

> Another chunk defines a phone as a high-risk item, therefore I should also retrieve chunks about personal possessions, specified items and cover away from home.

A knowledge graph or GraphRAG system could represent relationships between concepts such as mobile phone, high-risk item, personal possessions and cover away from home. The system could identify related policy concepts before or during retrieval and use them to retrieve additional relevant chunks, for example:

```text
phone → mobile phone, high-risk item, personal possessions, specified item
holiday → away from home, worldwide, outside the UK
```
