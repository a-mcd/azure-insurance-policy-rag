# Home Insurance RAG Document Pipeline

This project prepares home insurance documents for use in a retrieval-augmented generation (RAG) system. It extracts structured content from PDF documents and then divides that content into retrieval-friendly chunks.

The current pipeline supports two document groups:

- Insurance Product Information (IP) documents — concise product summaries describing the main cover, exclusions, restrictions, and customer obligations.
- Home insurance handbook (HH) documents — more detailed policy and cover documentation.


## Project structure

```text
.
├── data/
│   ├── raw/                         # Source PDF documents
│   └── processed/                   # Extracted documents and chunks
└── src/
    ├── extract_ip_pdfs.py           # Extracts IP PDFs into structured JSON
    ├── extract_hh_pdfs.py           # Extracts HH PDFs into structured JSON
    ├── chunk_ip_documents.py        # Chunks extracted IP documents
    └── chunk_hh_documents.py        # Chunks extracted HH documents
```



## Requirements

- Python 3: The Python packages required by the extraction and chunking scripts
- Source PDFs placed in data/raw
- Install the project dependencies before running the pipeline. If the repository contains a requirements.txt file, run: `python3 -m pip install -r requirements.txt`


## Steps

1) Extract Insurance Product Information documents
The extraction scripts read the PDF files from `data/raw` and convert them into structured JSON containing document metadata, sections, headings, paragraphs, bullet points, tables, and page references.

> [!NOTE]
> These scripts were developed and tested on macOS. PDF text positioning and line boundaries may differ across operating systems or dependency versions, which can affect how successfully the documents are parsed.
>
> For consistent and reproducible extraction, consider running the scripts in a Docker container with fixed Python and package versions. The parsing boundaries can then be configured and tested against that stable environment.


```bash
python3 src/extract_ip_pdfs.py \
  --input-dir data/raw \
  --output-file data/processed/ip_documents.json
```
```bash
python3 src/extract_hh_pdfs.py \
  --input-dir data/raw \
  --output-file data/processed/hh_documents.json
```

After running the scripts, compare the generated JSON files with their source PDFs. Check that:

- headings and paragraphs are correctly identified;
- bullet points remain grouped with their parent content;
- tables contain the correct rows and columns;
- words split across PDF lines are reconstructed correctly;
- page references are accurate; and
- no content is missing, duplicated, or incorrectly merged.
- spellings are correct

An LLM can help identify discrepancies between the PDFs and the generated JSON. However, any reported differences should be manually verified against the source PDFs before changing the extraction logic.

2) Chunk the json files
```bash
python3 src/chunk_ip_documents.py \
  data/processed/ip_documents.json \
  --output data/processed/ip_document_chunks.jsonl
```
```bash
python3 src/chunk_hh_documents.py \
  data/processed/hh_documents.json \
  --output data/processed/hh_document_chunks.jsonl
```