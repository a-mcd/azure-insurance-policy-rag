python3 src/extract_ip_pdfs.py \
  --input-dir data/raw \
  --output-file data/processed/ip_documents.json


python3 src/extract_hh_pdfs.py \
  --input-dir data/raw \
  --output-file data/processed/hh_documents.json


python3 src/chunk_ip_documents.py \
  data/processed/ip_documents.json \
  --output data/processed/ip_document_chunks.jsonl

python3 src/chunk_hh_documents.py \
  data/processed/hh_documents.json \
  --output data/processed/hh_document_chunks.jsonl