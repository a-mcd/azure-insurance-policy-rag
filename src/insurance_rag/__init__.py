"""Shared code for the Azure insurance-policy RAG pipeline.

The command-line scripts under ``scripts/`` are thin entry points that build on
the modules in this package:

- ``config``: project paths, index constants and settings loaded from ``.env``.
- ``clients``: factories for the Azure OpenAI and Azure AI Search clients.
- ``retrieval``: query embedding and keyword, vector or hybrid search.
- ``answering``: grounded answer generation, source reporting and saving.
- ``files``: small file helpers shared by the PDF extractors.
- ``logging_setup``: logging and ``--verbose``/``--quiet`` options for scripts.

The package never prints: it returns values and formatted text, and logs
progress through ``logging``. The scripts decide what goes to stdout.
"""
