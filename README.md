# ✨ LLM-Powered RAG Assistant for Retail Analytics

A Streamlit app that converts natural-language retail analytics questions into SQL (via an LLM), executes the SQL against Databricks, and returns the resulting table.

---

## Overview

This repository contains a small proof-of-concept that turns user queries like "Show me the top 10 product categories of the latest 1 year" into executable SQL using a retrieval-augmented LLM pipeline and runs the SQL on a Databricks SQL warehouse. The project indexes an Olist (Brazilian e‑commerce) data dictionary into a vector store so the model generates SQL only using the allowed tables/joins.

Key ideas:

* Retrieval of relevant table schemas and join edges from a vector store (Chroma) built from `tables_doc_LLM_friendly.docx`.
* Prompt assembly with strict system instructions so the LLM returns **SQL-only**.
* Automatic SQL repair loop (uses an LLM to fix runtime errors) and up to 3 attempts before failing.
* Streamlit UI for quick experimentation with natural language queries.

---

## Project structure

```
├── app.py                         # Streamlit front-end
├── vector_db.py                   # Build vector DB (Chroma) from the docx and set up retriever
├── retriever_executor.py          # Retrieval → LLM SQL generation → execute on Databricks
├── tables_doc_LLM_friendly.docx   # Data dictionary (indexed into the vector DB)
├── requirements.txt               # Python deps
└── README.md                      # This file
```

**File responsibilities (short):**

* `app.py` — Streamlit UI that accepts natural-language queries and displays the DataFrame returned by `retriever_executor.get_data_from_query()`.
* `vector_db.py` — Loads `tables_doc_LLM_friendly.docx`, splits parent/child docs and joins, creates Chroma collections and a `ParentDocumentRetriever` used at runtime.
* `retriever_executor.py` — Orchestrates retrieval, prompt construction, LLM calls (generate SQL), executes SQL on Databricks via `databricks-sql-connector`, and handles simple automatic repair of SQL using the LLM if errors occur.
* `tables_doc_LLM_friendly.docx` — The schema & relationships doc used as the knowledge source for RAG.
* `requirements.txt` — Tested package versions.

---

## Quick start

> **Before you start:** replace the dummy values of the API key, databricks access token, http path, and hostname with your own values or use environment variables.

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
.venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
```

2. Set environment variables (recommended). Example `.env`:

```
OPENAI_API_KEY=sk-...
DATABRICKS_HOST=dbc-XXXX.cloud.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/<your-http-path>
DATABRICKS_PAT=<your-databricks-personal-access-token>
```

The code expects `OPENAI_API_KEY` (used by `langchain-openai` / `openai` client) and the Databricks connection info (host/http_path/access token). By default the example code contains dummy values in code — **please remove them** and export the env vars instead.

3. Run the app locally:

```bash
streamlit run app.py
```

4. In the browser, enter natural language queries like:

* `Show me total sales by product category for last month.`
* `Who are my top customers?`

---

## Configuration details

* **OpenAI model choices:** `retriever_executor.generate_sql_from_query()` uses an LLM to produce SQL (defaults in code: `gpt-4o-mini` → repairs with `gpt-4o`). You can change model names via function args.
* **Vector store:** Chroma is used for embedding storage and retrieval. Embeddings are produced with `OpenAIEmbeddings`.
* **Databricks:** `retriever_executor.fetch_data()` uses the `databricks-sql-connector` to run generated SQL. Provide host/http_path/token via environment variables.

---

## Troubleshooting

* If the LLM invents table/column names: the prompt enforces using only allowed tables/columns. Ensure `vector_db.initialize_variables()` correctly builds the parent documents from `tables_doc_LLM_friendly.docx` and that the retriever returns relevant parents.
* Databricks connection errors: verify `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, and `DATABRICKS_PAT` are correct and have appropriate permissions.
* Large prompts / token limits: `retriever_executor._format_parents()` truncates context to control prompt size. Tune `k_schema`/`k_joins_per_side` or increase model `max_tokens` if necessary.

---

## Recommendations & next steps

* **Remove dummy keys/tokens** and use environment variables or a secret manager.
* Add unit tests for the prompt assembly and SQL post-processing.
* Add input sanitization and rate-limiting to the Streamlit UI for production readiness.
* Consider using a server-side orchestration (FastAPI) to avoid exposing internals in the frontend and to better manage secrets & concurrency.
* Add automatic integration tests that mock the Databricks connector and OpenAI responses.

---

## License

This project is provided under the MIT License — feel free to adapt it for your experiments.
