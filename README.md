# RAG Feedback Loop Demo

This project implements a Retrieval-Augmented Generation (RAG) web application inspired by the feedback workflow described in the MachineLearningPlus article on improving RAG systems with user interactions. Users can upload documents, ask questions against the vectorized knowledge base, review the AI-generated answer, and submit corrected responses. The corrections are prioritized by user role (driver, manager, owner) and fed back into the retrieval pipeline so that authoritative feedback influences future answers.

## Project structure

```
.
├── backend/              # FastAPI application powering the RAG pipeline
│   ├── app/
│   │   ├── main.py       # API routes for uploads, chat, and feedback
│   │   ├── config.py     # Paths and role weight configuration
│   │   ├── models.py     # Pydantic schemas
│   │   └── services/     # Document loading, FAISS stores, prompt builder
│   ├── requirements.txt  # Python dependencies
│   └── .env.example      # Environment variable template
└── frontend/             # Vite + React single-page app
    ├── index.html
    ├── package.json
    └── src/
        ├── App.jsx       # Main UI: upload, chat, feedback editor
        ├── main.jsx      # React entry point
        └── styles.css    # Styling
```

## Backend setup

1. Create and activate a Python environment (`python -m venv .venv && source .venv/bin/activate`).
2. Install dependencies:

   ```bash
   pip install -r backend/requirements.txt
   ```

3. Copy the environment file and add your OpenAI key/model:

   ```bash
   cp backend/.env.example backend/.env
   ```

4. Start the FastAPI server:

   ```bash
   uvicorn app.main:app --reload --app-dir backend --host 0.0.0.0 --port 8000
   ```

   The server exposes:

   - `POST /upload` to ingest documents (txt, md, json, pdf)
   - `POST /chat` to run the RAG pipeline with weighted feedback
   - `POST /feedback` to persist user-updated answers and update the feedback FAISS index

The backend stores uploaded files, FAISS indices, and feedback logs in `backend/data/`.

## Frontend setup

1. Install dependencies:

   ```bash
   cd frontend
   npm install
   ```

2. Start the Vite dev server:

   ```bash
   npm run dev
   ```

   By default the app looks for the backend at `http://localhost:8000`. Override by creating `.env` with `VITE_API_BASE_URL` if needed.

3. Build for production (optional):

   ```bash
   npm run build
   ```

## Feedback weighting logic

User feedback is stored with the fields `query`, `response`, `updated_response`, and `user_role`. When a new chat request arrives:

1. Relevant document chunks are retrieved from the FAISS vector store built from uploaded files.
2. The feedback FAISS index is queried with the new question to surface matching corrections.
3. Retrieved feedback snippets are scored using cosine similarity plus a role-based boost (`driver` < `manager` < `owner`).
4. The prompt sent to the OpenAI Responses API prioritizes higher-weight feedback over raw documents, ensuring subject-matter adjustments supersede base knowledge.

If no OpenAI API key is configured, the backend returns a simulated placeholder response so the workflow can be exercised end-to-end without external calls.

## Development notes

- Document ingestion currently supports text, Markdown, JSON, and PDF files. Extend `backend/app/services/documents.py` to add other loaders.
- Feedback is persisted both in a JSON log (`feedback_log.json`) and in a FAISS vector store, enabling quick reloading after restarts.
- Role weights and boosts are centralized in `backend/app/config.py` for easy tuning.

## Running tests

There are no automated tests bundled with this demo. You can validate the frontend build with `npm run build` and rely on manual testing for the FastAPI endpoints.
