import os
from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from . import config
from .models import (
    ChatRequest,
    ChatResponse,
    FeedbackEntry,
    FeedbackRequest,
    FeedbackResponse,
    SourceDocument,
    UploadResponse,
)
from .services import documents, feedback as feedback_service, prompt_builder, vector_store
from .services.embeddings import get_embedding_model

config.ensure_directories()

app = FastAPI(title="RAG Feedback Loop API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


class Stores:
    def __init__(self) -> None:
        self.document_store = vector_store.load_document_store()
        self.feedback_repository = feedback_service.FeedbackRepository()


stores_dependency = Stores()


def get_stores() -> Stores:
    return stores_dependency


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    stores: Stores = Depends(get_stores),
) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    embeddings = get_embedding_model()
    document_store = stores.document_store
    added_chunks = 0
    processed_files: List[str] = []

    for file in files:
        data = await file.read()
        stored_path = documents.store_upload(file.filename, data)
        try:
            loaded_documents = documents.load_documents(stored_path)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        chunks = documents.chunk_documents(loaded_documents)
        added_chunks += len(chunks)
        processed_files.append(file.filename)
        if document_store is None:
            document_store = FAISS.from_documents(chunks, embeddings)
        else:
            document_store.add_documents(chunks)

    if document_store is None:
        raise HTTPException(status_code=500, detail="Unable to create document store")

    stores.document_store = document_store
    vector_store.save_document_store(document_store)

    return UploadResponse(success=True, chunks_added=added_chunks, files=processed_files)


def _build_documents_context(query: str, stores: Stores) -> List[SourceDocument]:
    if stores.document_store is None:
        return []
    results = stores.document_store.similarity_search_with_score(query, k=5)
    context: List[SourceDocument] = []
    for doc, score in results:
        context.append(
            SourceDocument(
                source=doc.metadata.get("source"),
                content=doc.page_content,
                score=float(score),
            )
        )
    return context


def _call_openai(prompt: str, model: str | None = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "[Simulated response] "
            "Provide the final answer by prioritizing the user feedback over the raw documents."
        )
    client = OpenAI(api_key=api_key)
    chosen_model = model or os.getenv("OPENAI_MODEL", config.DEFAULT_MODEL)
    response = client.responses.create(
        model=chosen_model,
        input=[{"role": "user", "content": prompt}],
    )
    if hasattr(response, "output_text"):
        return response.output_text
    return "".join(block.text for block in response.output if getattr(block, "type", "") == "output_text")


@app.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    stores: Stores = Depends(get_stores),
) -> ChatResponse:
    if not payload.query:
        raise HTTPException(status_code=400, detail="Query is required")

    documents_context = _build_documents_context(payload.query, stores)
    feedback_snippets = stores.feedback_repository.as_snippets(payload.query, limit=5)
    prompt = prompt_builder.build_prompt(
        query=payload.query,
        documents=documents_context,
        feedback_snippets=feedback_snippets,
        history=payload.chat_history,
        user_role=payload.user_role,
    )
    answer = _call_openai(prompt)
    return ChatResponse(
        answer=answer,
        used_documents=documents_context,
        applied_feedback=feedback_snippets,
        prompt=prompt,
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    payload: FeedbackRequest,
    stores: Stores = Depends(get_stores),
) -> FeedbackResponse:
    entry = FeedbackEntry(
        query=payload.query,
        response=payload.response,
        updated_response=payload.updated_response,
        user_role=payload.user_role,
    )
    stored = stores.feedback_repository.add(entry)
    return FeedbackResponse(success=True, entry=stored)
