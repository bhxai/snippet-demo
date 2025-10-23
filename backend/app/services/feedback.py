import json
from dataclasses import dataclass
from typing import List

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from .. import config
from ..models import FeedbackEntry, FeedbackSnippet
from .embeddings import get_embedding_model
from . import vector_store


@dataclass
class RetrievedFeedback:
    entry: FeedbackEntry
    score: float
    weight: int


class FeedbackRepository:
    def __init__(self) -> None:
        config.ensure_directories()
        self.log_path = config.FEEDBACK_LOG_PATH
        self._entries: List[FeedbackEntry] = []
        self._store: FAISS | None = None
        self._load()

    @property
    def entries(self) -> List[FeedbackEntry]:
        return list(self._entries)

    def _load(self) -> None:
        if self.log_path.exists():
            data = json.loads(self.log_path.read_text(encoding="utf-8"))
            for raw in data:
                self._entries.append(FeedbackEntry(**raw))
        self._store = vector_store.load_feedback_store()

    def _persist(self) -> None:
        payload = [entry.model_dump() for entry in self._entries]
        self.log_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")

    def add(self, entry: FeedbackEntry) -> FeedbackEntry:
        self._entries.append(entry)
        self._persist()
        self._upsert_vector_entry(entry)
        return entry

    def _upsert_vector_entry(self, entry: FeedbackEntry) -> None:
        metadata = {
            "id": str(entry.id),
            "query": entry.query,
            "response": entry.response,
            "updated_response": entry.updated_response,
            "user_role": entry.user_role,
            "created_at": entry.created_at.isoformat(),
        }
        doc = Document(page_content=entry.updated_response, metadata=metadata)
        embeddings = get_embedding_model()
        if self._store is None:
            self._store = FAISS.from_documents([doc], embeddings)
        else:
            self._store.add_documents([doc])
        vector_store.save_feedback_store(self._store)

    def search(self, query: str, limit: int = 5) -> List[RetrievedFeedback]:
        if self._store is None:
            return []
        results = self._store.similarity_search_with_score(query, k=limit)
        retrieved: List[RetrievedFeedback] = []
        for doc, distance in results:
            metadata = doc.metadata
            entry_payload = {
                "query": metadata.get("query", ""),
                "response": metadata.get("response", ""),
                "updated_response": metadata.get("updated_response", doc.page_content),
                "user_role": metadata.get("user_role", "driver"),
            }
            if metadata.get("id"):
                entry_payload["id"] = metadata.get("id")
            if metadata.get("created_at"):
                entry_payload["created_at"] = metadata.get("created_at")
            entry = FeedbackEntry(**entry_payload)
            weight = config.ROLE_WEIGHTS.get(entry.user_role, 1)
            boost = config.ROLE_BOOST.get(entry.user_role, 0.0)
            similarity = 1.0 / (1.0 + distance)
            score = similarity + boost
            retrieved.append(RetrievedFeedback(entry=entry, score=score, weight=weight))
        retrieved.sort(key=lambda item: (item.weight, item.score), reverse=True)
        return retrieved

    def as_snippets(self, query: str, limit: int = 5) -> List[FeedbackSnippet]:
        snippets: List[FeedbackSnippet] = []
        for item in self.search(query, limit=limit):
            entry = item.entry
            snippets.append(
                FeedbackSnippet(
                    id=entry.id,
                    query=entry.query,
                    response=entry.response,
                    updated_response=entry.updated_response,
                    user_role=entry.user_role,
                    created_at=entry.created_at,
                    score=item.score,
                    weight=item.weight,
                )
            )
        return snippets


def format_feedback_context(snippets: List[FeedbackSnippet]) -> str:
    if not snippets:
        return "No user feedback matched the query."
    parts: List[str] = []
    for snippet in snippets:
        parts.append(
            "\n".join(
                [
                    f"Role: {snippet.user_role} (weight {snippet.weight})",
                    f"Original Query: {snippet.query}",
                    "User-updated response:",
                    snippet.updated_response,
                ]
            )
        )
    return "\n\n".join(parts)
