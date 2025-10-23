from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

UserRole = Literal["driver", "manager", "owner"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str
    user_role: UserRole
    chat_history: List[ChatMessage] = Field(default_factory=list)


class SourceDocument(BaseModel):
    source: Optional[str] = None
    content: str
    score: Optional[float] = None


class FeedbackSnippet(BaseModel):
    id: UUID
    query: str
    response: str
    updated_response: str
    user_role: UserRole
    score: float
    weight: int
    created_at: datetime


class ChatResponse(BaseModel):
    answer: str
    used_documents: List[SourceDocument] = Field(default_factory=list)
    applied_feedback: List[FeedbackSnippet] = Field(default_factory=list)
    prompt: Optional[str] = None


class FeedbackRequest(BaseModel):
    query: str
    response: str
    user_role: UserRole
    updated_response: str


class FeedbackEntry(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    query: str
    response: str
    updated_response: str
    user_role: UserRole
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeedbackResponse(BaseModel):
    success: bool
    entry: FeedbackEntry


class UploadResponse(BaseModel):
    success: bool
    chunks_added: int
    files: List[str]
