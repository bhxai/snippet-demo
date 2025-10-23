from typing import List

from ..models import ChatMessage, FeedbackSnippet, SourceDocument
from .. import config


SYSTEM_PROMPT = """You are an AI assistant helping a logistics company answer questions based on internal documents and user feedback.\nUser feedback captures corrections from subject matter experts and should be treated as authoritative.\nWhen multiple feedback entries conflict, prefer the one provided by the highest weighted role.\nRoles have the following priority from lowest to highest authority: driver, manager, owner."""


def format_chat_history(history: List[ChatMessage]) -> str:
    if not history:
        return ""
    formatted = []
    for message in history:
        prefix = "User" if message.role == "user" else "Assistant"
        formatted.append(f"{prefix}: {message.content}")
    return "\n".join(formatted)


def format_documents(documents: List[SourceDocument]) -> str:
    if not documents:
        return "No retrieved documents."
    parts: List[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.source or f"chunk-{index}"
        parts.append(
            "\n".join(
                [
                    f"Document {index} (source: {source})",
                    document.content,
                ]
            )
        )
    return "\n\n".join(parts)


def build_prompt(
    query: str,
    documents: List[SourceDocument],
    feedback_snippets: List[FeedbackSnippet],
    history: List[ChatMessage],
    user_role: str,
) -> str:
    history_text = format_chat_history(history)
    documents_text = format_documents(documents)
    if feedback_snippets:
        feedback_lines = []
        for snippet in feedback_snippets:
            feedback_lines.append(
                "\n".join(
                    [
                        f"Role: {snippet.user_role} (weight {snippet.weight}, score {snippet.score:.2f})",
                        f"Original query: {snippet.query}",
                        "Updated response:",
                        snippet.updated_response,
                    ]
                )
            )
        feedback_text = "\n\n".join(feedback_lines)
    else:
        feedback_text = "No user feedback matched the query."

    role_weight = config.ROLE_WEIGHTS.get(user_role, 1)

    return "\n".join(
        [
            SYSTEM_PROMPT,
            f"The active user role is {user_role} with weight {role_weight}.",
            "Chat history:",
            history_text or "No previous conversation.",
            "\nRetrieved documents:",
            documents_text,
            "\nRelevant user feedback (highest priority first):",
            feedback_text,
            "\nInstructions: Integrate the user feedback into your answer. Use the highest priority feedback as the authoritative source. If feedback contradicts documents, follow the feedback that has the highest role weight. Mention operational details captured in feedback when applicable. Provide a concise answer that references the relevant steps.",
            f"\nUser question: {query}",
        ]
    )
