import re
from typing import Dict, List, Set, Tuple

from ..models import ChatMessage, FeedbackSnippet, SourceDocument
from .. import config


SYSTEM_PROMPT = (
    "You are an AI assistant helping a logistics company answer questions using internal documents. "
    "Retrieved documents provide operational context. User feedback entries capture the most up-to-date guidance "
    "from subject matter experts and should be treated as authoritative when they apply to the current question."
)

QUERY_SIMILARITY_THRESHOLD = 0.2
DOCUMENT_RELEVANCE_THRESHOLD = 0.1


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


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"\b\w+\b", text.casefold()))


def _jaccard_similarity(first: Set[str], second: Set[str]) -> float:
    if not first or not second:
        return 0.0
    intersection = first & second
    union = first | second
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _select_applicable_feedback(
    query: str,
    documents: List[SourceDocument],
    feedback_snippets: List[FeedbackSnippet],
) -> List[FeedbackSnippet]:
    if not feedback_snippets:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    best_matches: Dict[str, Tuple[FeedbackSnippet, float]] = {}

    for document in documents:
        doc_tokens = _tokenize(document.content)
        if not doc_tokens:
            continue
        for snippet in feedback_snippets:
            snippet_id = str(snippet.id)
            snippet_tokens = _tokenize(snippet.query)
            if not snippet_tokens:
                continue

            query_similarity = _jaccard_similarity(query_tokens, snippet_tokens)
            intersection = doc_tokens & snippet_tokens
            if not intersection:
                continue
            document_relevance = len(intersection) / len(snippet_tokens)
            if query_similarity < QUERY_SIMILARITY_THRESHOLD or document_relevance < DOCUMENT_RELEVANCE_THRESHOLD:
                continue

            combined_score = query_similarity + document_relevance
            existing = best_matches.get(snippet_id)
            if existing is None or combined_score > existing[1]:
                best_matches[snippet_id] = (snippet, combined_score)

    if not best_matches:
        return []

    ranked_feedback = sorted(
        best_matches.values(),
        key=lambda item: (
            config.ROLE_WEIGHTS.get(item[0].user_role, 0),
            item[0].created_at
        ),
        reverse=True,
    )
    return [item[0] for item in ranked_feedback[:1]]


def build_prompt(
    query: str,
    documents: List[SourceDocument],
    feedback_snippets: List[FeedbackSnippet],
    history: List[ChatMessage],
    user_role: str,
) -> Tuple[str, List[FeedbackSnippet]]:
    history_text = format_chat_history(history)
    documents_text = format_documents(documents)
    applicable_feedback = _select_applicable_feedback(query, documents, feedback_snippets)

    if applicable_feedback:
        feedback_lines = []
        for snippet in applicable_feedback:
            feedback_lines.append(
                "\n".join(
                    [
                        f"Provided by: {snippet.user_role} on {snippet.created_at.isoformat()}",
                        f"Original query: {snippet.query}",
                        "Updated response:",
                        snippet.updated_response,
                    ]
                )
            )
        feedback_text = "\n\n".join(feedback_lines)
    else:
        feedback_text = "No applicable user feedback for this question."

    prompt_sections = [
        SYSTEM_PROMPT,
        f"Active user role: {user_role}.",
        "Chat history:",
        history_text or "No previous conversation.",
        "\nRetrieved documents:",
        documents_text,
        "\nApplicable user feedback:",
        feedback_text,
        "\nInstructions: If applicable user feedback is provided, treat it as the authoritative update while ensuring the final answer remains coherent with the retrieved documents. If no feedback is available, answer using the documents. Provide a concise, actionable response.",
        f"\nUser question: {query}",
    ]

    return "\n".join(prompt_sections), applicable_feedback
