from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS

from .. import config
from .embeddings import get_embedding_model


def _load_store(path: Path) -> Optional[FAISS]:
    if not path.exists():
        return None
    try:
        return FAISS.load_local(
            folder_path=str(path),
            embeddings=get_embedding_model(),
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


def _save_store(store: FAISS, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))


def load_document_store() -> Optional[FAISS]:
    return _load_store(config.VECTOR_STORE_DIR)


def save_document_store(store: FAISS) -> None:
    _save_store(store, config.VECTOR_STORE_DIR)


def load_feedback_store() -> Optional[FAISS]:
    return _load_store(config.FEEDBACK_VECTOR_DIR)


def save_feedback_store(store: FAISS) -> None:
    _save_store(store, config.FEEDBACK_VECTOR_DIR)
