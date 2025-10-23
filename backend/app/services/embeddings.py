from functools import lru_cache

from langchain_community.embeddings import SentenceTransformerEmbeddings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformerEmbeddings:
    """Return a shared embedding model instance."""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
