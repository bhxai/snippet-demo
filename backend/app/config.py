from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
FEEDBACK_VECTOR_DIR = DATA_DIR / "feedback_vector_store"
FEEDBACK_LOG_PATH = DATA_DIR / "feedback_log.json"

ROLE_WEIGHTS: Dict[str, int] = {
    "driver": 1,
    "manager": 2,
    "owner": 3,
}

ROLE_BOOST: Dict[str, float] = {
    "driver": 0.0,
    "manager": 0.3,
    "owner": 0.6,
}

DEFAULT_MODEL = "gpt-4.1-mini"

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    for path in (DATA_DIR, UPLOAD_DIR, VECTOR_STORE_DIR, FEEDBACK_VECTOR_DIR):
        path.mkdir(parents=True, exist_ok=True)
