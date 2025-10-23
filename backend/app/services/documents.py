import json
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from .. import config

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    length_function=len,
)


def _load_text_document(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": path.name})]


def _load_json_document(path: Path) -> List[Document]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, str):
        return [Document(page_content=data, metadata={"source": path.name})]
    if isinstance(data, dict):
        return [Document(page_content=json.dumps(data, ensure_ascii=False), metadata={"source": path.name})]
    if isinstance(data, list):
        return [
            Document(page_content=json.dumps(item, ensure_ascii=False), metadata={"source": path.name})
            for item in data
        ]
    return [Document(page_content=json.dumps(data, ensure_ascii=False), metadata={"source": path.name})]


def _load_pdf_document(path: Path) -> List[Document]:
    loader = PyPDFLoader(str(path))
    return loader.load()


def load_documents(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _load_text_document(path)
    if suffix == ".json":
        return _load_json_document(path)
    if suffix == ".pdf":
        return _load_pdf_document(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_documents(documents: List[Document]) -> List[Document]:
    return text_splitter.split_documents(documents)


def store_upload(file_name: str, data: bytes) -> Path:
    config.ensure_directories()
    destination = config.UPLOAD_DIR / file_name
    destination.write_bytes(data)
    return destination
