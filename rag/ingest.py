
import os
from pathlib import Path
from typing import List

from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.config import RAW_DIR, PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP

SUPPORTED = {".pdf", ".md", ".txt", ".html", ".htm"}


def _load_one(path: Path) -> List[Document]:
    suf = path.suffix.lower()
    if suf == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()
    if suf in {".md", ".txt"}:
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()
    if suf in {".html", ".htm"}:
        loader = UnstructuredHTMLLoader(str(path))
        return loader.load()
    return []


def load_corpus(raw_dir: str = RAW_DIR) -> List[Document]:
    p = Path(raw_dir)
    if not p.exists():
        raise FileNotFoundError(f"Raw dir '{raw_dir}' not found. Create it and add your files.")
    paths = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in SUPPORTED]
    docs = []
    for path in paths:
        try:
            loaded = _load_one(path)
            for d in loaded:
                if not d.metadata:
                    d.metadata = {}
                d.metadata["source_file"] = str(path.name)
            docs.extend(loaded)
            print(f"[INGEST] Loaded {len(loaded)} docs from {path.name}")
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
    print(f"[INGEST] Total loaded documents: {len(docs)} from {len(paths)} files.")
    return docs


def split_docs(docs: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"[INGEST] Split into {len(chunks)} chunks.")
    return chunks


def save_processed_text(chunks: List[Document], out_dir: str = PROCESSED_DIR):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(chunks):
        fname = outp / f"chunk_{i:05d}.txt"
        try:
            fname.write_text(c.page_content, encoding="utf-8")
        except Exception:
            fname.write_text(str(c.page_content), encoding="utf-8")
    print(f"[INGEST] Saved {len(chunks)} chunks to {out_dir}")


def run_ingest():
    docs = load_corpus(RAW_DIR)
    if not docs:
        print("[INGEST] No documents found to ingest.")
        return []
    chunks = split_docs(docs)
    save_processed_text(chunks, PROCESSED_DIR)
    return chunks


if __name__ == "__main__":
    run_ingest()
