# rag/vectorstore.py
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from app.config import VECTORSTORE_PATH
from rag.embed import get_embeddings


def build_vectorstore(chunks: List[Document], save_path: str = VECTORSTORE_PATH):
    emb = get_embeddings()
    db = FAISS.from_documents(chunks, emb)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    db.save_local(save_path)
    print(f"[VSTORE] Saved FAISS vectorstore to '{save_path}'")
    return db


def load_vectorstore(load_path: str = VECTORSTORE_PATH):
    emb = get_embeddings()
    if not Path(load_path).exists():
        raise FileNotFoundError(f"Vectorstore path '{load_path}' does not exist. Build it first.")
    db = FAISS.load_local(load_path, emb, allow_dangerous_deserialization=True)
    print(f"[VSTORE] Loaded FAISS vectorstore from '{load_path}'")
    return db
