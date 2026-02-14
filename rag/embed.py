# rag/embed.py
from typing import Any
from app.config import EMBEDDING_MODEL, OPENAI_API_KEY
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    """
    Returns an embeddings object compatible with LangChain vectorstore.
    Defaults to HuggingFaceEmbeddings (sentence-transformers).
    If EMBEDDING_MODEL indicates an OpenAI embedding and an API key is provided, use OpenAI.
    """
    model = EMBEDDING_MODEL or "sentence-transformers/all-MiniLM-L6-v2"
    if model.startswith("text-embedding-") and OPENAI_API_KEY:
        # OpenAI embeddings
        from langchain.embeddings import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model, openai_api_key=OPENAI_API_KEY)
    else:
        # HuggingFace local embeddings (no API key)
        from langchain.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model)
