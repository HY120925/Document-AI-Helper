
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  

LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore")
RAW_DIR = os.getenv("RAW_DIR", "data_ingest/raw")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data_ingest/processed")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))

GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
