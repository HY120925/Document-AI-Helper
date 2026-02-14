# rag/build.py
from rag.ingest import run_ingest
from rag.vectorstore import build_vectorstore

def build_all():
    print("[BUILD] Running ingestion...")
    chunks = run_ingest()
    if not chunks:
        print("[BUILD] No chunks created; aborting vectorstore build.")
        return
    print("[BUILD] Building vectorstore...")
    build_vectorstore(chunks)
    print("[BUILD] Done.")

if __name__ == "__main__":
    build_all()
