# tests/test_ingest.py
from rag.ingest import run_ingest

if __name__ == "__main__":
    chunks = run_ingest()
    print(f"[TEST] Ingest produced {len(chunks)} chunks.")
