# tests/test_queries.py
from rag.vectorstore import load_vectorstore
from rag.chain import build_qa_chain

SMOKE = [
    "Give me a short summary of the corpus.",
    "What does the document say about the main topic?",
    "List three key facts with citations."
]

if __name__ == "__main__":
    db = load_vectorstore()
    qa = build_qa_chain(db)
    for q in SMOKE:
        print("[Q]", q)
        try:
            a = qa.run(q)
        except Exception as e:
            a = f"[ERROR] {e}"
        print("[A]", a)
        print("-" * 60)
