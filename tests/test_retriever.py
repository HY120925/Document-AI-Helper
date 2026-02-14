# tests/test_retriever.py
from rag.vectorstore import load_vectorstore

if __name__ == "__main__":
    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    query = "test retrieval"
    results = retriever.get_relevant_documents(query)
    print(f"[TEST] Retrieved {len(results)} documents for query: '{query}'")
    for i, r in enumerate(results[:3]):
        print(f"--- Result {i+1} source: {r.metadata.get('source_file', 'unknown')}")
        print(r.page_content[:200].replace("\n", " ") + "...")
