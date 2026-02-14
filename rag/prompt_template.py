# rag/prompt_template.py
from langchain.prompts import PromptTemplate

def get_citation_prompt() -> PromptTemplate:
    template = (
        "You are an assistant that must answer using ONLY the provided context.\n"
        "If the answer is not in the context, reply: 'I don't know.'\n"
        "When you use context, include a short citation tag in square brackets like [source: filename.txt].\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (be concise, include citations when applicable):"
    )
    return PromptTemplate(input_variables=["context", "question"], template=template)
