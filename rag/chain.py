
from app.config import OPENAI_API_KEY, LLM_MODEL, TOP_K, LLM_BACKEND
from rag.prompt_template import get_citation_prompt
from langchain.chains import RetrievalQA

from langchain_community.llms import Ollama, HuggingFaceHub

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None


def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    prompt = get_citation_prompt()
    llm = None

    if LLM_BACKEND == "openai":
        if not OPENAI_API_KEY or not ChatOpenAI:
            raise RuntimeError("OpenAI backend selected but no API key found.")
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        print("[CHAIN] Using OpenAI backend:", LLM_MODEL)

    elif LLM_BACKEND == "ollama":
        llm = Ollama(model="llama2")
        print("[CHAIN] Using Ollama backend: llama2")

    elif LLM_BACKEND == "huggingface":
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0, "max_length": 512}
        )
        print("[CHAIN] Using HuggingFace backend: flan-t5-base")

    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
