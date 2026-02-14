
import gradio as gr
import os
from rag.vectorstore import load_vectorstore
from rag.chain import build_qa_chain
from app.config import VECTORSTORE_PATH, GRADIO_SERVER_NAME, GRADIO_SERVER_PORT

def launch_ui():
    if not os.path.exists(VECTORSTORE_PATH):
        return print(f"[ERROR] Vectorstore not found at '{VECTORSTORE_PATH}'. Run ingest/build first.")

    db = load_vectorstore()
    qa = build_qa_chain(db)

    def answer(query: str) -> str:
        if not query or not query.strip():
            return "Please enter a question about your documents."
        try:
            return qa.run(query)
        except Exception as e:
            return f"[ERROR] {e}"

    title = "Sentinel RAG Assistant"
    desc = "Ask questions about your local documents. Answers include citations where available."
    iface = gr.Interface(
        fn=answer,
        inputs=gr.Textbox(lines=2, placeholder="Ask about your docs..."),
        outputs=gr.Textbox(lines=10),
        title=title,
        description=desc,
        allow_flagging="never"
    )
    iface.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT, share=False)
