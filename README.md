#  Document AI Helper

This project is a document-based question answering system built using a Retrieval-Augmented Generation (RAG) pipeline.

The goal was simple:

I wanted a system that could understand my own documents and answer questions grounded in them — not hallucinate.

It ingests raw documents, converts them into embeddings, stores them in FAISS, and retrieves relevant context before generating answers.

#  Why Built This

Most LLM apps fail because:

They hallucinate

They don't remember large documents

They can't search efficiently

This project solves that by combining:

Vector embeddings

FAISS similarity search

Prompt engineering

Modular RAG architecture

#  How It Works

Documents are loaded and cleaned.

Text is split into chunks (configurable size + overlap).

Each chunk is converted into an embedding.

Embeddings are stored inside a FAISS index.

When a user asks a question:

The system retrieves the most relevant chunks

Injects them into a prompt

Sends everything to the LLM

Returns a context-grounded answer

No guessing. Only document-backed responses.

#  Project Structure
main.py                -> entry point
app/
  config.py            -> configuration
  ui.py                -> query interface

data_ingest/
  ingest.py            -> document processing
  embed.py             -> embedding wrapper
  build.py             -> index builder
  vectorstore.py       -> FAISS helpers
  prompt_template.py   -> prompt logic
  processed/           -> generated chunks

vectorstore/
  index.faiss          -> saved vector index

tests/                 -> ingestion & retrieval tests

#  Setup

Create environment:

python -m venv .venv


Install dependencies:

pip install -r requirements.txt


Build vectorstore:

python data_ingest/ingest.py
python data_ingest/build.py


Run:

python main.py

#  What This Project Demonstrates

Practical use of embeddings

Vector similarity search with FAISS

Clean separation of ingestion / retrieval / generation

Modular ML system design

Basic test coverage

#  Things I’d Improve Next

Add reranking layer

Add evaluation metrics (Precision@K, Recall)

Replace local FAISS with cloud vector DB

Wrap it with FastAPI

Add streaming responses

#  Notes

If the FAISS index is missing, rebuild it using:

python data_ingest/build.py

# Author : Yogesh