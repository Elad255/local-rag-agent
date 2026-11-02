# Local RAG + Agent — "Chat with your documents"

A Retrieval-Augmented Generation (RAG) system with an agent layer, built from scratch and running **100% locally** (no API keys, no cost) using Ollama. Ask questions about a document and get grounded answers with their sources — or let an agent decide which tool to use.

## What it does

- **RAG pipeline:** loads a document, splits it into overlapping chunks, embeds them into vectors, stores them in a Chroma vector database, and retrieves the most relevant chunks for any question by semantic similarity.
- **Grounded answers:** retrieved chunks are passed to a local LLM (Llama 3.2 via Ollama) with an instruction to answer *only* from that context — so it cites real sources and says "I don't know" instead of hallucinating.
- **REST API:** exposed as a FastAPI `POST /ask` endpoint returning the answer plus its source chunks.
- **Agent:** gives the LLM tools (document search + calculator) and lets it decide which to call for a given question.

## Architecture

INDEXING (once):   document → chunk (with overlap) → embed → store in Chroma
QUERYING (per request):  question → embed → retrieve top-k chunks → prompt → LLM → grounded answer (+ sources)
AGENT:   question → LLM picks a tool → run tool → LLM answers from the result

## Tech stack

- **Python**
- **ChromaDB** — vector database + embeddings (all-MiniLM-L6-v2)
- **Ollama** (Llama 3.2) — local LLM for generation and tool-calling
- **FastAPI** — REST API

## How to run

1. Install [Ollama](https://ollama.com) and pull the model: `ollama pull llama3.2`
2. Create a virtual environment and install deps: `pip install -r requirements.txt`
3. Run the RAG script: `python rag.py`
4. Or run the API: `uvicorn app:app --reload` → open http://127.0.0.1:8000/docs
5. Or run the agent: `python agent.py`

## Notes

- Runs fully offline — no API keys, no cost.
- Built from scratch (no LangChain/LlamaIndex) to understand every step of the pipeline.