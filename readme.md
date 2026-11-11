# RAG Q&A API

A FastAPI service that answers questions about any text you give it, using Retrieval-Augmented Generation (RAG) with LangChain and Google Gemini.

## How it works
1. `POST /documents` splits your text into chunks, embeds them, and stores them in a vector store.
2. `POST /ask` finds the 3 chunks most similar to your question and asks the LLM to answer using only that context.
3. The response includes the source chunks, so answers can be verified.

## Tech stack
Python, FastAPI, LangChain, Google Gemini (chat + embeddings), Pydantic

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and add your Google AI Studio API key.

## Run
```
uvicorn rag_api:app --reload
```
Open http://127.0.0.1:8000/docs to try the endpoints.

## Example
POST /documents
{"text": "SynapseBug is a Flask app that summarizes text and hashes passwords with bcrypt."}

POST /ask
{"question": "How are passwords protected?"}

## Limitations and next steps
- The vector store is in memory and resets on restart (next: Chroma for persistence).
- One document at a time (next: per-user collections, PDF upload).
- No automated tests yet (next: pytest for the endpoints).

## Evaluation
An automated eval suite checks the API against a golden set of questions:
- Answerable questions must contain the expected facts and return source passages
- Out-of-scope questions must be declined instead of answered
- `/ask` before any document is indexed must return an error

```
pip install -r requirements.txt
pytest -v
```
Tests call the real LLM, so results can vary slightly; rerun if the service reports high demand.