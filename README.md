# Final Project SDT204 - Local RAG

A simple **Retrieval-Augmented Generation (RAG)** application. It allows you to chat with locally running model with context from your own documents (PDFs and TXTs).

## Features

*   **100% Local & Private**: No data leaves your machine. Uses local embeddings, LLMs, and databases.
*   **Smart Indexing**: Checks for duplicates before adding data. You can run the index command multiple times safely.
*   **Manual Embeddings**: Direct control over the embedding process using Ollama API.
*   **Persistent Database**: Vectors are saved locally in `./chroma_db`.
*   **Chat History**: Saves all Q&A and conversations to `chat_history.json`.
*   **Multiple Modes**:
    *   `index`: Scan and vectorise documents.
    *   `ask`: Ask a single question.
    *   `chat`: Start a named chat session.
    *   `select`: Resume a chat conversation.

---

## Prerequisites

1.  **Python 3.12**
2.  **[Ollama](https://ollama.com/)** installed and running.

### Required Models
You must pull the models used in the configuration (default: `mistral` for chat and `nomic-embed-text` for embeddings).

Run these commands in your terminal:
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

## Running the application

```bash
docker compose build
```

# Index files (Fresh start)
```bash
docker compose run --rm rag-app index --reset
```

# Ask a question
```bash
docker compose run --rm rag-app ask "Explain the document"
```

# Start a Chat
```bash
docker compose run --rm rag-app chat my-docker-session
```
