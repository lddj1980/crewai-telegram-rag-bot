# crewai-telegram-rag-bot

This project implements a Telegram bot that answers questions using Retrieval-Augmented Generation (RAG) powered by CrewAI. The bot uses a vector store based on FAISS and the DeepSeek LLM API to generate responses.

See `AGENTS.md` for contributor guidelines.

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Create a `.env` file based on `.env.example` and provide your tokens. Set
   `OPENAI_API_KEY` and `OPENAI_API_BASE` for the DeepSeek API. `CREW_VERBOSE`
   controls whether CrewAI prints progress (default `True`).
3. Build the index by running the ingestion script:
```bash
python scripts/ingest_document.py
```
   This reads `data/documento.txt` and writes the FAISS index to
   `vector_store/faiss_index`. Edit `scripts/ingest_document.py` if you
   want to use different paths.
4. Start the bot:
```bash
python -m app.main
```

Run tests with `pytest -q`.
