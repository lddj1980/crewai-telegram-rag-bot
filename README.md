# crewai-telegram-rag-bot

This project implements a Telegram bot that answers questions using Retrieval-Augmented Generation (RAG) powered by CrewAI. The bot uses a vector store based on FAISS and the DeepSeek LLM API to generate responses.

See `AGENTS.md` for contributor guidelines.

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Create a `.env` file based on the variables in `.env.example` and provide your tokens.
3. Run the ingestion script before starting the bot:
```bash
python scripts/ingest_document.py
```
4. Start the bot:
```bash
python -m app.main
```

Run tests with `pytest -q`.