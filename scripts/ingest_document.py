"""Script to ingest the base document into a FAISS index.

The text is loaded from ``DATA_FILE`` (``data/documento.txt`` by default) and
the resulting index is stored in ``INDEX_DIR`` (``vector_store/faiss_index``).
OpenAI credentials are read from the environment. Set ``OPENAI_API_KEY`` and
optionally ``OPENAI_API_BASE`` before running. Run this script once before
starting the bot.
"""

from pathlib import Path
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()

REQUIRED_VARS = ("OPENAI_API_KEY",)
DATA_FILE = Path("data/documento.txt")
INDEX_DIR = Path("vector_store/faiss_index")


def main() -> None:
    for var in REQUIRED_VARS:
        if not os.environ.get(var):
            raise EnvironmentError(f"Environment variable {var} is not set")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Document not found at {DATA_FILE}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    text = DATA_FILE.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(str(INDEX_DIR))


if __name__ == "__main__":
    main()
