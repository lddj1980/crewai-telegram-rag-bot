"""Script to ingest the base document into a FAISS index."""

from pathlib import Path

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()

DATA_FILE = Path("data/documento.txt")
INDEX_DIR = Path("vector_store/faiss_index")


def main() -> None:
    text = DATA_FILE.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(str(INDEX_DIR))


if __name__ == "__main__":
    main()
