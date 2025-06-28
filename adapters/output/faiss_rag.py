from pathlib import Path
from typing import List

from domain.ports.rag_port import RAGPort


class FAISSRAG(RAGPort):
    """RAG adapter using a local FAISS index."""

    def __init__(self, index_path: Path) -> None:
        """Load an existing FAISS index from disk."""
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found at {index_path}. Run ingest_document.py first."
            )

        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS

        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings()
        self.store = FAISS.load_local(str(index_path), self.embeddings)

    def query(self, question: str) -> List[str]:
        docs = self.store.similarity_search(question, k=3)
        return [doc.page_content for doc in docs]
