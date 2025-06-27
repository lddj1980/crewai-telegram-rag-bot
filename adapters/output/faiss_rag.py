from pathlib import Path
from typing import List

import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from domain.ports.rag_port import RAGPort


class FAISSRAG(RAGPort):
    """RAG adapter using a local FAISS index."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings()
        self.store = FAISS.load_local(str(index_path), self.embeddings)

    def query(self, question: str) -> List[str]:
        docs = self.store.similarity_search(question, k=3)
        return [doc.page_content for doc in docs]
