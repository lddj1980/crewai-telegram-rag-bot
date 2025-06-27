from typing import List

from domain.ports.rag_port import RAGPort


class DocumentSearchTool:
    """Tool used by CrewAI agents to search the document."""

    def __init__(self, rag: RAGPort) -> None:
        self.rag = rag

    def run(self, query: str) -> List[str]:
        return self.rag.query(query)
