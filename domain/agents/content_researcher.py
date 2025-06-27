from dataclasses import dataclass
from typing import List

from domain.ports.rag_port import RAGPort


@dataclass
class ContentResearcher:
    """Searches for relevant document snippets."""

    rag: RAGPort

    def research(self, topics: str) -> List[str]:
        """Return snippets relevant to the given topics."""
        return self.rag.query(topics)
