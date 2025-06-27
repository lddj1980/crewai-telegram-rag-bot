from abc import ABC, abstractmethod
from typing import List


class RAGPort(ABC):
    """Interface for retrieval of context snippets."""

    @abstractmethod
    def query(self, question: str) -> List[str]:
        """Return a list of relevant document snippets for the given question."""
        raise NotImplementedError
