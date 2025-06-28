from typing import List

from domain.ports.rag_port import RAGPort

try:  # pragma: no cover - optional dependency
    from pydantic import PrivateAttr
except Exception:  # pragma: no cover - fallback when pydantic isn't installed
    def PrivateAttr(default=None):  # type: ignore[misc]
        return default


try:  # pragma: no cover - optional dependency
    from langchain_core.tools import BaseTool
except Exception:  # pragma: no cover - fallback when langchain isn't installed
    BaseTool = object  # type: ignore[misc]


class DocumentSearchTool(BaseTool):
    """Search the document for relevant snippets."""

    name: str = "document_search"
    description: str = "Busca trechos relevantes em um documento."
    rag: RAGPort = PrivateAttr()

    def __init__(self, rag: RAGPort) -> None:
        super().__init__()
        self.rag = rag

    def run(self, query: str) -> List[str]:
        """Return snippets matching the query."""
        return self.rag.query(query)

    def _run(self, query: str) -> str:  # pragma: no cover - crewai usage
        return "\n".join(self.run(query))

    async def _arun(self, query: str) -> str:  # pragma: no cover - rarely used
        raise NotImplementedError
