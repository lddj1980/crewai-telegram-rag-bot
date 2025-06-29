from typing import List

from domain.ports.rag_port import RAGPort

try:  # pragma: no cover - optional dependency
    from pydantic import PrivateAttr
except Exception:  # pragma: no cover - fallback when pydantic isn't installed
    def PrivateAttr(default=None):  # type: ignore[misc]
        return default


try:  # pragma: no cover - prefer CrewAI BaseTool when available
    from crewai.tools.base_tool import BaseTool
except Exception:  # pragma: no cover - fallback when CrewAI isn't installed
    try:  # pragma: no cover - optional dependency
        from langchain_core.tools import BaseTool
    except Exception:  # pragma: no cover - fallback when langchain isn't installed
        try:  # pragma: no cover - attempt pydantic-based replacement
            from pydantic import BaseModel
            from typing import Any

            class BaseTool(BaseModel):  # type: ignore[misc]
                """Minimal BaseTool replacement."""

                name: str = ""
                description: str = ""

                def __init__(self, **data: Any) -> None:  # pragma: no cover - simple init
                    super().__init__(**data)

                def _run(self, *args: str, **kwargs: str) -> str:  # pragma: no cover - stub
                    raise NotImplementedError

                async def _arun(self, *args: str, **kwargs: str) -> str:  # pragma: no cover - stub
                    raise NotImplementedError

        except Exception:  # pragma: no cover - ultimate fallback
            class BaseTool:  # type: ignore[misc]
                pass


class DocumentSearchTool(BaseTool):
    """Search the document for relevant snippets."""

    name: str = "document_search"
    description: str = "Busca trechos relevantes em um documento."
    _rag: RAGPort = PrivateAttr()

    def __init__(self, rag: RAGPort) -> None:
        super().__init__()
        self._rag = rag

    def run(self, query: str) -> List[str]:
        """Return snippets matching the query."""
        return self._rag.query(query)

    def _run(self, query: str) -> str:  # pragma: no cover - crewai usage
        return "\n".join(self.run(query))

    async def _arun(self, query: str) -> str:  # pragma: no cover - rarely used
        raise NotImplementedError
