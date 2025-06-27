from dataclasses import dataclass
from typing import List

from tools.document_search_tool import DocumentSearchTool


@dataclass
class ContentResearcher:
    """Searches for relevant document snippets."""

    tool: DocumentSearchTool

    def research(self, topics: str) -> List[str]:
        """Return snippets relevant to the given topics."""
        return self.tool.run(topics)
