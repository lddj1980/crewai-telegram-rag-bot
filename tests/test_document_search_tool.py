from unittest.mock import Mock

from tools.document_search_tool import DocumentSearchTool


def test_document_search_tool_uses_rag():
    rag = Mock(query=Mock(return_value=["snippet"]))
    tool = DocumentSearchTool(rag)
    result = tool.run("question")
    assert result == ["snippet"]
    rag.query.assert_called_with("question")
