from unittest.mock import Mock

from domain.agents.answer_writer import AnswerWriter
from domain.agents.content_researcher import ContentResearcher
from domain.agents.question_analyst import QuestionAnalyst
from domain.services.qa_service import QAService
from tools.document_search_tool import DocumentSearchTool


def test_qa_service_simple():
    llm = Mock(complete=Mock(return_value="topics"))
    analyst = QuestionAnalyst(llm=llm)
    rag_tool = DocumentSearchTool(Mock(query=Mock(return_value=["context"])))
    researcher = ContentResearcher(tool=rag_tool)
    writer = AnswerWriter(llm=Mock(complete=Mock(return_value="final answer")))

    service = QAService(analyst=analyst, researcher=researcher, writer=writer)
    result = service.answer("pergunta")

    assert result == "final answer"
