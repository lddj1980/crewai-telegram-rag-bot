from unittest.mock import Mock

from domain.agents.answer_writer import AnswerWriter
from domain.agents.content_researcher import ContentResearcher
from domain.agents.question_analyst import QuestionAnalyst
from domain.services.qa_service import QAService


def test_qa_service_simple():
    analyst = QuestionAnalyst()
    researcher = ContentResearcher(rag=Mock(query=Mock(return_value=["context"])))
    writer = AnswerWriter(llm=Mock(complete=Mock(return_value="final answer")))

    service = QAService(analyst=analyst, researcher=researcher, writer=writer)
    result = service.answer("pergunta")

    assert result == "final answer"
