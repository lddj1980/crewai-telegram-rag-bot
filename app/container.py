import os
from pathlib import Path

from adapters.input.telegram_bot import TelegramBot
from adapters.output.openai_llm import OpenAILLM
from adapters.output.faiss_rag import FAISSRAG
from domain.agents.answer_writer import AnswerWriter
from domain.agents.content_researcher import ContentResearcher
from domain.agents.question_analyst import QuestionAnalyst
from domain.services.qa_service import QAService
from tools.document_search_tool import DocumentSearchTool


class Container:
    """Simple dependency injection container."""

    def __init__(self) -> None:
        index_dir = Path("vector_store/faiss_index")
        # Infrastructure adapters
        rag = FAISSRAG(index_dir)
        llm = OpenAILLM()
        tool = DocumentSearchTool(rag)
        verbose = os.getenv("CREW_VERBOSE", "True").lower() in ("1", "true", "yes")

        # Domain agents
        analyst = QuestionAnalyst(llm=llm)
        researcher = ContentResearcher(tool=tool)
        writer = AnswerWriter(llm=llm)

        # Service that orchestrates the crew
        self.qa_service = QAService(
            analyst=analyst,
            researcher=researcher,
            writer=writer,
            verbose=verbose,
        )
        self.bot = TelegramBot(qa_handler=self.qa_service.answer)
