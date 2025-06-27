from dataclasses import dataclass
from typing import List

from domain.agents.answer_writer import AnswerWriter
from domain.agents.content_researcher import ContentResearcher
from domain.agents.question_analyst import QuestionAnalyst


@dataclass
class QAService:
    """Orchestrates the three agents to answer questions."""

    analyst: QuestionAnalyst
    researcher: ContentResearcher
    writer: AnswerWriter

    def answer(self, question: str) -> str:
        """Run the pipeline and return the final answer."""
        topics = self.analyst.analyze(question)
        context = self.researcher.research(topics)
        return self.writer.write(question, context)
