from dataclasses import dataclass
from typing import List

from domain.agents.answer_writer import AnswerWriter
from domain.agents.content_researcher import ContentResearcher
from domain.agents.question_analyst import QuestionAnalyst


@dataclass
class QAService:
    """Orchestrates the three agents to answer questions.

    Attributes
    ----------
    analyst:
        Agent responsible for analyzing the question.
    researcher:
        Agent that searches for relevant snippets.
    writer:
        Agent that writes the final answer.
    verbose:
        Whether to enable Crew progress output.
    """

    analyst: QuestionAnalyst
    researcher: ContentResearcher
    writer: AnswerWriter
    verbose: bool = True

    def answer(self, question: str) -> str:
        """Execute the crew to produce the final answer."""
        try:
            from crewai import Agent, Task, Crew
        except Exception:  # pragma: no cover - used only when CrewAI isn't installed
            topics = self.analyst.analyze(question)
            context = self.researcher.research(topics)
            return self.writer.write(question, context)

        analyst_agent = Agent(
            role="Analista de Perguntas",
            goal=(
                "Compreender a pergunta do usuário e identificar tópicos principais"
            ),
            backstory=(
                "Especialista em entender perguntas complexas e transformá-las em tópicos"
            ),
            llm=self.analyst.llm,  # type: ignore[arg-type]
        )

        researcher_agent = Agent(
            role="Pesquisador de Conteúdo",
            goal="Buscar trechos relevantes no documento com base nos tópicos",
            backstory="Especialista em encontrar informação textual precisa",
            tools=[self.researcher.tool],
            llm=self.writer.llm,  # type: ignore[arg-type]
        )

        writer_agent = Agent(
            role="Redator Especializado",
            goal="Escrever uma resposta clara e útil baseada no conteúdo encontrado",
            backstory="Redator experiente com clareza textual e didática",
            llm=self.writer.llm,  # type: ignore[arg-type]
        )

        analyze = Task(
            description="Identifique os tópicos principais da pergunta: {question}",
            agent=analyst_agent,
        )

        research = Task(
            description="Pesquise no documento usando os tópicos: {analyze}",
            agent=researcher_agent,
        )

        write = Task(
            description=(
                "Redija a resposta para a pergunta '{question}' usando o contexto: {research}"
            ),
            agent=writer_agent,
        )

        crew = Crew(
            agents=[analyst_agent, researcher_agent, writer_agent],
            tasks=[analyze, research, write],
            verbose=self.verbose,
        )

        return crew.kickoff(inputs={"question": question})
