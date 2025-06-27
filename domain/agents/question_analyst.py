from dataclasses import dataclass

from domain.ports.llm_port import LLMPort


@dataclass
class QuestionAnalyst:
    """Analyzes a user question and extracts main topics using an LLM."""

    llm: LLMPort

    def analyze(self, question: str) -> str:
        """Return a short description of main topics."""
        prompt = (
            "Liste de forma resumida os principais tópicos da pergunta a seguir:"
            f"\n{question}\nTópicos:"
        )
        return self.llm.complete(prompt)