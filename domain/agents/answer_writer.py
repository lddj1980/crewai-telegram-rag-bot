from dataclasses import dataclass
from typing import List

from domain.ports.llm_port import LLMPort


@dataclass
class AnswerWriter:
    """Composes the final answer using an LLM."""

    llm: LLMPort

    def write(self, question: str, context: List[str]) -> str:
        """Return the final answer given the question and context."""
        prompt = f"Question: {question}\nContext: {' '.join(context)}\nAnswer:"
        return self.llm.complete(prompt)
