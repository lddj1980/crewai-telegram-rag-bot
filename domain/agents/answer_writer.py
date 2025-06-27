from dataclasses import dataclass
from typing import List

from domain.ports.llm_port import LLMPort


@dataclass
class AnswerWriter:
    """Composes the final answer using an LLM."""

    llm: LLMPort

    def write(self, question: str, context: List[str]) -> str:
        """Return the final answer given the question and context."""
        snippet_text = "\n".join(context)
        prompt = (
            "Você é um redator especializado. Utilize os trechos abaixo para "
            f"responder à pergunta de forma clara.\nPergunta: {question}\n\n"
            f"Trechos:\n{snippet_text}\n\nResposta:"
        )
        return self.llm.complete(prompt)
