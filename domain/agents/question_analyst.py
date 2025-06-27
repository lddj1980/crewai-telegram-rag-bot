from dataclasses import dataclass


@dataclass
class QuestionAnalyst:
    """Analyzes a user question and extracts main topics."""

    def analyze(self, question: str) -> str:
        """Return a short description of main topics.

        This is a placeholder implementation that simply returns the question
        itself. A real implementation would use NLP techniques.
        """
        return question
