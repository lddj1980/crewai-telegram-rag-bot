from abc import ABC, abstractmethod


class LLMPort(ABC):
    """Interface for LLM completions."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return a completion for the given prompt."""
        raise NotImplementedError
