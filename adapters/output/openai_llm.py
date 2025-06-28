import os

from openai import OpenAI

from domain.ports.llm_port import LLMPort


class OpenAILLM(LLMPort):
    """LLM adapter using the OpenAI API."""

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE")
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def complete(self, prompt: str) -> str:
        chat = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        return chat.choices[0].message.content
