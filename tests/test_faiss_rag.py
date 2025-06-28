from pathlib import Path
from types import ModuleType
import sys
import pytest

from adapters.output.faiss_rag import FAISSRAG


def test_faiss_rag_missing_index(tmp_path: Path):
    missing = tmp_path / "no_index"
    with pytest.raises(FileNotFoundError):
        FAISSRAG(missing)


def test_faiss_rag_loads_index(monkeypatch, tmp_path: Path) -> None:
    index_dir = tmp_path / "idx"
    index_dir.mkdir()

    embeddings = object()

    class FakeFAISS:
        @staticmethod
        def load_local(*args, **kwargs):
            nonlocal called_args, called_kwargs
            called_args = args
            called_kwargs = kwargs
            return object()

    called_args = ()
    called_kwargs = {}

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        ModuleType("openai_module"),
    )
    sys.modules["langchain_openai"].OpenAIEmbeddings = lambda: embeddings

    monkeypatch.setitem(
        sys.modules,
        "langchain_community.vectorstores",
        ModuleType("vectorstores"),
    )
    sys.modules["langchain_community.vectorstores"].FAISS = FakeFAISS

    FAISSRAG(index_dir)

    assert called_args == (str(index_dir), embeddings)
    assert called_kwargs["allow_dangerous_deserialization"] is True
