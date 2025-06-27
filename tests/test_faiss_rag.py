from pathlib import Path
import pytest

from adapters.output.faiss_rag import FAISSRAG


def test_faiss_rag_missing_index(tmp_path: Path):
    missing = tmp_path / "no_index"
    with pytest.raises(FileNotFoundError):
        FAISSRAG(missing)
