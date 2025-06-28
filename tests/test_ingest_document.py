from pathlib import Path
from types import ModuleType
from unittest.mock import Mock
import importlib
import sys


def test_ingest_document(tmp_path: Path, monkeypatch) -> None:
    splitter_cls = Mock()
    embeddings_cls = Mock()
    faiss_module = ModuleType("faiss")
    faiss_module.FAISS = Mock()

    monkeypatch.setitem(
        sys.modules, "langchain_community.embeddings", ModuleType("emb")
    )
    sys.modules["langchain_community.embeddings"].OpenAIEmbeddings = embeddings_cls
    monkeypatch.setitem(sys.modules, "langchain.text_splitter", ModuleType("split"))
    sys.modules["langchain.text_splitter"].RecursiveCharacterTextSplitter = splitter_cls
    monkeypatch.setitem(
        sys.modules, "langchain_community.vectorstores", ModuleType("vec")
    )
    sys.modules["langchain_community.vectorstores"].FAISS = faiss_module.FAISS
    monkeypatch.setitem(sys.modules, "dotenv", ModuleType("dotenv"))
    sys.modules["dotenv"].load_dotenv = Mock()

    ingest_document = importlib.import_module("scripts.ingest_document")

    data_file = tmp_path / "doc.txt"
    data_file.write_text("text", encoding="utf-8")
    index_dir = tmp_path / "idx"

    monkeypatch.setattr(ingest_document, "DATA_FILE", data_file)
    monkeypatch.setattr(ingest_document, "INDEX_DIR", index_dir)

    docs = ["chunk"]
    splitter = Mock(create_documents=Mock(return_value=docs))
    splitter_cls.return_value = splitter

    embeddings = Mock()
    embeddings_cls.return_value = embeddings

    store = Mock(save_local=Mock())
    faiss_module.FAISS.from_documents = Mock(return_value=store)

    ingest_document.main()

    faiss_module.FAISS.from_documents.assert_called_with(docs, embeddings)
    store.save_local.assert_called_with(str(index_dir))
    assert index_dir.exists()
