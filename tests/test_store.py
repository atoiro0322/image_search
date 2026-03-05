"""ImageStore のユニットテスト（ChromaDB は一時ディレクトリを使用）"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.store.DB_PATH", tmpdir), \
             patch("src.store.COLLECTION_NAME", "test_images"):
            from src.store import ImageStore
            yield ImageStore(reset=True)


def _dummy_embedding(dim: int = 512) -> list[float]:
    return [0.0] * dim


def test_add_and_query(store):
    emb = _dummy_embedding()
    store.add(
        ids=["img_001"],
        embeddings=[emb],
        metadatas=[{"caption": "a cat", "path": "/tmp/cat.jpg"}],
    )
    results = store.query(emb, n_results=1)
    assert len(results["metadatas"][0]) == 1
    assert results["metadatas"][0][0]["caption"] == "a cat"


def test_get_all_returns_added_items(store):
    emb = _dummy_embedding()
    store.add(
        ids=["img_001", "img_002"],
        embeddings=[emb, emb],
        metadatas=[
            {"caption": "a cat", "path": "/tmp/cat.jpg"},
            {"caption": "a dog", "path": "/tmp/dog.jpg"},
        ],
    )
    result = store.get_all()
    assert len(result["ids"]) == 2


def test_query_n_results(store):
    emb = _dummy_embedding()
    store.add(
        ids=["img_001", "img_002", "img_003"],
        embeddings=[emb, emb, emb],
        metadatas=[
            {"caption": f"image {i}", "path": f"/tmp/img_{i}.jpg"}
            for i in range(3)
        ],
    )
    results = store.query(emb, n_results=2)
    assert len(results["metadatas"][0]) == 2
