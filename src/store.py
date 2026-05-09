import chromadb

from .config import COLLECTION_NAME, DB_PATH


class ImageStore:
    def __init__(self, model_key: str = "clip", reset: bool = False):
        self._client = chromadb.PersistentClient(path=DB_PATH)
        collection_name = f"{COLLECTION_NAME}_{model_key}"
        if reset:
            try:
                self._client.delete_collection(collection_name)
            except Exception:
                pass
            self._col = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._col = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def add(self, ids: list[str], embeddings: list, metadatas: list[dict]) -> None:
        self._col.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query(self, embedding: list[float], n_results: int = 4) -> dict:
        return self._col.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

    def get_all(self) -> dict:
        return self._col.get(include=["embeddings", "metadatas"])

    def count(self) -> int:
        return self._col.count()
