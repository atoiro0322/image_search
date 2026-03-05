import chromadb

from .config import COLLECTION_NAME, DB_PATH


class ImageStore:
    def __init__(self, reset: bool = False):
        self._client = chromadb.PersistentClient(path=DB_PATH)
        if reset:
            try:
                self._client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self._col = self._client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._col = self._client.get_collection(COLLECTION_NAME)

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
