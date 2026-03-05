"""
CLIP + ChromaDB によるセマンティック画像検索
Step 2: テキストクエリで画像を検索する (CLI版)
"""

import sys

from src.embedder import CLIPEmbedder
from src.store import ImageStore


def search(query: str, top_k: int = 3):
    embedder = CLIPEmbedder()
    store = ImageStore()

    emb = embedder.get_text_embedding(query)

    results = store.query(emb, n_results=top_k)

    print(f"\n🔍 クエリ: '{query}'")
    print(f"{'─' * 50}")
    for i, (meta, dist) in enumerate(zip(
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = 1 - dist
        print(f"#{i+1}  類似度: {similarity:.3f}")
        print(f"    キャプション: {meta['caption']}")
        print(f"    ファイル: {meta['path']}")
        print()


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "a dog running outside"
    search(query)
