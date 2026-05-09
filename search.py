"""
セマンティック画像検索 — CLI版

使い方:
  python search.py "a car driving on the road"
  python search.py --model siglip "pedestrian walking on sidewalk"
"""

import argparse

from src.config import AVAILABLE_MODELS, DEFAULT_MODEL
from src.embedder import get_embedder
from src.store import ImageStore


def search(query: str, model_key: str = DEFAULT_MODEL, top_k: int = 3):
    embedder = get_embedder(model_key)
    store = ImageStore(model_key=model_key)

    count = store.count()
    if count == 0:
        print(f"⚠️ '{model_key}' のインデックスが空です。")
        print(f"   先に `python scripts/index_kitti.py --model {model_key}` を実行してください。")
        return

    emb = embedder.get_text_embedding(query)
    results = store.query(emb, n_results=top_k)

    print(f"\n🔍 クエリ: '{query}'  ［モデル: {model_key}］")
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
    parser = argparse.ArgumentParser(description="セマンティック画像検索")
    parser.add_argument("query", nargs="?", default="a dog running outside", help="検索クエリ（英語）")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"使用する埋め込みモデル（デフォルト: {DEFAULT_MODEL}）",
    )
    parser.add_argument("--top-k", type=int, default=3, help="取得件数")
    args = parser.parse_args()
    search(args.query, args.model, args.top_k)
