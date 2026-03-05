"""
CLIP + ChromaDB によるセマンティック画像検索
Step 1: サンプル画像をダウンロードしてベクトルDBに登録する
"""

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from src.config import IMAGE_DIR
from src.embedder import CLIPEmbedder
from src.store import ImageStore

SAMPLE_IMAGES = [
    {
        "id": "img_001",
        "url": "https://images.unsplash.com/photo-1544985361-b420d7a77043?w=400",
        "caption": "A cat sitting on a window sill looking outside"
    },
    {
        "id": "img_002",
        "url": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400",
        "caption": "A golden retriever dog running in a park"
    },
    {
        "id": "img_003",
        "url": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400",
        "caption": "City street at night with bright lights"
    },
    {
        "id": "img_004",
        "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
        "caption": "Mountain landscape with snow peaks"
    },
    {
        "id": "img_005",
        "url": "https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?w=400",
        "caption": "Ocean beach at sunset with waves"
    },
    {
        "id": "img_006",
        "url": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=400",
        "caption": "Coffee cup on a wooden table"
    },
    {
        "id": "img_007",
        "url": "https://images.unsplash.com/photo-1498050108023-c5249f4df085?w=400",
        "caption": "Person working on a laptop computer"
    },
    {
        "id": "img_008",
        "url": "https://images.unsplash.com/photo-1465146344425-f00d5f5c8f07?w=400",
        "caption": "Colorful flowers in a garden"
    },
    {
        "id": "img_009",
        "url": "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=400",
        "caption": "Food dish at a restaurant"
    },
    {
        "id": "img_010",
        "url": "https://images.unsplash.com/photo-1543269865-cbf427effbad?w=400",
        "caption": "People having a conversation"
    },
]


def download_images():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("サンプル画像をダウンロード中...")
    for item in SAMPLE_IMAGES:
        path = IMAGE_DIR / f"{item['id']}.jpg"
        if not path.exists():
            try:
                urllib.request.urlretrieve(item["url"], path)
                print(f"  ✓ {item['id']}")
            except Exception as e:
                print(f"  ✗ {item['id']}: {e}")
        else:
            print(f"  (skip) {item['id']}")


def build_index():
    print("\nCLIPモデルをロード中...")
    embedder = CLIPEmbedder()

    print("ChromaDBを初期化中...")
    store = ImageStore(reset=True)

    print("\n画像をベクトル化してインデックス登録中...")
    ids, embeddings, metadatas = [], [], []

    for item in SAMPLE_IMAGES:
        path = IMAGE_DIR / f"{item['id']}.jpg"
        if not path.exists():
            print(f"  スキップ (ファイルなし): {item['id']}")
            continue
        try:
            image = Image.open(path).convert("RGB")
            emb = embedder.get_image_embedding(image)
            ids.append(item["id"])
            embeddings.append(emb)
            metadatas.append({"caption": item["caption"], "path": str(path)})
            print(f"  ✓ {item['id']}: {item['caption'][:40]}...")
        except Exception as e:
            print(f"  ✗ {item['id']}: {e}")

    store.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    print(f"\n✅ {len(ids)} 件の画像をインデックスに登録しました")

    with open(IMAGE_DIR / "metadata.json", "w") as f:
        json.dump(SAMPLE_IMAGES, f, ensure_ascii=False, indent=2)
    print(f"メタデータを {IMAGE_DIR}/metadata.json に保存しました")


if __name__ == "__main__":
    download_images()
    build_index()
    print("\n完了！次は search.py または app.py で検索できます")
