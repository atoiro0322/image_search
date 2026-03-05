"""
CLIP + ChromaDB によるセマンティック画像検索
Step 1: サンプル画像をダウンロードしてベクトルDBに登録する
"""

import os
import json
import urllib.request
from pathlib import Path

import chromadb
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ---- 設定 ----
DB_PATH = "./chroma_db"
COLLECTION_NAME = "images"
IMAGE_DIR = "./sample_images"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---- サンプル画像 (Unsplash public domain) ----
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
    Path(IMAGE_DIR).mkdir(exist_ok=True)
    print("サンプル画像をダウンロード中...")
    for item in SAMPLE_IMAGES:
        path = f"{IMAGE_DIR}/{item['id']}.jpg"
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(item["url"], path)
                print(f"  ✓ {item['id']}")
            except Exception as e:
                print(f"  ✗ {item['id']}: {e}")
        else:
            print(f"  (skip) {item['id']}")


def build_index():
    print("\nCLIPモデルをロード中...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    print("ChromaDBを初期化中...")
    client = chromadb.PersistentClient(path=DB_PATH)
    # 既存コレクションを削除して再作成
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("\n画像をベクトル化してインデックス登録中...")
    ids, embeddings, metadatas = [], [], []

    for item in SAMPLE_IMAGES:
        path = f"{IMAGE_DIR}/{item['id']}.jpg"
        if not os.path.exists(path):
            print(f"  スキップ (ファイルなし): {item['id']}")
            continue
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                emb = model.get_image_features(**inputs).pooler_output
                emb = emb / emb.norm(dim=-1, keepdim=True)  # L2正規化
                emb = emb.cpu().float().numpy()[0].tolist()

            ids.append(item["id"])
            embeddings.append(emb)
            metadatas.append({"caption": item["caption"], "path": path})
            print(f"  ✓ {item['id']}: {item['caption'][:40]}...")
        except Exception as e:
            print(f"  ✗ {item['id']}: {e}")

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    print(f"\n✅ {len(ids)} 件の画像をインデックスに登録しました")

    # メタデータ保存
    with open(f"{IMAGE_DIR}/metadata.json", "w") as f:
        json.dump(SAMPLE_IMAGES, f, ensure_ascii=False, indent=2)
    print(f"メタデータを {IMAGE_DIR}/metadata.json に保存しました")


if __name__ == "__main__":
    download_images()
    build_index()
    print("\n完了！次は search.py または app.py で検索できます")
