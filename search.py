"""
CLIP + ChromaDB によるセマンティック画像検索
Step 2: テキストクエリで画像を検索する (CLI版)
"""

import sys
import chromadb
import torch
from transformers import CLIPProcessor, CLIPModel

DB_PATH = "./chroma_db"
COLLECTION_NAME = "images"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

def search(query: str, top_k: int = 3):
    model, processor = load_model()
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    # テキストをベクトル化
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        emb = model.get_text_features(**inputs).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().float().numpy()[0].tolist()

    # ベクトル検索
    results = collection.query(
        query_embeddings=[emb],
        n_results=top_k,
        include=["metadatas", "distances"]
    )

    print(f"\n🔍 クエリ: '{query}'")
    print(f"{'─' * 50}")
    for i, (meta, dist) in enumerate(zip(
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = 1 - dist  # cosine距離 → 類似度
        print(f"#{i+1}  類似度: {similarity:.3f}")
        print(f"    キャプション: {meta['caption']}")
        print(f"    ファイル: {meta['path']}")
        print()

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "a dog running outside"
    search(query)
