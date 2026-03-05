"""
CLIP + ChromaDB によるセマンティック画像検索
Step 4: 保存済みEmbeddingをUMAPで2次元圧縮し、Plotlyで散布図表示
"""

import base64
import io

import chromadb
import numpy as np
import plotly.graph_objects as go
import umap
from PIL import Image

DB_PATH = "./chroma_db"
COLLECTION_NAME = "images"


def image_to_base64(path: str, size: tuple = (120, 90)) -> str:
    """画像をリサイズしてBase64エンコードされたHTML img タグ用URIに変換"""
    img = Image.open(path).convert("RGB")
    img.thumbnail(size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def visualize():
    # 1. ChromaDBから全Embeddingを取得
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    result = collection.get(include=["embeddings", "metadatas"])

    embeddings = np.array(result["embeddings"])  # shape: (N, 512)
    metadatas = result["metadatas"]              # caption, path のリスト
    ids = result["ids"]
    n = len(ids)
    print(f"取得したEmbedding: {n} 件")

    # 2. UMAPで2次元に圧縮
    print("UMAPで次元削減中...")
    n_neighbors = min(15, n - 1)  # データ件数が少ない場合の安全対策
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42, metric="cosine")
    coords = reducer.fit_transform(embeddings)  # shape: (N, 2)

    # 3. ホバー用のBase64サムネイルを準備
    print("サムネイルを生成中...")
    captions = [m["caption"] for m in metadatas]
    paths = [m["path"] for m in metadatas]
    thumbnails = [image_to_base64(p) for p in paths]

    # 4. Plotlyでインタラクティブ散布図
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers+text",
        text=ids,
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=14, color="steelblue", opacity=0.8, line=dict(width=1, color="white")),
        customdata=list(zip(captions, thumbnails)),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "%{customdata[0]}<br>"
            "<img src='%{customdata[1]}' width='120'><br>"
            "UMAP: (%{x:.2f}, %{y:.2f})"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text="画像Embeddingの分布（UMAP 2D）", font=dict(size=18)),
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        width=900,
        height=650,
        hoverlabel=dict(bgcolor="white", font_size=12),
        plot_bgcolor="rgba(245,245,245,1)",
    )

    output_path = "embedding_map.html"
    fig.write_html(output_path)
    print(f"散布図を保存しました: {output_path}")
    fig.show()


if __name__ == "__main__":
    visualize()
