"""
CLIP + ChromaDB によるセマンティック画像検索
Step 4: 保存済みEmbeddingをUMAPで2次元圧縮し、Plotlyで散布図表示
"""

import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
import umap
from PIL import Image

from src.store import ImageStore

OUTPUT_PATH = Path(__file__).parent.parent / "embedding_map.html"


def image_to_base64(path: str, size: tuple) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def visualize():
    # 1. ChromaDBから全Embeddingを取得
    store = ImageStore()
    result = store.get_all()

    embeddings = np.array(result["embeddings"])  # shape: (N, 512)
    metadatas = result["metadatas"]
    ids = result["ids"]
    n = len(ids)
    print(f"取得したEmbedding: {n} 件")

    # 2. UMAPで2次元に圧縮
    print("UMAPで次元削減中...")
    n_neighbors = min(15, n - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42, metric="cosine")
    coords = reducer.fit_transform(embeddings)  # shape: (N, 2)

    # 3. サムネイル（ホバー用小・クリック用中）を準備
    print("サムネイルを生成中...")
    captions = [m["caption"] for m in metadatas]
    paths = [m["path"] for m in metadatas]
    thumbs_medium = [image_to_base64(p, size=(320, 240)) for p in paths]

    # 4. Plotlyでインタラクティブ散布図（テキストラベルなし）
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers",
        marker=dict(size=10, color="steelblue", opacity=0.8, line=dict(width=1, color="white")),
        customdata=list(zip(captions, thumbs_medium, ids)),
        hovertemplate=(
            "<b>%{customdata[2]}</b><br>"
            "%{customdata[0]}<br>"
            "UMAP: (%{x:.2f}, %{y:.2f})"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        height=600,
        hoverlabel=dict(bgcolor="white", font_size=12),
        plot_bgcolor="rgba(245,245,245,1)",
        margin=dict(l=40, r=10, t=20, b=40),
    )

    # 5. カスタムHTMLでクリック時に下パネルへ画像表示
    chart_div = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id="scatter-plot",
    )

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>画像Embedding分布</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: sans-serif; background: #f5f5f5; padding: 16px 20px 32px; }}
    h1 {{ font-size: 18px; padding-bottom: 12px; color: #333; }}
    #chart-col {{ background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-bottom: 16px; }}
    #info-panel {{
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,.1);
      padding: 16px;
      display: flex;
      gap: 16px;
      align-items: flex-start;
    }}
    #placeholder {{ color: #aaa; font-size: 13px; text-align: center; line-height: 1.8; width: 100%; padding: 12px 0; }}
    #info-img {{ max-width: 480px; width: 100%; border-radius: 6px; display: none; box-shadow: 0 2px 8px rgba(0,0,0,.15); flex-shrink: 0; }}
    #info-text {{ display: none; flex-direction: column; gap: 6px; }}
    #info-title {{ font-weight: bold; font-size: 14px; color: #222; }}
    #info-caption {{ font-size: 13px; color: #555; line-height: 1.5; }}
  </style>
</head>
<body>
  <h1>画像Embeddingの分布（UMAP 2D）</h1>
  <div id="chart-col">{chart_div}</div>
  <div id="info-panel">
    <div id="placeholder">↑ 点をクリックすると画像が表示されます</div>
    <img id="info-img" src="" alt="">
    <div id="info-text">
      <div id="info-title"></div>
      <div id="info-caption"></div>
    </div>
  </div>
  <script>
    document.getElementById('scatter-plot').on('plotly_click', function(data) {{
      var pt = data.points[0];
      document.getElementById('placeholder').style.display = 'none';
      var img = document.getElementById('info-img');
      img.src = pt.customdata[1];
      img.style.display = 'block';
      var txt = document.getElementById('info-text');
      txt.style.display = 'flex';
      document.getElementById('info-title').textContent = pt.customdata[2];
      document.getElementById('info-caption').textContent = pt.customdata[0];
    }});
  </script>
</body>
</html>"""

    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"散布図を保存しました: {OUTPUT_PATH}")

    import webbrowser
    webbrowser.open(OUTPUT_PATH.as_uri())


if __name__ == "__main__":
    visualize()
