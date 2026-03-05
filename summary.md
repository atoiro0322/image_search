# clip-image-search — プロジェクト引き継ぎ Summary

## プロジェクト概要

CLIPモデル + ChromaDB を使った**セマンティック画像検索**の試作プロジェクト。  
[チューリング社の運転動画検索システム](https://zenn.dev/turing_motors/articles/cf86e1bf0e41ca) を参考に、MacBookで動く軽量版として実装。

**コアコンセプト（記事と同じ非対称アーキテクチャ）:**
- 画像は**事前にベクトル化**してChromaDBに保存（重い処理は一度だけ）
- 検索時は**テキスト/画像クエリのみリアルタイム**でベクトル化（軽い）

---

## ディレクトリ構成

```
clip-image-search/
├── setup_and_index.py   # 画像ダウンロード＆ChromaDBへのインデックス作成
├── app.py               # Gradio WebUI（テキスト検索 + 画像アップロード検索）
├── search.py            # CLI版の検索スクリプト
├── requirements.txt     # 依存ライブラリ
├── README.md            # セットアップ手順
├── SUMMARY.md           # このファイル
├── chroma_db/           # ChromaDBのデータ（setup後に生成）
└── sample_images/       # サンプル画像（setup後に生成）
    ├── img_001.jpg
    ├── ...
    └── metadata.json
```

---

## 技術スタック

| 役割 | ライブラリ |
|------|-----------|
| 画像・テキストのEmbedding | `openai/clip-vit-base-patch32`（HuggingFace） |
| ベクトルDB | ChromaDB（ローカル永続化） |
| WebUI | Gradio |
| GPU加速 | PyTorch MPS（Apple Silicon）/ CPU fallback |

---

## セットアップ手順

```bash
# 1. 依存ライブラリのインストール
pip install -r requirements.txt

# 2. 画像ダウンロード＆インデックス作成（初回のみ）
python setup_and_index.py

# 3. WebUI起動 → http://localhost:7860
python app.py

# または CLI で検索
python search.py "a dog running in the park"
```

---

## 実装済みの機能

### `setup_and_index.py`
- Unsplashのサンプル画像10枚をダウンロード
- CLIPでベクトル化（512次元）してChromaDBに登録
- コサイン類似度空間でインデックス構築

### `app.py`（Gradio WebUI）
- **テキスト検索タブ**: 英語テキスト → 類似画像をギャラリー表示
- **画像検索タブ**: 画像アップロード → 類似画像をギャラリー表示
- 表示件数スライダー（1〜6件）
- クエリ例のExamplesウィジェット

### `search.py`（CLI）
- コマンドライン引数でテキスト検索
- 類似度スコアとキャプションを表示

---

## 次に実装予定：`analyze.py`（散布図可視化）

### 目的
収集データの**分布を可視化**して冗長なデータを把握する。  
ChromaDBに保存済みのEmbeddingをそのまま流用するため、追加の重い計算は不要。

### 実装方針

```
ChromaDB から全Embeddingを取得（512次元）
        ↓
UMAP で2次元に圧縮
        ↓
Plotly でインタラクティブ散布図を表示
  - 点にホバーすると画像サムネイルとキャプションが表示
  - 密集エリア = 冗長なデータが多い
  - 疎なエリア = 希少・多様なシーン
```

### 必要な追加ライブラリ

```
umap-learn
plotly
```

### `analyze.py` の実装イメージ

```python
import chromadb
import umap
import plotly.express as px
import numpy as np

DB_PATH = "./chroma_db"
COLLECTION_NAME = "images"

def visualize():
    # 1. ChromaDBから全Embeddingを取得
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    result = collection.get(include=["embeddings", "metadatas"])

    embeddings = np.array(result["embeddings"])   # shape: (N, 512)
    metadatas  = result["metadatas"]              # caption, path のリスト
    ids        = result["ids"]

    # 2. UMAPで2次元に圧縮
    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
    coords = reducer.fit_transform(embeddings)    # shape: (N, 2)

    # 3. Plotlyで散布図
    captions = [m["caption"] for m in metadatas]
    paths    = [m["path"]    for m in metadatas]

    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        hover_name=captions,
        text=ids,
        title="画像Embeddingの分布（UMAP 2D）",
    )
    fig.update_traces(marker=dict(size=12))
    fig.show()   # ブラウザで開く

if __name__ == "__main__":
    visualize()
```

### 発展的な拡張アイデア（優先度順）

1. **サムネイル表示**: ホバー時に画像をポップアップ表示（Plotly の customdata + hovertemplate を使用）
2. **クラスタリング色分け**: HDBSCANでクラスタリングして点を色分け → どのシーンが固まっているか一目でわかる
3. **Gradio UIに統合**: `app.py` にタブとして追加（散布図タブを追加）
4. **密度ヒートマップ**: 散布図に密度ヒートマップを重ねて冗長エリアを強調

---

## 参考リンク

- [チューリング社の記事（Cosmos-Embed1 + Databricks）](https://zenn.dev/turing_motors/articles/cf86e1bf0e41ca)
- [CLIP モデル（HuggingFace）](https://huggingface.co/openai/clip-vit-base-patch32)
- [UMAP ドキュメント](https://umap-learn.readthedocs.io/)
- [ChromaDB ドキュメント](https://docs.trychroma.com/)
- [Plotly Scatter](https://plotly.com/python/line-and-scatter/)