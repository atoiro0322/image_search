"""
KITTI データセット（data_object_image_2）を
CLIPでベクトル化してChromaDBに登録する

設定は src/config.py の KITTI_* 変数で行う:
  KITTI_DIR_PATH  : 展開済みKITTIディレクトリのパス
  KITTI_SPLIT     : "training" or "testing"
  KITTI_MAX_IMAGES: 登録上限枚数（None = 全件）

使い方:
  python scripts/index_kitti.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from src.config import KITTI_DIR_PATH, KITTI_MAX_IMAGES, KITTI_SPLIT
from src.embedder import CLIPEmbedder
from src.store import ImageStore

BATCH_SIZE = 64  # ChromaDB への一括登録サイズ


def iter_images(kitti_dir: Path, split: str):
    """展開済みディレクトリから (img_id, image_path_str, PIL.Image) を yield"""
    image_dir = kitti_dir / split / "image_2"
    if not image_dir.exists():
        raise FileNotFoundError(f"ディレクトリが見つかりません: {image_dir}")
    paths = sorted(image_dir.glob("*.png"))
    for path in paths:
        yield f"{split}_{path.stem}", str(path), Image.open(path).convert("RGB")


# ---- メイン処理 ----

def index_kitti(source, split: str, max_images: int | None):
    print("CLIPモデルをロード中...")
    embedder = CLIPEmbedder()

    print("ChromaDBを初期化中...")
    store = ImageStore(reset=True)

    ids, embeddings, metadatas = [], [], []
    count = 0

    print(f"\n画像をベクトル化してインデックス登録中 (split={split})...")

    for img_id, path_str, image in source:
        if max_images and count >= max_images:
            break
        try:
            emb = embedder.get_image_embedding(image)
            ids.append(img_id)
            embeddings.append(emb)
            metadatas.append({
                "path": path_str,
                "caption": f"KITTI driving scene {Path(path_str).stem}",
                "split": split,
            })
            count += 1

            # バッチ登録
            if len(ids) >= BATCH_SIZE:
                store.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                print(f"  [{count}] 登録済み...")
                ids, embeddings, metadatas = [], [], []

        except Exception as e:
            print(f"  ✗ {img_id}: {e}")

    # 残りを登録
    if ids:
        store.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    print(f"\n✅ {count} 枚の画像をインデックスに登録しました")
    print("次は search.py または app.py で検索できます")


def main():
    if KITTI_DIR_PATH is None:
        raise ValueError("src/config.py の KITTI_DIR_PATH を設定してください")
    source = iter_images(KITTI_DIR_PATH, KITTI_SPLIT)
    index_kitti(source, KITTI_SPLIT, KITTI_MAX_IMAGES)


if __name__ == "__main__":
    main()
