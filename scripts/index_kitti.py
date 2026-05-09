"""
KITTIデータセット（data_object_image_2）を
指定モデルでベクトル化してChromaDBに登録する

使い方:
  python scripts/index_kitti.py                  # CLIPで登録（デフォルト）
  python scripts/index_kitti.py --model siglip   # SigLIPで登録
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from src.config import AVAILABLE_MODELS, DEFAULT_MODEL, KITTI_DIR_PATH, KITTI_MAX_IMAGES, KITTI_SPLIT
from src.embedder import get_embedder
from src.store import ImageStore

BATCH_SIZE = 64


def iter_images(kitti_dir: Path, split: str):
    image_dir = kitti_dir / split / "image_2"
    if not image_dir.exists():
        raise FileNotFoundError(f"ディレクトリが見つかりません: {image_dir}")
    paths = sorted(image_dir.glob("*.png"))

    def _gen():
        for path in paths:
            yield f"{split}_{path.stem}", str(path), Image.open(path).convert("RGB")

    return paths, _gen()


def _progress_bar(count: int, total: int, width: int = 35) -> str:
    pct = count / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"\r[{bar}] {pct:5.1%}  ({count}/{total})"


def index_kitti(paths, source, split: str, max_images: int | None, model_key: str):
    print(f"モデルをロード中: {model_key} ...")
    embedder = get_embedder(model_key)

    print("ChromaDBを初期化中...")
    store = ImageStore(model_key=model_key, reset=True)

    total = min(len(paths), max_images) if max_images else len(paths)
    ids, embeddings, metadatas = [], [], []
    count = 0

    print(f"\n画像をベクトル化してインデックス登録中 (model={model_key}, split={split}, 計{total}枚)")

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
            print(_progress_bar(count, total), end="", flush=True)

            if len(ids) >= BATCH_SIZE:
                store.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                ids, embeddings, metadatas = [], [], []

        except Exception as e:
            print(f"\n  ✗ {img_id}: {e}")

    if ids:
        store.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    print(f"\n✅ {count} 枚の画像をインデックスに登録しました（モデル: {model_key}）")
    print("次は search.py または app.py で検索できます")


def main():
    parser = argparse.ArgumentParser(description="KITTIデータセットをインデックス登録する")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"使用する埋め込みモデル（デフォルト: {DEFAULT_MODEL}）",
    )
    args = parser.parse_args()

    if KITTI_DIR_PATH is None:
        raise ValueError("src/config.py の KITTI_DIR_PATH を設定してください")
    paths, source = iter_images(KITTI_DIR_PATH, KITTI_SPLIT)
    index_kitti(paths, source, KITTI_SPLIT, KITTI_MAX_IMAGES, args.model)


if __name__ == "__main__":
    main()
