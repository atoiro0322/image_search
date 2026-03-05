import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
DB_PATH = str(DATA_DIR / "chroma_db")

COLLECTION_NAME = "images"
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---- KITTI 設定 ----
KITTI_DIR_PATH: Path | None = BASE_DIR / "kitti_dataset"  # 展開済みディレクトリのパス

KITTI_SPLIT = "training"   # "training" or "testing"
KITTI_MAX_IMAGES: int | None = None  # None = 全件, 整数 = 上限枚数（動作確認用）
