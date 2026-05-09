import os
import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
DB_PATH = str(BASE_DIR / "db")

COLLECTION_NAME = "images"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

DEFAULT_MODEL: str = os.getenv("EMBEDDING_MODEL", "clip")
AVAILABLE_MODELS: list[str] = ["clip", "siglip"]

# ---- KITTI 設定 ----
KITTI_DIR_PATH: Path | None = DATA_DIR
KITTI_SPLIT = "training"
KITTI_MAX_IMAGES: int | None = None
