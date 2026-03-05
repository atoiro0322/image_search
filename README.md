# clip-image-search

Semantic image search over the KITTI driving dataset using CLIP and ChromaDB.

Pre-compute image embeddings offline once, then search at query time using only a fast text or image embedding — making semantic search practical without a GPU server.

---

## Architecture

```
[KITTI dataset (training/image_2/*.png)]
       │
       ▼  run once (scripts/index_kitti.py)
  CLIP Embedding (512-dim)
       │
       ▼
  ChromaDB  ←  persisted locally (data/chroma_db/)
       │
       ▼  at query time (app.py / search.py)
  Text / Image Query → CLIP → Cosine Similarity Search → Results
```

---

## Features

- **Text-to-image search** — describe a driving scene in natural language and retrieve visually similar images
- **Image-to-image search** — upload an image and find similar frames from the index
- **UMAP scatter plot** — visualise the distribution of image embeddings to identify redundant or underrepresented scenes
- **Apple Silicon support** — automatically uses MPS backend when available
- **Gradio web UI** — browser-based interface with gallery view

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the KITTI dataset

Extract `data_object_image_2.zip` so that the directory structure is:

```
kitti_dataset/
└── training/
    └── image_2/
        ├── 000000.png
        ├── 000001.png
        └── ...
```

### 3. Configure paths

Edit `src/config.py`:

```python
KITTI_DIR_PATH   = BASE_DIR / "kitti_dataset"  # path to extracted dataset
KITTI_SPLIT      = "training"                  # "training" or "testing"
KITTI_MAX_IMAGES = None                        # None = all, int = limit (for testing)
```

### 4. Build the vector index (run once)

```bash
python scripts/index_kitti.py
```

### 5. Search

```bash
# Web UI → http://localhost:7860
python app.py

# CLI
python search.py "a car driving on the road"
python search.py "pedestrian walking on sidewalk"
```

### 6. Visualise embedding distribution (optional)

```bash
python scripts/analyze.py
# → embedding_map.html を生成してブラウザで表示
```

---

## Project Structure

```
image-search-project/
├── src/                        # Core package
│   ├── config.py               # Paths and settings (edit here)
│   ├── embedder.py             # CLIP wrapper
│   └── store.py                # ChromaDB wrapper
├── scripts/
│   ├── index_kitti.py          # Build vector index from KITTI dataset
│   └── analyze.py              # UMAP visualisation
├── tests/
│   ├── test_embedder.py
│   └── test_store.py
├── app.py                      # Gradio web UI
├── search.py                   # CLI search
├── kitti_dataset/              # Place extracted KITTI data here (gitignored)
├── data/                       # Generated index (gitignored)
│   └── chroma_db/
└── requirements.txt
```

---

## Query Examples

| Query | Expected results |
|-------|-----------------|
| `a car driving on the road` | Moving vehicles on highway/street |
| `pedestrian walking on sidewalk` | Scenes with people on foot |
| `intersection with traffic` | Complex road junctions |
| `highway with multiple lanes` | Wide road scenes |
| `parked cars on street` | Stationary vehicles |
| `urban street scene` | City driving environments |

---

## Requirements

- Python 3.10+
- KITTI object detection dataset (`data_object_image_2.zip`)
- Apple Silicon recommended (MPS acceleration), but CPU-only also works

---

## Background

A common challenge in autonomous driving dataset curation is that collections tend to be heavily skewed — similar scenes (e.g. straight highway driving) are over-represented while rare or diverse examples (intersections, pedestrians, adverse weather) are underrepresented. This project uses CLIP embeddings and vector search as a foundation for:

1. **Semantic search** — find frames matching a natural language description
2. **Distribution visualisation** — identify which scene types are over- or under-represented
3. *(future)* **Deduplication** — remove near-duplicate frames to improve dataset diversity

---

## References

- [CLIP model on HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32)
- [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/)
- [ChromaDB documentation](https://docs.trychroma.com/)
- [UMAP documentation](https://umap-learn.readthedocs.io/)
- [Plotly Scatter](https://plotly.com/python/line-and-scatter/)