# Semantic Image Search

Semantic image search over the KITTI driving dataset using CLIP or SigLIP and ChromaDB.

Pre-compute image embeddings offline once, then search at query time using only a fast text or image embedding — making semantic search practical without a GPU server.

---

## Screenshots

![Parked cars search](assets/image-search_parked-cars.png)

![Urban street search](assets/image-search_urban-street.png)

![UMAP scatter plot](assets/scatter-plot_umap.png)

---

## Architecture

```
[KITTI dataset (training/image_2/*.png)]
       │
       ▼  run once (scripts/index_kitti.py --model <model>)
  CLIP (512-dim) or SigLIP (768-dim) embedding
       │
       ▼
  ChromaDB  ←  persisted locally (db/)
             ※ separate collection per model (images_clip / images_siglip)
       │
       ▼  at query time (app.py / search.py)
  Text / Image Query → Embedding → Cosine Similarity Search → Results
```

---

## Features

- **Text-to-image search** — describe a driving scene in natural language and retrieve visually similar images
- **Image-to-image search** — upload an image and find similar frames from the index
- **Model switching** — switch between CLIP and SigLIP via the Gradio UI or CLI
- **UMAP scatter plot** — visualise the distribution of image embeddings interactively
- **Apple Silicon support** — automatically uses MPS backend when available
- **Gradio web UI** — browser-based interface with gallery view and pagination

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the KITTI dataset

Extract `data_object_image_2.zip` so that the directory structure is:

```
data/
├── training/
│   └── image_2/
│       ├── 000000.png
│       ├── 000001.png
│       └── ...
└── testing/
    └── image_2/
        └── ...
```

### 3. Build the vector index (run once per model)

```bash
# CLIP (default)
python scripts/index_kitti.py

# SigLIP (run separately to build a second index)
python scripts/index_kitti.py --model siglip
```

Each model's embeddings are stored in a separate ChromaDB collection (`images_clip` / `images_siglip`).

### 4. Search

```bash
# Web UI → http://localhost:7860
python app.py

# CLI with CLIP (default)
python search.py "a car driving on the road"

# CLI with SigLIP
python search.py --model siglip "pedestrian walking on sidewalk"
```

In the web UI, use the radio buttons at the top to switch models. If the selected model's index is empty, a warning is shown.

### 5. Visualise embedding distribution (optional)

```bash
python scripts/analyze.py
# → generates embedding_map.html
```

Runs UMAP to compress all image embeddings to 2D and renders an interactive scatter plot (`embedding_map.html`). Each dot represents one image frame. Hover over a point to see the scene name; click to display the full image.

```bash
cd /path/to/image-search-project
python -m http.server 8080
# → http://localhost:8080/embedding_map.html
```

---

## Project Structure

```
image-search-project/
├── src/                        # Core package
│   ├── config.py               # Paths and settings (edit here)
│   ├── embedder.py             # BaseEmbedder abstract class + get_embedder() factory
│   ├── clip_embedder.py        # CLIPEmbedder (512-dim)
│   ├── siglip_embedder.py      # SiglipEmbedder (768-dim)
│   └── store.py                # ChromaDB wrapper
├── scripts/
│   ├── index_kitti.py          # Build vector index (supports --model)
│   └── analyze.py              # UMAP visualisation
├── tests/
│   ├── test_embedder.py        # Unit tests for CLIP and SigLIP embedders
│   └── test_store.py
├── app.py                      # Gradio web UI
├── search.py                   # CLI search (supports --model)
├── README.md                   # Japanese README
├── README_en.md                # This file
├── data/                       # KITTI dataset (gitignored)
├── db/                         # ChromaDB index (gitignored)
└── requirements.txt
```

---

## Model Comparison

| Model | Embedding dim | Size | Notes |
|-------|--------------|------|-------|
| CLIP (`openai/clip-vit-base-patch32`) | 512 | ~150 MB | Lightweight, fast. Default. |
| SigLIP (`google/siglip-base-patch16-224`) | 768 | ~400 MB | Trained with sigmoid loss; generally stronger than CLIP. Requires `sentencepiece`. |

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

## References

- [CLIP on HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32)
- [SigLIP on HuggingFace](https://huggingface.co/google/siglip-base-patch16-224)
- [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/)
- [ChromaDB documentation](https://docs.trychroma.com/)
- [UMAP documentation](https://umap-learn.readthedocs.io/)

Japanese README: [README.md](README.md)
