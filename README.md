# clip-image-search

A lightweight semantic image search prototype using CLIP and ChromaDB, designed to run locally on a MacBook.

Pre-compute image embeddings offline once, then search at query time using only a fast text or image embedding — making semantic search practical without a GPU server.

---

## Architecture

```
[Image Collection]
       │
       ▼  run once (setup_and_index.py)
  CLIP Embedding (512-dim)
       │
       ▼
  ChromaDB  ←  persisted locally
       │
       ▼  at query time (app.py)
  Text / Image Query → CLIP → Cosine Similarity Search → Results
```

---

## Features

- **Text-to-image search** — describe a scene in natural language and retrieve visually similar images
- **Image-to-image search** — upload an image and find similar ones from the index
- **UMAP scatter plot** *(in progress)* — visualise the distribution of image embeddings to identify redundant or underrepresented scenes
- **Apple Silicon support** — automatically uses MPS backend when available
- **Gradio web UI** — browser-based interface with gallery view

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download sample images and build the vector index (run once)
python setup_and_index.py

# 3. Launch the web UI → http://localhost:7860
python app.py

# Or search from the command line
python search.py "a dog running in the park"
```

---

## Project Structure

```
clip-image-search/
├── setup_and_index.py   # Download images and build ChromaDB index
├── app.py               # Gradio web UI (text search + image search)
├── search.py            # CLI search script
├── analyze.py           # (planned) UMAP visualisation of embedding distribution
├── requirements.txt
├── chroma_db/           # Generated after running setup_and_index.py
└── sample_images/       # Generated after running setup_and_index.py
```

---

## Using Your Own Images

Replace the `SAMPLE_IMAGES` list in `setup_and_index.py` with your own data:

```python
SAMPLE_IMAGES = [
    {
        "id": "my_img_001",
        "url": "",                        # leave empty for local files
        "caption": "description of image"
    },
    ...
]
```

Or modify `build_index()` to load images directly from a local directory.

---

## Requirements

- Python 3.10+
- MacBook with Apple Silicon recommended (MPS acceleration), but CPU-only also works

```
torch
transformers
chromadb
Pillow
gradio
```

---

## Planned Features

- `analyze.py` — interactive UMAP scatter plot (Plotly) with hover thumbnails to visualise dataset distribution and spot redundant scenes
- HDBSCAN clustering with colour-coded scatter plot
- Integration of scatter plot as a tab in the Gradio UI
- Support for large-scale driving datasets (KITTI, WayveScenes101, nuScenes)

---

## Background

A common challenge in image dataset curation is that collections tend to be heavily skewed — similar scenes are over-represented while rare or diverse examples are underrepresented. This project explores using CLIP embeddings and vector search as a foundation for:

1. **Semantic search** — find images matching a natural language description
2. **Distribution visualisation** — identify which types of scenes are over- or under-represented in a dataset
3. *(future)* **Deduplication** — remove near-duplicate frames to improve dataset diversity

---

## References

- [CLIP model on HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32)
- [ChromaDB documentation](https://docs.trychroma.com/)
- [UMAP documentation](https://umap-learn.readthedocs.io/)
- [Plotly Scatter](https://plotly.com/python/line-and-scatter/)