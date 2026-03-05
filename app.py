"""
CLIP + ChromaDB によるセマンティック画像検索
Step 3: Gradio WebUI版
"""

import gradio as gr
from PIL import Image

from src.config import DEVICE
from src.embedder import CLIPEmbedder
from src.store import ImageStore

print(f"Using device: {DEVICE}")
print("CLIPモデルをロード中...")
embedder = CLIPEmbedder()
store = ImageStore()
print("モデルロード完了")


def text_search(query: str, top_k: int = 4):
    """テキストクエリで画像を検索"""
    if not query.strip():
        return []

    emb = embedder.get_text_embedding(query)
    results = store.query(emb, n_results=top_k)

    output = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        similarity = 1 - dist
        img = Image.open(meta["path"]).convert("RGB")
        label = f"類似度: {similarity:.2%}\n{meta['caption']}"
        output.append((img, label))
    return output


def image_search(query_image, top_k: int = 4):
    """画像クエリで類似画像を検索"""
    if query_image is None:
        return []

    image = Image.fromarray(query_image).convert("RGB")
    emb = embedder.get_image_embedding(image)
    results = store.query(emb, n_results=top_k)

    output = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        similarity = 1 - dist
        img = Image.open(meta["path"]).convert("RGB")
        label = f"類似度: {similarity:.2%}\n{meta['caption']}"
        output.append((img, label))
    return output


# ---- Gradio UI ----
with gr.Blocks(
    title="🔍 セマンティック画像検索",
    theme=gr.themes.Soft(),
    css="""
    .header { text-align: center; padding: 20px 0; }
    .subtext { color: #666; font-size: 0.9em; }
    """
) as demo:
    gr.Markdown("""
    # 🔍 セマンティック画像検索
    **CLIP + ChromaDB** を使ったテキスト・画像検索のデモ
    """, elem_classes="header")

    with gr.Tabs():
        # --- テキスト検索タブ ---
        with gr.TabItem("📝 テキストで検索"):
            gr.Markdown("英語でシーンを説明すると、類似する画像を返します。", elem_classes="subtext")
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="検索クエリ（英語）",
                        placeholder="例: a dog running in a park",
                        lines=2
                    )
                    top_k_text = gr.Slider(1, 6, value=4, step=1, label="表示件数")
                    search_btn = gr.Button("🔍 検索", variant="primary")

                    gr.Markdown("**クエリ例:**")
                    gr.Examples(
                        examples=[
                            ["a car driving on the road"],
                            ["pedestrian walking on sidewalk"],
                            ["intersection with traffic"],
                            ["highway with multiple lanes"],
                            ["parked cars on street"],
                            ["urban street scene"],
                        ],
                        inputs=text_input
                    )

            text_results = gr.Gallery(
                label="検索結果",
                columns=4,
                height=320,
                show_label=True
            )
            search_btn.click(fn=text_search, inputs=[text_input, top_k_text], outputs=text_results)
            text_input.submit(fn=text_search, inputs=[text_input, top_k_text], outputs=text_results)

        # --- 画像検索タブ ---
        with gr.TabItem("🖼️ 画像で検索"):
            gr.Markdown("画像をアップロードすると、似ている画像を返します。", elem_classes="subtext")
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="クエリ画像をアップロード")
                    top_k_img = gr.Slider(1, 6, value=4, step=1, label="表示件数")
                    img_search_btn = gr.Button("🔍 類似画像を検索", variant="primary")

            img_results = gr.Gallery(
                label="検索結果",
                columns=4,
                height=320,
                show_label=True
            )
            img_search_btn.click(fn=image_search, inputs=[image_input, top_k_img], outputs=img_results)

    gr.Markdown("""
    ---
    **仕組み**: 画像は事前にCLIPでベクトル化してChromaDBに保存。検索時はテキスト/画像クエリをベクトル化し、コサイン類似度で最近傍を返します。
    """, elem_classes="subtext")

if __name__ == "__main__":
    demo.launch(share=False)
